import sys
import os
import time
from functools import partial
import numpy as np

import fastapi
from fastapi import Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

from utils import client

from modeling.retrieve import RetrievaltModel
from modeling.rerank import ReRank

# Load environment variables
#
model_name = os.getenv("MODEL_NAME")
model_version = os.getenv("MODEL_VERSION", "")
batch_size = int(os.getenv("BATCH_SIZE", 1))
#
url = os.getenv("TRITON_URL", "localhost:8000")
protocol = os.getenv("PROTOCOL", "HTTP")
verbose = os.getenv("VERBOSE", "False").lower() in ("true", "1", "t")
async_set = os.getenv("ASYNC_SET", "False").lower() in ("true", "1", "t")



# Initialize Triton client
#
try:
    if protocol.lower() == "grpc":
        triton_client = grpcclient.InferenceServerClient(url=url, verbose=verbose)
    else:
        concurrency = 20 if async_set else 1
        triton_client = httpclient.InferenceServerClient(url=url, verbose=verbose, concurrency=concurrency)
except Exception as e:
    print("Client creation failed: " + str(e))
    sys.exit(1)



# Retrieve model metadata and config
#
try:
    model_metadata = triton_client.get_model_metadata(model_name=model_name, model_version=model_version)
    model_config = triton_client.get_model_config(model_name=model_name, model_version=model_version)
except InferenceServerException as e:
    print("Failed to retrieve model metadata: " + str(e))
    sys.exit(1)



# Adjust model config based on protocol
#
if protocol.lower() == "grpc":
    model_config = model_config.config
else:
    model_metadata, model_config = client.convert_http_metadata_config(model_metadata, model_config)

# Parse model metadata and config
max_batch_size, input_names, output_name, formats, dtypes = client.parse_model(model_metadata, model_config)


# Check model_name to initialize the corresponding model
if 'retrieve' in model_name.lower():
    print("Initializing Retrieval Model")
    token_model = RetrievaltModel('model_repository/mbert-retrieve-qry-onnx/1')
elif 'rerank' in model_name.lower():
    print("Initializing Rerank Model")
    token_model = ReRank('model_repository/mbert-rerank-onnx/1')
else:
    print(f"Model name '{model_name}' not recognized for either retrieval or rerank.")
    sys.exit(1)



############
# FastAPI
############

app = fastapi.FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    # Render the form with one input field
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process/", response_class=HTMLResponse)
async def process(request: Request, text1: str = Form(...)):
    # Tokenize single input text
    tokenized_inputs = token_model.tokenizer_fn(text=text1, max_length=32)

    # Prepare input data for Triton
    input_data = {name: tokenized_inputs[name] for name in input_names if name in tokenized_inputs}

    # Generate Triton input objects
    inputs, outputs = request_generator(input_data, input_names, output_name, dtypes)

    # Perform inference
    try:
        start_time = time.time()

        if protocol.lower() == "grpc":
            user_data = client.UserData()
            embeddings = triton_client.async_infer(
                model_name, inputs, partial(client.completion_callback, user_data), model_version=model_version, outputs=outputs
            )
        else:
            async_request = triton_client.async_infer(
                model_name, inputs, model_version=model_version, outputs=outputs
            )
    except InferenceServerException as e:
        return templates.TemplateResponse("index.html", {"request": request, "result": "Inference failed with error: " + str(e)})

    # Collect results from the ongoing async requests
    if protocol.lower() == "grpc":
        (embeddings, error) = user_data._completed_requests.get()
        if error is not None:
            return templates.TemplateResponse("index.html", {"request": request, "result": "Inference failed with error: " + str(error)})
    else:
        embeddings = async_request.get_result()

    # Process the results
    end_time = time.time()
    print("Process time: ", end_time - start_time)

    result = embeddings.as_numpy(output_name).tolist()
    return templates.TemplateResponse("index.html", {"request": request, "result": result})



###################
# Helper functions
###################

def request_generator(tokenized_inputs, input_names, output_name, dtypes):
    # Define the protocol
    client_type = grpcclient if protocol.lower() == "grpc" else httpclient

    # Create list of inputs for Triton
    inputs = []
    for input_name, dtype in zip(input_names, dtypes):
        input_data = tokenized_inputs[input_name]  # Access the correct input data by name
        infer_input = client_type.InferInput(input_name, input_data.shape, dtype)
        infer_input.set_data_from_numpy(input_data)
        inputs.append(infer_input)

    outputs = [client_type.InferRequestedOutput(output_name)]
    
    return inputs, outputs
