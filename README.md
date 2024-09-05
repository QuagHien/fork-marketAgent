## Setup Triton Server

Clone github
```
  git clone https://github.com/tien-ngnvan/fork-marketAgent.git
  cd fork-marketAgent
```
Download models and move 'model.onnx' files from downloaded model folders to model_repository/<'Corresponding model name folder'> 
```
  gdown 1IaCcVfB9ibZi52WfvTg7FKihSAKlQ9uZ
```
Run container docker triton server (24.08) with CPU:
```
	docker run --rm --net=host -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:24.08-py3 tritonserver --model-repository=/models
```
Setup:
```
	export MODEL_NAME=mbert-retrieve-qry-onnx        #   mbert-rerank-onnx 
	export MODEL_VERSION=1																
	export BATCH_SIZE=1
	export TRITON_URL=localhost:8000                 #   localhost:8001  (if gRPC)
	export PROTOCOL=HTTP                             #   gRPC
	export VERBOSE=True                              #   show more details
	export ASYNC_SET=True                            #   asynchronous handling of multiple requests
```
Run inference:
```
	uvicorn app:app --host 0.0.0.0 --port 8003	
```
