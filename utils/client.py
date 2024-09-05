import sys

if sys.version_info >= (3, 0):
    import queue
else:
    import Queue as queue


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


# Callback function used for async_stream_infer()
def completion_callback(user_data, result, error):
    # passing error raise and handling out
    user_data._completed_requests.put((result, error))


# def parse_model(model_metadata, model_config):
#     """
#     Check the configuration of a model to make sure it meets the
#     requirements for an image classification network (as expected by
#     this client)
#     """
#     if len(model_metadata.inputs) != 1:
#         raise Exception(
#             "expecting 1 input, got {}".format(
#                 len(model_metadata.inputs)
#             )
#         )
#     if len(model_metadata.outputs) != 1:
#         raise Exception(
#             "expecting 1 output, got {}".format(
#                 len(model_metadata.outputs)
#             )
#         )

#     if len(model_config.input) != 1:
#         raise Exception(
#             "expecting 1 input in model configuration, got {}".format(
#                 len(model_config.input)
#             )
#         )

#     input_metadata = model_metadata.inputs[0]
#     input_config = model_config.input[0]
#     output_metadata = model_metadata.outputs[0]
    

#     if output_metadata.datatype != "FP32":
#         raise Exception(
#             "expecting output datatype to be FP32, model '"
#             + model_metadata.name
#             + "' output type is "
#             + output_metadata.datatype
#         )

#     # Model input must have 3 dims, either CHW or HWC (not counting
#     # the batch dimension), either CHW or HWC
#     input_batch_dim = model_config.max_batch_size > 0
#     expected_input_dims = 2 if input_batch_dim else 1
#     if len(input_metadata.shape) != expected_input_dims:
#         raise Exception(
#             "expecting input to have {} dimensions, model '{}' input has {}".format(
#                 expected_input_dims, model_metadata.name, len(input_metadata.shape)
#             )
#         )

#     return (
#         model_config.max_batch_size,
#         input_metadata.name,
#         output_metadata.name,
#         input_config.format,
#         input_metadata.datatype,
#     )
def parse_model(model_metadata, model_config):
    """
    Check the configuration of a model and retrieve necessary details
    """
    # Check the number of inputs and outputs
    if len(model_metadata.inputs) < 1:
        raise Exception(
            "expecting at least 1 input, got {}".format(
                len(model_metadata.inputs)
            )
        )
    if len(model_metadata.outputs) != 1:
        raise Exception(
            "expecting 1 output, got {}".format(
                len(model_metadata.outputs)
            )
        )

    # Get the first input and output for default processing
    input_metadata = model_metadata.inputs[0]
    output_metadata = model_metadata.outputs[0]

    # Log the number of inputs and outputs
    print(f"Input names: {[input.name for input in model_metadata.inputs]}")
    print(f"Output name: {output_metadata.name}")

    # Check the data type of the output
    if output_metadata.datatype != "FP32":
        raise Exception(
            "expecting output datatype to be FP32, model '"
            + model_metadata.name
            + "' output type is "
            + output_metadata.datatype
        )

    return (
        model_config.max_batch_size,
        [input.name for input in model_metadata.inputs],  # List of input names
        output_metadata.name,
        [input.format for input in model_config.input],   # List of input formats
        [input.datatype for input in model_metadata.inputs],  # List of input data types
    )




def convert_http_metadata_config(_metadata, _config):
    # NOTE: attrdict broken in python 3.10 and not maintained.
    # https://github.com/wallento/wavedrompy/issues/32#issuecomment-1306701776
    try:
        from attrdict import AttrDict
    except ImportError:
        # Monkey patch collections
        import collections
        import collections.abc

        for type_name in collections.abc.__all__:
            setattr(collections, type_name, getattr(collections.abc, type_name))
        from attrdict import AttrDict

    return AttrDict(_metadata), AttrDict(_config)