
import logging

logger = logging.getLogger(__name__)


class ModelBase:
    def load(self):
        pass

    def inference(self, x):
        pass


class AdaptiveModel:
    def __init__(self, model_path) -> None:
        if 'onnx' in model_path:
            self.adapter = ONNXAdapter(model_path)
        elif 'openvino' in model_path:
            self.adapter = OpenvinoAdapter(model_path)
        else:# 'pt' in model_path or 'safetensor' in model_path:
            self.adapter = PytorchAdapter(model_path)
            
        self.load()
        
    def load(self):
        self.adapter.load_model()
        
    def inference(self, x):
        # Forward the call to the adapter's inference method
        return self.adapter.inference(x)

        
class PytorchAdapter(ModelBase):
    import torch
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def __init__(self, model_path):
        self.model_path = model_path
        
    def load_model(self):
        if 'rank' in self.model_path:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            
        else:
            from transformers import AutoModel, AutoTokenizer
            self.model = AutoModel.from_pretrained(self.model_path)
            
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model.to(self.device)
        
        print('--- LOAD MODEL WITH PYTORCH CHECKPOINT ---')
        
    def inference(self, tokenized):
        with self.torch.no_grad():
            output = self.model(**tokenized.to(self.device), return_dict=True)
        return output
    
    
class OpenvinoAdapter(ModelBase):
    def __init__(self, model_path):
        self.model_path = model_path
    
    def load_model(self):
        return
    
    
class ONNXAdapter(ModelBase):
    def __init__(self, model_path):
        self.model_path = model_path
        
    def load_model(self, providers=[("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),
                              "CPUExecutionProvider"]):
        import os
        import onnxruntime as rt
        from hftokenizer import BertTokenizer

        _model_path = os.path.join(self.model_path, 'model.onnx')
        self.tokenizer = BertTokenizer(vocab_file=os.path.join(self.model_path, 'vocab.txt'))
        
        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = rt.InferenceSession(_model_path, sess_options, providers)
        self.input_names = [input.name for input in self.sess.get_inputs()]
        self.output_names = [output.name for output in self.sess.get_outputs()]
        
        print('--- LOAD MODEL WITH ONNX CHECKPOINT ---')

    def inference(self, x):
        assert hasattr(self, "sess"), "Model not loaded"
        assert len(x) == len(self.input_names), "Input shape mismatch"
        tokened = list(x.values())
        return self.sess.run(self.output_names, dict(zip(self.input_names, tokened)))[0]