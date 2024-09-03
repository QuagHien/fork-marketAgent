import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modeling import AdaptiveModel, PytorchAdapter



class RetrievaltModel(AdaptiveModel):
    def __init__(self, model_path, *args, **kwargs):
        super().__init__(model_path, *args, **kwargs)
        
    def cls_pooling(self, model_output):
        if isinstance(self.adapter, PytorchAdapter):
            return model_output.last_hidden_state[:,0].cpu().numpy()
        return model_output[:, 0]
    
    def encode(self, text, max_length):          
        tokenized = self.tokenizer_fn(text, max_length)
        model_opt = self.inference(tokenized)
        embedding = self.cls_pooling(model_opt)
        
        return embedding
        
    def tokenizer_fn(self, text, max_length):
        mode = 'pt' if isinstance(self.adapter, PytorchAdapter) else 'np'
        return self.adapter.tokenizer(text, 
                                      padding=True, 
                                      truncation=True,
                                      max_length=max_length,
                                      return_tensors=mode)
    

if __name__ == '__main__':
    text = "hello friend"
    print("\nLoad Pytorch model: ")
    context_pytorch = RetrievaltModel('full_weights/pytorch/mbert-retrieve-ctx-base')
    ctx_embd = context_pytorch.encode(text=text, max_length=32)
    print("CTX_pytorch_embd: ", ctx_embd.shape)
    
    print("\nLoad ONNX model: ")
    context_onnx = RetrievaltModel('full_weights/onnx/mbert-retrieve-ctx-onnx')
    ctx_embd = context_onnx.encode(text=text, max_length=32)
    print("CTX_onnx_embd: ", ctx_embd.shape)
    
    # print("\nQuery model: ")
    # context_model = ContextModel('full_weights/onnx/mbert-retrieve-qry-onnx')
    