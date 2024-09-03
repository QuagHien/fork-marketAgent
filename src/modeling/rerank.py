import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modeling import AdaptiveModel, PytorchAdapter



class ReRank(AdaptiveModel):
    def __init__(self, model_path, *args, **kwargs):
        super().__init__(model_path, *args, **kwargs)
          
    def cls_pooling(self, model_output):
        if isinstance(self.adapter, PytorchAdapter):
            return model_output.logits.cpu().numpy()
        return model_output
    
    def encode(self, text, max_length):     
        tokenized = self.tokenizer_fn(text, max_length)
        model_opt = self.inference(tokenized)
        score = self.cls_pooling(model_opt)
        
        return score
        
    def tokenizer_fn(self, text, max_length):
        mode = 'pt' if isinstance(self.adapter, PytorchAdapter) else 'np'
        # Encoding pair sentence
        return self.adapter.tokenizer(
            text[0], text[1],
            max_length=max_length,
            truncation='only_first', 
            padding='max_length',
            return_tensors=mode
        )
        
if __name__ == '__main__':
    text = ["hello", "friend"]
    print("\nLoad Pytorch model: ")
    rerank_onnx = ReRank('full_weights/pytorch/mbert-rerank-base')
    logit = rerank_onnx.encode(text=text, max_length=128)
    print("logit: ", logit)
    
    print("\nLoad ONNX model: ")
    rerank_pytorch = ReRank('full_weights/onnx/mbert-rerank-onnx')
    logit = rerank_pytorch.encode(text=text, max_length=128)
    print("logit: ", logit)