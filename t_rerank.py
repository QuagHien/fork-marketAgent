from marketAgent.src.modeling.rerank import ReRank

text = ["hello", "friend"]
rerank_pytorch = ReRank('model_repository/mbert-rerank-onnx/1')
logit = rerank_pytorch.encode(text=text, max_length=128)
print("logit: ", logit)