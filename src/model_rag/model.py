from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def init_embed_model(model_name, device='cpu', normalize_embeddings=False):
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': normalize_embeddings}
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs = model_kwargs,
        encode_kwargs = encode_kwargs
    )
    embedding_dim = len(embedding_model.embed_query("test"))
    return embedding_model, embedding_dim

def init_llm(model_name, device = -1, max_length = 1024 ):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        truncation=True,
        device=device  # 0 for GPU, -1 for CPU
        #temperature,top_k,top_p,do_sample=True,
    )
    llm = HuggingFacePipeline(pipeline=llm_pipeline)
    return llm