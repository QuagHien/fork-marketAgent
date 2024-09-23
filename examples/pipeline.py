import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.chains import RetrievalQA

def chunking_recursive(docs, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    return chunks

def chunking_semantic(docs, embed_model):
    text_splitter = SemanticChunker(
        embeddings = embed_model,
        add_start_index = True,
        breakpoint_threshold_type = 'gradient',
    )
    chunks = text_splitter.split_documents(docs)
    return chunks

def load_documents_semantic(pdf_path, embed_model):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    chunks = chunking_semantic(docs, embed_model)
    ###
    chunks_as_dict = [
        {
            "page_content": chunk.page_content,
            "metadata": chunk.metadata
        } for chunk in chunks
    ]   
    json_file_path = os.path.splitext(pdf_path)[0] + '.json'
    
    with open(json_file_path, 'w', encoding='utf-8') as file:
        json.dump(chunks_as_dict, file, ensure_ascii=False, indent=4)
    print(f"Chunks have been saved to {json_file_path}.")
    ###
    return chunks

def load_documents(pdf_paths, chunk_size, chunk_overlap):
    # If the input is a single file path (string), convert it to a list
    if isinstance(pdf_paths, str):
        pdf_paths = [pdf_paths]

    all_chunks = []
    
    # Loop through all PDF files in the list
    for pdf_path in pdf_paths:
        try:
            print(f"Loading file: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            chunks = chunking_recursive(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            all_chunks.extend(chunks)

            # Optionally save the chunks to a JSON file
            chunks_as_dict = [
                {
                    "page_content": chunk.page_content,
                    "metadata": chunk.metadata
                } for chunk in chunks
            ]
            json_file_path = os.path.splitext(pdf_path)[0] + '.json'
            with open(json_file_path, 'w', encoding='utf-8') as file:
                json.dump(chunks_as_dict, file, ensure_ascii=False, indent=4)
            print(f"Chunks from {pdf_path} have been saved to {json_file_path}.")

        except Exception as e:
            print(f"Error loading file {pdf_path}: {e}")
    
    return all_chunks


def add_documents(qdrant_vecter_store, chunks):
    if qdrant_vecter_store is None:
        print("Error: qdrant_vecter_store is not initialized.")
        return  # Exit the function if qdrant_vecter_store does not exist
    docs_contents = []
    docs_metadatas = []
    for doc in chunks:
        if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
            docs_contents.append(doc.page_content)
            docs_metadatas.append(doc.metadata)
        else:
            print("Warning: Some documents do not have 'page_content' or 'metadata' attributes.")  
    qdrant_vecter_store.add_texts(texts=docs_contents, metadatas=docs_metadatas)

def connect_qdrant(url, api_key):
    client = QdrantClient(url=url, api_key=api_key)
    try:
        collections = client.get_collections()
        print("Successfully connected to Qdrant. Existing collections:")
        for collection in collections.collections:
            print(f"- {collection.name}")
    except Exception as e:
        print(f"Failed to connect to Qdrant: {e}")
        client = None
    return client


def init_embed_model(model_name):
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    embedding_dim = len(embedding_model.embed_query("test"))
    return embedding_model, embedding_dim

def create_or_recreate_collection(client, collection_name, embedding_dim):
    try:
        collections = client.get_collections().collections
        
        # Check if the collection already exists
        if collection_name in [collection.name for collection in collections]:
            print(f"Collection '{collection_name}' already exists.")
            
            # Ask the user if they want to delete and recreate the collection
            choice = input(f"Do you want to delete and recreate the collection '{collection_name}'? (y/n): ").lower()
            
            if choice == 'y':
                # If the user selects 'y', delete the existing collection
                try:
                    client.delete_collection(collection_name)
                    print(f"Collection '{collection_name}' has been deleted.")
                except Exception as e:
                    print(f"Failed to delete collection '{collection_name}': {e}")
                    return

                # Attempt to recreate the collection after deletion
                try:
                    client.create_collection(
                        collection_name=collection_name,
                        vectors_config=models.VectorParams(size=embedding_dim, distance=models.Distance.COSINE)
                    )
                    print(f"Collection '{collection_name}' has been recreated.")
                except Exception as e:
                    print(f"Could not recreate collection '{collection_name}': {e}")
            else:
                print("Operation cancelled. The collection was not recreated.")
        else:
            # Create a new collection if it doesn't exist
            try:
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(size=embedding_dim, distance=models.Distance.COSINE)
                )
                print(f"Collection '{collection_name}' has been created.")
            except Exception as e:
                print(f"Could not create collection '{collection_name}': {e}")
    
    except Exception as e:
        print(f"Error checking collections: {e}")


def create_qdrant_vector_store(client_qdrant, collection_name, embedding_model):
    qdrant_vecter_store = QdrantVectorStore(
        client=client_qdrant,
        collection_name=collection_name,
        embedding=embedding_model,
    ) 
    return qdrant_vecter_store


def init_llm(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        truncation=True,
        device=device  # 0 for GPU, -1 for CPU
        # temperature,top_k,top_p,do_sample=True,
    )
    llm = HuggingFacePipeline(pipeline=llm_pipeline)
    return llm


def create_qa_chain(llm, retriever, top_k):
    retriever = retriever.as_retriever(search_kwargs={"k": top_k})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )
    return qa_chain


        
if __name__ == "__main__":
    pdf_path = ["fork-marketAgent/examples/DIAMOND _ 2K.pdf","fork-marketAgent/examples/Mentorship PLATINUM _10K.pdf"]
    qdrant_url = "https://43b556ed-6be2-4e05-85d2-c3a2cfe503b2.europe-west3-0.gcp.cloud.qdrant.io:6333"
    qdrant_api_key = "WUq64bGlo1HPkKRzByEK0JVGjzaTdU-SNnOfkrNdjw_Up8b6iY4NZQ"
    embedding_model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" # ****dim 768
                        #"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" ***dim 384
    llm_model_name = "bigscience/bloomz-1b7"
                    #"bigscience/bloomz-7b1"
                    
    collection_name = "qdrant-test"
    device = -1  # Use 0 if running on GPU

    
    # Step 1: Connect to Qdrant
    client = connect_qdrant(qdrant_url, qdrant_api_key)

    # Step 2: Initialize embedding model
    embedding_model, embedding_dim = init_embed_model(embedding_model_name)

    # Step 3: Create or recreate collection
    create_or_recreate_collection(client, collection_name, embedding_dim)

    # Step 4: Create Qdrant vector store and add documents
    qdrant = create_qdrant_vector_store(client, collection_name, embedding_model)

    # Step 5: Load and process documents
    chunks = load_documents(pdf_path, chunk_size=512, chunk_overlap=100)

    # Step 6: Add chunks to vector store
    add_documents(qdrant, chunks)

    # Step 7: Initialize language model
    llm = init_llm(llm_model_name, device)

    # Step 8: Create QA chain
    qa_chain = create_qa_chain(llm, qdrant, top_k=3)

    # Step 9: Example query
    query = "Chi phí tham gia Chương trình Mentorship PLATINUM của BeQ Holdings là bao nhiêu?"
    answer = qa_chain.invoke(query)
    print("My RAG's answer:")

    # Extract the content after '\nHelpful Answer:' in answer['result']
    helpful_answer = answer['result'].split('\nHelpful Answer:')[-1].strip()

    # Print only the helpful answer
    print(helpful_answer)
    
    # Save the entire content of the answer variable to a JSON file
    json_file_path = 'ANSWER_RAG.json'
    with open(json_file_path, 'w', encoding='utf-8') as file:
        json.dump(answer, file, ensure_ascii=False, indent=4)

    print(f"Answer has been saved to {json_file_path}.")
