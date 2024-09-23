from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_qdrant import QdrantVectorStore
from langchain.chains import RetrievalQA

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
    print(f"Qdrant vector store for collection '{collection_name}' created successfully.")
    return qdrant_vecter_store


def create_qa_chain(llm, retriever, top_k):
    """
    Create a question-answering (QA) chain with a retriever and a language model (LLM).
    
    Args:
        llm: The language model to use for generating answers.
        retriever: The retriever to search for relevant documents.
        top_k: The number of top results to retrieve from the document store.
    
    Returns:
        A QA chain object configured with the retriever and LLM.
    """
    # Convert the retriever into a LangChain-compatible retriever
    retriever = retriever.as_retriever(search_kwargs={"k": top_k})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )
    return qa_chain