from model_rag import *
from upload_docs import *
from vector_store import *
import json

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

