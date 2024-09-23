from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
import os
import json

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
    try:
        qdrant_vecter_store.add_texts(texts=docs_contents, metadatas=docs_metadatas)
        print(f"Successfully added documents to Qdrant.")
    except Exception as e:
        print(f"Error while adding documents to Qdrant: {e}")