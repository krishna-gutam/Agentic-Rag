import os
import time
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# Configuration
DOCS_DIR = Path(os.getenv("AGENT_DATA_ROOT")) /"docs"
PERSIST_DIR = Path(os.getenv("AGENT_DATA_ROOT"))/"chroma_db"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

def build_db():
    # 1. Load documents
    # Get existing files from DB
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=os.getenv("GOOGLE_API_KEY"))
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    
    existing_docs = vectorstore.get(include=['metadatas'])
    existing_sources = set()
    if existing_docs and 'metadatas' in existing_docs:
        for meta in existing_docs['metadatas']:
            if 'source' in meta:
                existing_sources.add(os.path.basename(meta['source']))
    
    documents = []
    for filename in os.listdir(DOCS_DIR):
        if filename.endswith(".pdf") and filename not in existing_sources:
            file_path = os.path.join(DOCS_DIR, filename)
            print(f"Loading {filename}...")
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        elif filename.endswith(".pdf"):
            print(f"Skipping {filename}, already in database.")

    if not documents:
        print("No new documents to add.")
        return

    # 2. Split documents (by page)
    # PyPDFLoader already loads documents page-by-page.
    # We simply use the loaded documents directly.
    docs = documents
    print(f"Loaded {len(docs)} pages.")

    # 3. Embed and store
    print("Embedding and storing in Chroma...")
    
    # Add documents in batches to avoid rate limits
    batch_size = 1
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(docs)-1)//batch_size + 1}...")
        vectorstore.add_documents(batch)
        time.sleep(4) # Wait to avoid rate limits
        
    print("Database updated successfully.")

if __name__ == "__main__":
    build_db()

