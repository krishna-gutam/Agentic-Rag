import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Configuration
PERSIST_DIR = os.path.join(os.getenv("AGENT_DATA_ROOT"), "chroma_db")
DOCS_DIR = os.path.join(os.getenv("AGENT_DATA_ROOT"), "docs")

# Initialize embeddings
api_key = os.getenv("GOOGLE_API_KEY")
embedding_model = os.getenv("EMBEDDING_MODEL")
embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model, google_api_key=api_key)

# Initialize vector store
vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings
)

# Cache for BM25 retrievers
bm25_cache = {}

def get_bm25_for_file(file_path: str):
    if file_path not in bm25_cache:
        loader = PyPDFLoader(file_path)
        docs = loader.load()  # PyPDFLoader.load() loads each page as a Document
        retriever = BM25Retriever.from_documents(docs)
        retriever.k = 3
        bm25_cache[file_path] = retriever
    return bm25_cache[file_path]

@tool
def paper_rag_search(query: str, filename: str) -> str:
    """
    Performs a hybrid search (vector + BM25) restricted to a specific document.
    
    Args:
        query: The search query string.
        filename: The exact name of the PDF file (e.g., 'attention_is_all_you_need.pdf').
        
    Returns:
        A string containing the concatenated content of the most relevant 
        document chunks found within the specified file.
    """
    try:
        file_path = os.path.join(DOCS_DIR, filename)
        if not os.path.exists(file_path):
            return f"File '{filename}' not found in {DOCS_DIR}."

        # 1. Vector Retriever with filter
        filter_dict = {"source": file_path}
        vector_retriever = vectorstore.as_retriever(
            search_kwargs={"k": 3, "filter": filter_dict}
        )
        
        # 2. BM25 Retriever (cached)
        try:
            bm25_retriever = get_bm25_for_file(file_path)
        except Exception as e:
            return f"Failed to initialize BM25 retriever for {filename}: {str(e)}"
        
        # 3. Ensemble
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.5, 0.5]
        )
        
        results = ensemble_retriever.invoke(query)
        if not results:
            return f"No relevant information found in {filename} for query: '{query}'."
        return "\n\n".join([doc.page_content for doc in results])
    except Exception as e:
        return f"An unexpected error occurred while searching {filename}: {type(e).__name__} - {str(e)}"
