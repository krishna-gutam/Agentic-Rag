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
PERSIST_DIR = "./chroma_db"
DOCS_DIR = "./docs"

# Initialize embeddings
api_key = os.getenv("GOOGLE_API_KEY")
embedding_model = os.getenv("EMBEDDING_MODEL")
embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model, google_api_key=api_key)

# Initialize vector store
vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings
)

# Initialize BM25 Retriever
def get_bm25_retriever():
    documents = []
    for filename in os.listdir(DOCS_DIR):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DOCS_DIR, filename))
            documents.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    return BM25Retriever.from_documents(docs)

bm25_retriever = get_bm25_retriever()
bm25_retriever.k = 3

# Initialize Ensemble Retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[vectorstore.as_retriever(search_kwargs={"k": 3}), bm25_retriever],
    weights=[0.5, 0.5]
)

@tool
def rag_search(query: str) -> str:
    """
    Performs a hybrid search to retrieve relevant context from the knowledge base.
    
    This tool combines semantic vector search (for conceptual understanding) 
    and BM25 keyword search (for exact term matching) to provide the most 
    accurate and relevant information for the given query.
    
    Args:
        query: The search query string to look up in the knowledge base.
        
    Returns:
        A string containing the concatenated content of the most relevant 
        document chunks found.
    """
    try:
        docs = ensemble_retriever.invoke(query)
        if not docs:
            return "No relevant documents found."
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        return f"Error searching database: {str(e)}"
