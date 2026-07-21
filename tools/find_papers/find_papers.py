import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

# Initialize embeddings and vector store
embeddings = GoogleGenerativeAIEmbeddings(model=os.getenv("EMBEDDING_MODEL"), google_api_key=os.getenv("GOOGLE_API_KEY"))
vectorstore = Chroma(persist_directory=os.getenv("AGENT_DATA_ROOT") + "/chroma_db", embedding_function=embeddings)

@tool
def find_relevant_papers(query: str) -> str:
    """
    Finds the top 10 most relevant papers by searching only the first pages 
    of all documents in the knowledge base.
    
    Args:
        query: The search query string.
        
    Returns:
        A string listing the top 10 most relevant papers.
    """
    try:
        # Perform similarity search filtered by page 0
        # Chroma performs pre-filtering: it narrows the search space to page 0 first.
        results = vectorstore.similarity_search(
            query, 
            k=10, 
            filter={"page": 0}
        )
        
        if not results:
            return "No papers found."
            
        result = "Top 10 relevant papers based on first page:\n"
        for i, doc in enumerate(results):
            source = os.path.basename(doc.metadata.get('source', 'Unknown'))
            result += f"{i+1}. {source}\n"
            
        return result
    except Exception as e:
        return f"Error finding papers: {str(e)}"
