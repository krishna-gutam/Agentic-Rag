import os
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

DOCS_DIR = os.path.join(os.getenv("AGENT_DATA_ROOT"), "docs")

@tool
def list_papers() -> str:
    """
    Lists all PDF papers currently available in the knowledge base.
    
    Returns:
        A string containing a list of filenames of the available PDF papers.
    """
    try:
        files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")]
        if not files:
            return "No papers found in the knowledge base."
        return "Available papers:\n" + "\n".join(files)
    except Exception as e:
        return f"Error listing papers: {str(e)}"
