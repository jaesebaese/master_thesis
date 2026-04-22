from langchain.chat_models import init_chat_model
from langchain.tools import tool
from tavily import TavilyClient
from dotenv import load_dotenv
import os


OLLAMA_MODEL = "mistral-nemo:latest"

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Initialize the model
model = init_chat_model(model=OLLAMA_MODEL, model_provider="ollama", temperature=0.0)

# Tavily search (requires TAVILY_API_KEY)
client = TavilyClient(api_key=TAVILY_API_KEY)

# Wrap Tavily search with decorator
@tool
def tavily_search(query: str) -> str:
    """Search Microsoft documentation for security best practices and Intune guidance."""
    response = client.search(
        query=query,
        max_results=3,
        include_domains=["learn.microsoft.com/en-us/intune/"],
    )
    return str(response)
@tool
def tavily_search_specific_configurations(query: str) -> str:
    """Search Microsoft documentation for specific Intune configuration guidance."""
    response = client.search(
        query=query,
        max_results=3,
        include_domains=["learn.microsoft.com/en-us/intune/"],
    )
    return str(response)


search_agent = {
    "name": "search_agent",
    "description": "Searches the web for security best practices and vendor guidance.",
    "system_prompt": """You are a helpful search assistant. 
        Always use the tavily_search tool to research best practices, and vendor-specific information. 
        It is important to only return information from trusted sources such as Microsoft documentation. 
        Always use the tavily_search tool to perform searches, and never attempt to answer based on your own knowledge or make up sources. 
        Search for specific settings, controls, and recommendations related to the user's query.
        """,
    "tools": [tavily_search],
}

if __name__ == "__main__":
    # result = tavily_search.invoke("Bitlocker policies for Windows devices")
    result = tavily_search_specific_configurations.invoke("device_vendor_msft_bitlocker_fixeddrivesrecoveryoptions")
    print(result)