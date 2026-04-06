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
    """Search the web using Tavily"""
    response = client.search(
        query=query,
        max_results=3,
    )
    print(str(response))
    return str(response)

search_agent = {
    "name": "search_agent",
    "description": "Searches the web for security best practices and vendor guidance.",
    "system_prompt": "You are a helpful search assistant. Always use the tavily_search tool to research best practices, and vendor-specific information.",
    "tools": [tavily_search],
}