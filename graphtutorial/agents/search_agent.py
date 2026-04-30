from langchain.chat_models import init_chat_model
from langchain.tools import tool
from tavily import TavilyClient
from dotenv import load_dotenv
import os


OLLAMA_MODEL = "llama3.1:latest"

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Initialize the model
model = init_chat_model(model=OLLAMA_MODEL, model_provider="ollama", temperature=0.0)

# Tavily search (requires TAVILY_API_KEY)
client = TavilyClient(api_key=TAVILY_API_KEY)

@tool
def tavily_search_specific_configurations(query: str) -> str:
    """Search Microsoft documentation for specific Intune configuration guidance."""
    response = client.search(
        query=query,
        max_results=3,
        include_domains=["learn.microsoft.com/"],
    )
    return str(response)


search_agent = {
    "name": "search_agent",
    "description": "Searches the web for security best practices and vendor guidance.",
    "system_prompt": (
        "You are a Microsoft documentation search specialist. "
        "Your role is to find vendor-specific guidance and best practice "
        "recommendations from official Microsoft sources.\n\n"

        "## How to search\n"
        "1. Always call tavily_search_specific_configurations with the user's query.\n"
        "2. If the first search returns fewer than 2 relevant results, "
        "try a more specific reformulated query.\n"
        "3. Never answer from your own knowledge — only from search results.\n\n"

        "## Output format\n"
        "For each relevant result found:\n"
        "- State the recommendation clearly in one or two sentences.\n"
        "- Cite the source URL.\n"
        "- Note the recommended configuration value if specified.\n\n"

        "## Confidence and sourcing\n"
        "Always end your response with:\n"
        "SOURCE: web_search | learn.microsoft.com\n"
        "CONFIDENCE: medium\n"
        "NOTE: This recommendation is sourced from live web search and "
        "should be verified by a human reviewer before applying.\n\n"

        "## If no results found\n"
        "State explicitly: 'No Microsoft documentation found for this "
        "query. Human review required before proceeding.'\n"
        "Never generate a recommendation when search returns nothing."
    ),
    "tools": [tavily_search_specific_configurations],
}

""" if __name__ == "__main__":
    # result = tavily_search.invoke("Bitlocker policies for Windows devices")
    result = tavily_search_specific_configurations.invoke("device_vendor_msft_bitlocker_fixeddrivesrecoveryoptions")
    print(result) """