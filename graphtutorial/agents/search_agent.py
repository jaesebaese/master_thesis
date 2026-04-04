from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent
from langchain.tools import tool
from tavily import TavilyClient
from config import OLLAMA_MODEL, TAVILY_API_KEY

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


search_agent = create_deep_agent(
    model=model,
    tools=[tavily_search],
    system_prompt="You are a helpful search assistant. Always use the tavily_search tool to research best practices, and vendor-specific information."
)

if __name__ == "__main__":
    query = input("Ask the search agent: ")
    result = search_agent.invoke({"messages": [{"role": "user", "content": query}]})
    print(result["messages"][-1].content)