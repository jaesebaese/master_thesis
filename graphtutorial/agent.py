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



# Using Gemini
#from langchain_google_genai import ChatGoogleGenerativeAI
#model = ChatGoogleGenerativeAI(model="gemini-3-pro-preview")

agent = create_deep_agent(
    model=model,
    tools=[tavily_search],
    system_prompt="You are a helpful assistant. Always use the tavily_search tool for any questions about current events, news, or real-time information."
)

query = input("Ask the agent: ") 
result = agent.invoke({"messages": [{"role": "user", "content": query}]})
print(result["messages"][-1].content)