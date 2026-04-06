from langchain.chat_models import init_chat_model
from langchain.tools import tool


OLLAMA_MODEL = "mistral-nemo:latest"

# Initialize the model
model = init_chat_model(model=OLLAMA_MODEL, model_provider="ollama", temperature=0.0)

# Wrap Tavily search with decorator
@tool
def analyze_configs(query: str) -> str:
    """Retrieve all security configuration policies. Returns policy names,
    descriptions, platforms and technologies so you can analyze which ones
    are relevant to the query."""
    import json
    import os

    path = os.path.join(os.path.dirname(__file__), "../configurations", "policies_and_settings_shorter.json")
    with open(path, "r") as f:
        configurations = json.load(f)

    return configurations

config_agent = {
    "name": "config_agent",
    "description": "Retrieves and analyzes Intune configuration policies.",
    "system_prompt": "You are a helpful security expert. To retrieve the security configurations, use the analyze_configs tool and search for specific policies.",
    "tools": [analyze_configs],
}