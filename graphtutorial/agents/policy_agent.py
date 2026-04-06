import os

from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent
from langchain.tools import tool

OLLAMA_MODEL = "mistral-nemo:latest"


# Initialize the model
model = init_chat_model(model=OLLAMA_MODEL, model_provider="ollama", temperature=0.0)


@tool
def policy_analyzer(query: str) -> str:
    """Analyze security policies"""
    path = os.path.join(os.path.dirname(__file__), "../policies", "password_policy.txt")
    with open(path, "r") as f:
        policy_data = f.read()
    print("Policy Analysis Result: " + str(policy_data))
    return str(policy_data)


policy_agent = {
    "name": "policy_agent",
    "description": "Looks up CIS/Microsoft baseline policy controls.",
    "system_prompt": "You are a helpful policy analyzer. Always use the policy_analyzer tool to evaluate security policies.",
    "tools": [policy_analyzer],
}