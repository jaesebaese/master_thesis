from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent
from langchain.tools import tool
from search_agent import search_agent
from config_agent import config_agent
from policy_agent import policy_agent

OLLAMA_MODEL = "mistral-nemo:latest"


# Initialize the model
model = init_chat_model(model=OLLAMA_MODEL, model_provider="ollama", temperature=0.0)

agent = create_deep_agent(
    model=model,
    subagents=[
        {"runnable": config_agent, "name": "config-agent", "description": "Retrieves and analyzes Intune configuration policies."},
        {"runnable": policy_agent, "name": "policy-agent", "description": "Looks up CIS/Microsoft baseline policy controls."},
        {"runnable": search_agent, "name": "search-agent", "description": "Searches the web for security best practices and vendor guidance."},
    ],
    system_prompt="""You are a security supervisor orchestrating a 
    policy drift analysis. Follow these steps in order:

    1. Send the user's input to policy-agent → get structured controls
    2. In parallel, send those controls to:
       - search-agent → get baseline recommendations  
       - config-agent → get actual configurations
    3. For each control, perform a 3-way comparison:
       - Policy intent vs. actual config      (compliance gap)
       - Best practice vs. actual config      (security gap)
       - Policy intent vs. best practice      (policy quality gap)
    4. Classify each deviation by severity
    5. Generate a remediation suggestion for each gap
    6. Present the full drift report and await human approval before any action

    IMPORTANT: Always delegate to subagents. Do not attempt to fetch Intune 
    data or search for standards yourself.""",
)

query = input("Ask the supervisor agent: ") 
result = agent.invoke({"messages": [{"role": "user", "content": query}]})
print(result["messages"][-1].content)