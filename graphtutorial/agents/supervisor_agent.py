from langchain.chat_models import init_chat_model
from langchain.agents.middleware import wrap_tool_call
from deepagents import create_deep_agent
from search_agent import search_agent
from config_agent import config_agent
from policy_agent import policy_agent
from cis_benchmark_agent import cis_benchmark_agent
from typing import Any
from langchain.agents import AgentState 
import asyncio
from agent_utils import stream_agent, format_messages

OLLAMA_MODEL = "llama3.1:latest"

model = init_chat_model(model=OLLAMA_MODEL, model_provider="ollama", temperature=0.0)

""" class AgentLogger(BaseCallbackHandler):
    Prints a readable trace of every agent event to stdout.

    def on_chain_start(self, serialized: dict, _inputs: dict, **kwargs: Any) -> None:
        serialized = serialized or {}
        name = serialized.get("name") or (serialized.get("id") or ["?"])[-1]
        run_id = str(kwargs.get("run_id", ""))[:8]
        parent = str(kwargs.get("parent_run_id", ""))[:8]
        parent_str = f"  (parent: {parent})" if parent else ""
        print(f"\n[Chain start]  {name}  run={run_id}{parent_str}")

    def on_chain_end(self, outputs: dict, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id", ""))[:8]
        print(f"[Chain end]    run={run_id}")

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs: Any) -> None:
        model_name = serialized.get("name", "LLM")
        run_id = str(kwargs.get("run_id", ""))[:8]
        print(f"\n[LLM start]  {model_name}  run={run_id}")

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id", ""))[:8]
        # Extract the text of the first generation if available
        try:
            text = response.generations[0][0].text[:200]
            print(f"[LLM end]    run={run_id}  text: {text}{'…' if len(response.generations[0][0].text) > 200 else ''}")
        except Exception:
            print(f"[LLM end]    run={run_id}")

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs: Any) -> None:
        tool_name = serialized.get("name", "tool")
        run_id = str(kwargs.get("run_id", ""))[:8]
        parent = str(kwargs.get("parent_run_id", ""))[:8]
        truncated = input_str[:300] + ("…" if len(input_str) > 300 else "")
        print(f"\n[Tool call]  {tool_name}  run={run_id}  (parent: {parent})")
        print(f"   Input: {truncated}")

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id", ""))[:8]
        truncated = str(output)[:300] + ("…" if len(str(output)) > 300 else "")
        print(f"   Output: {truncated}")
        print(f"[Tool end]   run={run_id}")

    def on_tool_error(self, error: Exception, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id", ""))[:8]
        print(f"[Tool error] run={run_id}  {error}")

    def on_agent_action(self, action: Any, **kwargs: Any) -> None:
        print(f"\n[Action]  {action.tool}  —  {str(action.tool_input)[:200]}")

    def on_agent_finish(self, finish: Any, **kwargs: Any) -> None:
        output = str(finish.return_values)[:200]
        print(f"\n[Agent finish]  {output}") """

@wrap_tool_call
def tool_logger(request, handler):
    name = request.tool_call["name"]
    args = request.tool_call["args"]
    print(f"\n[Middleware] Tool call: {name}")
    print(f"[Middleware] Args: {args}")
    result = handler(request)
    content = result.content if hasattr(result, "content") else str(result)
    truncated = content[:300] + ("..." if len(content) > 300 else "")
    print(f"[Middleware] Result: {truncated}")
    return result


agent = create_deep_agent(
    model=model,
    middleware=[tool_logger],
    subagents=[config_agent, policy_agent, search_agent, cis_benchmark_agent],
    system_prompt= """
    You are a Microsoft Intune security supervisor. You orchestrate specialised subagents — never answer from your own knowledge alone.

    Create a task for each subagent by delegating to the appropriate subagent(s) to get the information you need to answer the user's question. 

    Call the subagents and delegate the work to them and use the information they return to construct your answer to the user.

    Always use EVERY subagent that is relevant to the question. Do not leave any relevant subagents out of your analysis.

    Use the following process to answer questions about security posture and compliance:
        Step 1 → delegate to the policy_agent to find relevant policies and settings for the topic.
        Step 2 → delegate to config_agent to get current configured values for the topic.
        Step 3 → delegate to cis_benchmark_agent to get CIS Benchmark recommendations for the configured settings.
        Step 4 → delegate to search_agent to get Microsoft recommendations for any settings not covered by the CIS benchmark.
        Step 5 → Present as table:
                Setting | Configured | Recommended | Status | Dependencies
                Status values: COMPLIANT / NON-COMPLIANT / NOT CONFIGURED
                For each non-compliant setting:
                - State the remediation path from CIS data
                - Note any interdependency warnings from Step 4
                - Flag if web search was used (medium confidence)
        Step 6 → Give a recommendation after having considered all the data and dependencies received from the subagents. End with a security posture summary paragraph.
""",

)

query = input("Ask the supervisor agent: ")
result = agent.invoke({"messages": [{"role": "user", "content": query}], "files": {"password_policy.txt": open("/home/hochuli/project/master_thesis/graphtutorial/agents/password_policy.txt").read()}})
#for chunk in agent.stream({"messages": [{"role": "user", "content": query}], "files": {"password_policy.txt": open("/home/hochuli/project/master_thesis/graphtutorial/agents/password_policy.txt").read()}}):
    #print(chunk)
formatted_result = format_messages(result["messages"])