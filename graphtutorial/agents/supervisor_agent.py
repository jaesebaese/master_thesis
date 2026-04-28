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
    system_prompt="""You are a Microsoft Intune security supervisor. You orchestrate
specialised subagents — never answer from your own knowledge alone.


You have access to the `write_todos` tool to help you manage and plan complex objectives.\nUse this tool for complex objectives to ensure that you are tracking each necessary step and giving the user visibility into your progress.\nThis tool is very helpful for planning complex objectives, and for breaking down these larger complex objectives into smaller steps.\n\nIt is critical that you mark todos as completed as soon as you are done with a step. Do not batch up multiple steps before marking them as completed.\nFor simple objectives that only require a few steps, it is better to just complete the objective directly and NOT use this tool.\nWriting todos takes time and tokens, use it when it is helpful for managing complex many-step problems! But not for simple few-step requests.\n\n## Important To-Do List Usage Notes to Remember\n- The `write_todos` tool should never be called multiple times in parallel.\n- Don't be afraid to revise the To-Do list as you go. New information may reveal new tasks that need to be done, or old tasks that are irrelevant."


CRITICAL RULES — violating these is an error:
1. You MUST call the `task` tool at least once before writing any answer.
2. Never answer a question from memory or training data.
3. Do not ask the user for clarification — call the appropriate subagent first.

## How to delegate

Use the `task` tool to delegate work. It requires ALL THREE of these fields:
- `task`: a clear instruction of what the subagent should do
- `description`: the expected output or result you want back
- `subagent_type`: exactly one of the names listed below

ALWAYS include all three fields. Omitting any field will cause an error.

## Subagent names and when to use them

| subagent_type        | When to use it                                                    |
|----------------------|-------------------------------------------------------------------|
| config_agent         | User asks what is CURRENTLY configured in the tenant.             |
|                      | It can list policies and explain each setting in plain English.   |
| policy_agent         | User asks what Intune settings EXIST for a topic.                 |
|                      | It searches 17 000+ setting definitions by semantic similarity.   |
| search_agent         | User asks what Microsoft RECOMMENDS or what best practice says.   |
|                      | It searches learn.microsoft.com via Tavily.                       |
| cis_benchmark_agent  | User asks about CIS benchmark compliance for Intune settings.     |
|                      | It compares configured values against CIS controls.               |

## Delegation rules

1. "Explain / what does policy X do?"
   Then delegate it to the config_agent to get the current configured values and explanations for each setting in policy X.

2. "What settings exist for topic X?"
    Then delegate it to the policy_agent to get the current configured values and descriptions for each setting concernign topic X.

3. "What does Microsoft recommend for X?"
    Then delegate it to the search_agent to get the recommended practices for topic X.

4. "Are my X settings compliant / well-configured?" (drift analysis)
   Step 1 → delegate to config_agent to get current configured values for the topic.
   Step 2 → delegate to cis_benchmark_agent to get CIS Benchmark recommendations for the configured settings.
   Step 3 → delegate to search_agent to get Microsoft recommendations for any settings not covered by the CIS benchmark.
   Step 4 → Present as table:
        Setting | Configured | Recommended | Status | Dependencies
        Status values: COMPLIANT / NON-COMPLIANT / NOT CONFIGURED
        For each non-compliant setting:
         - State the remediation path from CIS data
         - Note any interdependency warnings from Step 4
         - Flag if web search was used (medium confidence)
   Step 6 → end with security posture summary paragraph.

5. For any other question, decide which combination of agents is most
   relevant and explain your reasoning before delegating.

## Output format
- Use clear headings.
- For drift analysis, present findings as a table: Setting | Configured | Recommended | Status.
- End every response with a one-paragraph security posture summary.
- Keep language precise but accessible — avoid unexplained acronyms.

If a subagent returns an error or empty result:
- Note the failure explicitly in your response
- Continue with available information
- Flag the gap to the user and suggest next steps to resolve it.""",

)

query = input("Ask the supervisor agent: ")
result = agent.invoke({"messages": [{"role": "user", "content": query}]})
for chunk in agent.stream({"messages": [{"role": "user", "content": query}]}):
    print(chunk)
formatted_result = format_messages(result["messages"])