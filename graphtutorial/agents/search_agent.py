import logging

from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain.tools import ToolRuntime, tool
from tavily import TavilyClient
from dotenv import load_dotenv
import os
import json
from rich_renderer import RichRenderer
from activity_stream import astream_activity
import asyncio


OPENAI_MODEL = "gpt-5.4-nano-2026-03-17"

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Initialize the model
model = init_chat_model(model=OPENAI_MODEL, model_provider="openai", temperature=0.0)

# Tavily search (requires TAVILY_API_KEY)
client = TavilyClient(api_key=TAVILY_API_KEY)

_CATALOG_PATH = os.path.join(
    os.path.dirname(__file__),
    "../intune_configurations/intune_configuration_settings.json",
)

def _load_catalog() -> dict:
    with open(_CATALOG_PATH) as f:
        entries = json.load(f)
    return {e["id"]: e for e in entries}


_MAX_QUERY_LEN = 400

def _catalog_query(setting_id: str, catalog: dict) -> str:
    """Build a concise search query from displayName and keywords."""
    entry = catalog.get(setting_id, {})
    name = (entry.get("displayName") or entry.get("name") or setting_id).strip()
    # Use the catalog keywords as additional context — they're short and targeted
    keywords = [
        k for k in (entry.get("keywords") or [])
        if k.lower() != name.lower() and not k.startswith("\\") and "\\" not in k
    ]
    extra = " ".join(keywords[:2])  # at most 2 keywords to keep it short
    query = f"Microsoft Intune {name} {extra} configuration recommendation".strip()
    return query[:_MAX_QUERY_LEN]


@tool
def tavily_search_specific_configurations(runtime: ToolRuntime) -> str:
    """Search Microsoft documentation for specific Intune configuration guidance."""

    files = runtime.state.get("files", {})
    file_entry = files.get("/tenant_configs_vs_benchmark.json") or files.get("tenant_configs_vs_benchmark.json")

    if file_entry is None:
        return json.dumps({"error": "tenant_configs_vs_benchmark.json not found. Ensure policy_agent has run first."})
    if isinstance(file_entry, dict):
        raw = file_entry.get("content", [])
        content_str = "\n".join(raw) if isinstance(raw, list) else str(raw)
    else:
        content_str = str(file_entry)

    benchmark_output = json.loads(content_str)
    results = benchmark_output.get("results", [])

    not_in_benchmark = [
        r["setting_definition_id"]
        for r in results
        if r.get("cis_status") == "no_benchmark_reference" and r.get("setting_definition_id")
    ]

    catalog = _load_catalog()
    search_results = []

    for setting_id in not_in_benchmark:
        query = _catalog_query(setting_id, catalog)
        response = client.search(
            query=query,
            max_results=3,
            include_domains=["learn.microsoft.com/"],
        )
        search_results.append({setting_id: response["results"]})

    return json.dumps(search_results, indent=2)


search_agent = {
    "name": "search_agent",
    "description": "Searches the microsoft documentation for security best practices and vendor guidance.",
    "system_prompt": (
        "You are a Microsoft Intune documentation specialist. "
        "Your task is to find Microsoft's own recommendations for Intune settings "
        "that are not covered by the CIS benchmark.\n\n"

        "## Workflow\n"
        "1. Call tavily_search_specific_configurations to retrieve all settings that "
        "have no CIS benchmark reference and their Microsoft documentation search results."
        "Simply call the tool without any file data, it will be called in the tool anyways\n"
        "2. Use write_file tool to save these results in a file called search_agent_results.json "
        "for the supervisor_agent to read and use in its analysis.\n"
        "3. For each setting in the result of the tavily_search_specific_configurations tool,"
        "extract the relevant recommendation from the search results.\n"
        "4. If a setting returned no documentation results, mark it explicitly as "
        "'No Microsoft documentation found'.\n\n"

        "## Output format\n"
        "Return a structured list. For each setting:\n"
        "- **Setting ID**: the raw setting definition ID\n"
        "- **Description**: a one or two sentence summary of Microsoft's recommendation for this setting, based on the search results.\n"
        "- **Microsoft Recommendation**: one or two sentences summarising what "
        "Microsoft recommends for this setting, in plain language.\n"
        "- **Recommended Value**: the specific value or range Microsoft recommends, "
        "if stated in the documentation.\n"
        "- **Source**: the URL from learn.microsoft.com\n\n"

        "## Rules\n"
        "- Only report what the search results say. Do not invent recommendations.\n"
        "- If no documentation was found for a setting, state: "
        "- **Note**:'No Microsoft documentation found — human review required.'\n"
        "- Do not include Description, Recommendation or Recommended Value for settings with no documentation."
        "- Do not include settings that are already covered by the CIS benchmark."
    ),
    "tools": [tavily_search_specific_configurations],
    "model": model,
}

s_agent = create_deep_agent(
    model=model,
    system_prompt=(
        "You are a Microsoft Intune documentation specialist. "
        "Your task is to find Microsoft's own recommendations for Intune settings "
        "that are not covered by the CIS benchmark.\n\n"

        "## Workflow\n"
        "1. Call tavily_search_specific_configurations to retrieve all settings that "
        "have no CIS benchmark reference and their Microsoft documentation search results."
        "Simply call the tool without any file data, it will be called in the tool anyways\n"
        "2. Use write_file tool to save these results in a file called search_agent_results.json "
        "for the supervisor_agent to read and use in its analysis.\n"
        "3. For each setting in the result of the tavily_search_specific_configurations tool,"
        "extract the relevant recommendation from the search results.\n"
        "4. If a setting returned no documentation results, mark it explicitly as "
        "'No Microsoft documentation found'.\n\n"

        "## Output format\n"
        "Return a structured list. For each setting:\n"
        "- **Setting ID**: the raw setting definition ID\n"
        "- **Description**: a one or two sentence summary of Microsoft's recommendation for this setting, based on the search results.\n"
        "- **Microsoft Recommendation**: one or two sentences summarising what "
        "Microsoft recommends for this setting, in plain language.\n"
        "- **Recommended Value**: the specific value or range Microsoft recommends, "
        "if stated in the documentation.\n"
        "- **Source**: the URL from learn.microsoft.com\n\n"

        "## Rules\n"
        "- Only report what the search results say. Do not invent recommendations.\n"
        "- If no documentation was found for a setting, state: "
        "- **Note**:'No Microsoft documentation found — human review required.'\n"
        "- Do not include Description, Recommendation or Recommended Value for settings with no documentation."
        "- Do not include settings that are already covered by the CIS benchmark."
    ),
    tools=[tavily_search_specific_configurations],
)

def _file_data(path: str) -> dict:
    """Wrap a file's content in the FileData format deepagents expects."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    with open(path) as f:
        lines = f.read().splitlines()
    return {"content": lines, "created_at": now, "modified_at": now}

"""if __name__ == "__main__":
    # result = tavily_search.invoke("Bitlocker policies for Windows devices")
    logger = logging.getLogger(__name__)
    renderer = RichRenderer(logger=logger)

    #result = stream_agent_v2(agent, pending, config=run_config, on_interrupt=handle_interrupt)
    pending = {
        "messages": [{"role": "user", "content": "Check Microsoft documentation for any recommendations on settings that are not covered by the CIS benchmark. Read tenant_configs_vs_benchmark.json for the full settings list."}],
        "files": {
            "tenant_configs_vs_benchmark.json": _file_data(os.path.join(os.path.dirname(__file__), "tenant_configs_vs_benchmark.json"))
        },
    }
    run_config = {"configurable": {"thread_id": "1"}}

    final_state = asyncio.run(
        astream_activity(s_agent, agent_input=pending, config=run_config, render=False, on_event=renderer)
    )
    print("\nFINAL STATE:\n" + str(final_state))"""