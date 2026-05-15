from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
import json
import os
from dotenv import load_dotenv
import logging


load_dotenv()

OLLAMA_MODEL = "mistral-nemo:latest"
OPENAI_API_MODEL = "gpt-5.4-nano-2026-03-17"

# Initialize the model
#model = init_chat_model(model=OLLAMA_MODEL, model_provider="ollama", temperature=0.0)
model = init_chat_model(model=OPENAI_API_MODEL, model_provider="openai", temperature=0.0)

""" model = init_chat_model(
    model=OLLAMA_MODEL,
    model_provider="ollama",
    base_url="https://ollama.com",
    client_kwargs={"headers": {"Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"}},
    temperature=0.0,
) """

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s", 
    handlers=[
        logging.FileHandler("agent.log", mode='w'),  # overwrite log file on each run
    ],)


import time
from contextvars import ContextVar
from langchain.agents.middleware import before_model, after_model, wrap_tool_call

_start = ContextVar("model_start", default=None)

@before_model
def log_before_model(state, runtime):
    _start.set(time.time())
    return None

@after_model
def log_after_model(state, runtime):
    started = _start.get()
    elapsed = time.time() - started if started else 0
    last_msg = state["messages"][-1]
    
    tool_calls = getattr(last_msg, "tool_calls", None) or []
    usage = getattr(last_msg, "usage_metadata", None) or {}
    
    content = getattr(last_msg, "content", "") or ""
    logger.info(
        "← Model call done in %.2fs | tokens=%s | tool_calls=%s\n%s",
        elapsed,
        f"{usage.get('input_tokens', '?')}→{usage.get('output_tokens', '?')}",
        [tc["name"] for tc in tool_calls] if tool_calls else "none",
        content[:4000] + ("..." if len(content) > 4000 else ""),
    )
    return None


def log_chunk(chunk: dict) -> None:
    if not isinstance(chunk, dict):
        logger.info("Chunk: %s", str(chunk)[:1000])
        return

    for node, payload in chunk.items():
        if payload is None:
            logger.info("[%s] (no output)", node)
            continue

        # Payload might be an Overwrite/Append wrapper, a dict, or something else
        if not isinstance(payload, dict):
            logger.info("[%s] %s", node, str(payload)[:1000])
            continue

        # Drop noisy keys
        payload = {k: v for k, v in payload.items() if k != "files"}

        messages = payload.get("messages")
        if messages is None:
            # No messages field — just log keys touched
            logger.info("[%s] keys=%s", node, list(payload.keys()))
            continue

        # messages might also be an Overwrite wrapper, not a list
        if not isinstance(messages, list):
            logger.info("[%s] messages=%s", node, str(messages)[:300])
            continue

        for msg in messages:
            role = type(msg).__name__
            content = getattr(msg, "content", "")
            if not isinstance(content, str):
                content = str(content)
            preview = content[:300] + ("..." if len(content) > 300 else "")
            logger.info("[%s] %s: %s", node, role, preview)


@wrap_tool_call
def tool_logger(request, handler):
    name = request.tool_call["name"]
    args = request.tool_call["args"]
    logger.info("Tool call: %s args=%s", name, args)
    result = handler(request)
    content = result.content if hasattr(result, "content") else str(result)
    truncated = content[:300] + ("..." if len(content) > 300 else "")
    logger.info("Tool result: %s", truncated)
    return result

def _collect_settings(instance: dict, result: list[dict]) -> None:
    """Recursively extract every leaf setting from an Intune settingInstance tree."""
    dtype = instance.get("@odata.type", "")
    sid = instance.get("settingDefinitionId", "")

    if "ChoiceSettingInstance" in dtype:
        choice_val = instance.get("choiceSettingValue", {})
        result.append({
            "setting_definition_id": sid,
            "raw_value": choice_val.get("value", ""),
            "value_type": "choice",
        })
        for child in choice_val.get("children", []):
            _collect_settings(child, result)

    elif "SimpleSettingInstance" in dtype:
        result.append({
            "setting_definition_id": sid,
            "raw_value": instance.get("simpleSettingValue", {}).get("value"),
            "value_type": "integer",
        })

    elif "GroupSettingCollectionInstance" in dtype:
        for group_val in instance.get("groupSettingCollectionValue", []):
            for child in group_val.get("children", []):
                _collect_settings(child, result)

    elif "GroupSettingInstance" in dtype:
        for child in instance.get("groupSettingValue", {}).get("children", []):
            _collect_settings(child, result)

    elif "SimpleSettingCollectionInstance" in dtype:
        values = [v.get("value") for v in instance.get("simpleSettingCollectionValue", [])]
        result.append({
            "setting_definition_id": sid,
            "raw_value": values,
            "value_type": "simpleCollection",
        })

def check_compliance(tenant_value, cis_value, operator="=="):
    try:
        t = int(tenant_value)
        c = int(cis_value)
        if operator == ">=": return t >= c
        if operator == "<=": return t <= c
        if operator == ">":  return t > c
        if operator == "<":  return t < c
    except (ValueError, TypeError):
        pass
    return str(tenant_value) == str(cis_value)


@tool
def compare_to_cis_benchmark(query: str) -> str:
    """Compare the tenant's Intune policies against the CIS benchmark.

    Scans all policies in policies_and_settings_expand.json for every setting defined
    in matched_all_level1.json and produces a per-setting compliance report.
    Each setting is marked compliant, non_compliant, or not_configured.
    Settings with Assessment Status 'Manual' cannot be verified automatically and
    are marked accordingly.

    Args:
        query: Optional context passed through for LLM use; not used for filtering.
    """
    base = os.path.dirname(__file__)

    cis_benchmark_path = os.path.join(
        base,
        "../configurations/cis_benchmarks/matched_all_level1.json",
    )
    policies_path = os.path.join(
        base, "../configurations/policies_and_settings_expand.json"
    )

    with open(cis_benchmark_path) as f:
        cis_data = json.load(f)

    with open(policies_path) as f:
        policies = json.load(f)

    # Build index: setting_definition_id -> [{policy_name, policy_id, raw_value, value_type}]
    setting_index: dict[str, list[dict]] = {}
    for policy in policies:
        policy_settings: list[dict] = []
        for setting in policy.get("settings", []):
            _collect_settings(setting.get("settingInstance", {}), policy_settings)
        for s in policy_settings:
            sid = s["setting_definition_id"]
            setting_index.setdefault(sid, []).append({
                "policy_name": policy["name"],
                "policy_id": policy["id"],
                "raw_value": s["raw_value"],
                "value_type": s["value_type"],
            })

    results = []
    for benchmark_policy in cis_data["policies"]:
        policy_name = benchmark_policy["policy_metadata"]["intune_policy_name"]
        for item in benchmark_policy.get("matched_configurations", []):
            sid = item["setting_definition_id"]
            cis_raw = item["raw_value"]
            rec = item["cis_benchmark"]
            assessment_status = rec.get("Assessment Status", "")

            tenant_hits = setting_index.get(sid, [])

            if assessment_status == "Manual":
                status = "manual_check_required"
                tenant_configurations = []
            elif not tenant_hits:
                status = "not_configured"
                tenant_configurations = []
            else:
                tenant_configurations = [
                    {**hit, "compliant": check_compliance(hit["raw_value"], cis_raw, item.get("operator", "=="))}
                    for hit in tenant_hits
                ]
                status = (
                    "compliant"
                    if all(h["compliant"] for h in tenant_configurations)
                    else "non_compliant"
                )

            results.append({
                "setting_definition_id": sid,
                "cis_id": item["cis_recommendation_number"],
                "cis_title": rec["Title"],
                "assessment_status": assessment_status,
                "benchmark_policy": policy_name,
                "cis_reference_value": cis_raw,
                "cis_reference_value_display": item["configured_value"],
                "value_type": item["value_type"],
                "status": status,
                "tenant_configurations": tenant_configurations,
            })

    summary = {
        "total": len(results),
        "compliant": sum(1 for r in results if r["status"] == "compliant"),
        "non_compliant": sum(1 for r in results if r["status"] == "non_compliant"),
        "not_configured": sum(1 for r in results if r["status"] == "not_configured"),
        "manual_check_required": sum(1 for r in results if r["status"] == "manual_check_required"),
    }

    return json.dumps({
        "benchmark_source": cis_data["benchmark_csv_source"],
        "policies_processed": cis_data["total_policies_processed"],
        "summary": summary,
        "results": results,
    }, indent=2)

@tool
def compare_relevant_settings_to_benchmark(runtime: ToolRuntime) -> str:
    """Compare the tenant's relevant configured settings against CIS benchmarks.

    Reads relevant_configs.json written by config_agent, extracts the 'found'
    settings list, and checks each one against the CIS benchmark data.
    Settings not covered by any CIS benchmark are marked not_in_benchmark;
    settings marked as Manual in CIS are flagged for human review.
    """
    # Try virtual filesystem first
    files = runtime.state.get("files", {})
    file_entry = files.get("/relevant_configs.json") or files.get("relevant_configs.json")
    if file_entry is not None:
        if isinstance(file_entry, dict):
            # Proper FileData: content is list[str] lines
            raw = file_entry.get("content", [])
            content_str = "\n".join(raw) if isinstance(raw, list) else raw
        else:
            # Raw string passed directly
            content_str = str(file_entry)
        relevant = json.loads(content_str)
    else:
        # Fall back to disk — find_configs_in_policies writes it there directly
        disk_path = os.path.join(os.path.dirname(__file__), "relevant_configs.json")
        try:
            with open(disk_path) as f:
                relevant = json.load(f)
        except FileNotFoundError:
            return "relevant_configs.json not found in virtual filesystem or on disk. Run config_agent first."

    base = os.path.dirname(__file__)
    cis_benchmark_path = os.path.join(
        base, "../configurations/cis_benchmarks/matched_all_level1.json"
    )
    with open(cis_benchmark_path) as f:
        cis_data = json.load(f)       
    # Support two formats:
    #   {"found": [...], "missing": [...]}  — written by find_configs_in_policies
    #   [...]                               — flat list of settings passed directly
    if isinstance(relevant, list):
        raw_items = relevant
    else:
        raw_items = relevant.get("found", [])

    # Build settings_to_check, deduplicated by setting id
    seen: set[str] = set()
    settings_to_check: list[dict] = []
    for item in raw_items:
        config = item.get("config", item)
        sid = config.get("id")
        if not sid or sid in seen:
            continue
        seen.add(sid)
        settings_to_check.append({
            "id": sid,
            "configured_value": config.get("configured_value"),
            "configured_value_label": config.get("configured_value_label", ""),
            "name": config.get("name", sid),
            "policy_name": item.get("policy_name", config.get("policy_name", "")),
            "policy_id": item.get("policy_id", config.get("policy_id", "")),
        })

    # Build CIS lookup index: setting_definition_id → {cis_item, benchmark_policy_name}
    cis_index: dict[str, dict] = {}
    for benchmark_policy in cis_data["policies"]:
        policy_name = benchmark_policy["policy_metadata"]["intune_policy_name"]
        for item in benchmark_policy.get("matched_configurations", []):
            sid = item["setting_definition_id"]
            if sid not in cis_index:
                cis_index[sid] = {"cis_item": item, "benchmark_policy_name": policy_name}

    results = []
    for setting in settings_to_check:
        sid = setting["id"]
        configured_value = setting.get("configured_value")
        configured_label = setting.get("configured_value_label", str(configured_value))
        cis_match = cis_index.get(sid)
        print(f"Checking setting {setting.get('name', sid)} (configured: {configured_label}) against CIS benchmark...")
        print(f"CIS match found: {cis_match is not None}")
        if cis_match is None:
            results.append({
                "setting_definition_id": sid,
                "setting_name": setting.get("name", sid),
                "configured_value": configured_value,
                "configured_value_label": configured_label,
                "policy_name": setting.get("policy_name", ""),
                "status": "not_in_benchmark",
                "cis_match": None,
            })
            continue

        cis_item = cis_match["cis_item"]
        rec = cis_item["cis_benchmark"]
        cis_raw = cis_item["raw_value"]
        operator = cis_item.get("operator", "==")
        assessment_status = rec.get("Assessment Status", "")

        if assessment_status == "Manual":
            status = "manual_check_required"
            compliant = None
        else:
            compliant = check_compliance(configured_value, cis_raw, operator)
            status = "compliant" if compliant else "non_compliant"

        results.append({
            "setting_definition_id": sid,
            "setting_name": setting.get("name", sid),
            "configured_value": configured_value,
            "configured_value_label": configured_label,
            "policy_name": setting.get("policy_name", ""),
            "status": status,
            "cis_match": {
                "cis_id": cis_item["cis_recommendation_number"],
                "cis_title": rec["Title"],
                "assessment_status": assessment_status,
                "cis_reference_value": cis_raw,
                "cis_reference_value_display": cis_item["configured_value"],
                "operator": operator,
                "rationale": rec.get("Rationale Statement", ""),
                "impact": rec.get("Impact Statement", ""),
                "remediation": rec.get("Remediation Procedure", ""),
                "benchmark_policy": cis_match["benchmark_policy_name"],
            },
        })
    print("Results:", json.dumps(results, indent=2))
    summary = {
        "total_checked": len(results),
        "compliant": sum(1 for r in results if r["status"] == "compliant"),
        "non_compliant": sum(1 for r in results if r["status"] == "non_compliant"),
        "not_in_benchmark": sum(1 for r in results if r["status"] == "not_in_benchmark"),
        "manual_check_required": sum(1 for r in results if r["status"] == "manual_check_required"),
    }

    return json.dumps({"summary": summary, "results": results}, indent=2)

@tool
def compare_relevant_settings_to_cis_benchmark(runtime: ToolRuntime, settings_to_check: list[dict] | None = None) -> str:
    """Compare configured tenant settings against CIS benchmarks.

    When settings_to_check is omitted (or empty), reads relevant_configs.json
    from the virtual filesystem (or disk) automatically.  When settings_to_check
    is provided explicitly, uses that list directly.

    Each item in settings_to_check must contain:
      - id: setting definition ID
      - configured_value: current configured value in tenant
      - configured_value_label (optional): human-readable label
      - name (optional): setting display name
      - policy_name (optional): policy that configures it
      - policy_id (optional): policy ID

    Returns:
        JSON string with summary stats and per-setting compliance findings.
    """
    if not settings_to_check:
        # Try virtual filesystem first, then fall back to disk
        files = runtime.state.get("files", {})
        file_entry = files.get("/relevant_configs.json") or files.get("relevant_configs.json")
        if file_entry is not None:
            if isinstance(file_entry, dict):
                raw = file_entry.get("content", [])
                content_str = "\n".join(raw) if isinstance(raw, list) else str(raw)
            else:
                content_str = str(file_entry)
            relevant = json.loads(content_str)
        else:
            disk_path = os.path.join(os.path.dirname(__file__), "relevant_configs.json")
            try:
                with open(disk_path) as f:
                    relevant = json.load(f)
            except FileNotFoundError:
                return json.dumps({"error": "relevant_configs.json not found. Run config_agent first."})

        if isinstance(relevant, list):
            raw_items = relevant
        else:
            raw_items = relevant.get("found", [])

        seen: set[str] = set()
        settings_to_check = []
        for item in raw_items:
            config = item.get("config", item)
            sid = config.get("id")
            if not sid or sid in seen:
                continue
            seen.add(sid)
            settings_to_check.append({
                "id": sid,
                "configured_value": config.get("configured_value"),
                "configured_value_label": config.get("configured_value_label", ""),
                "name": config.get("name", sid),
                "policy_name": item.get("policy_name", config.get("policy_name", "")),
                "policy_id": item.get("policy_id", config.get("policy_id", "")),
            })
    base = os.path.dirname(__file__)
    cis_benchmark_path = os.path.join(
        base,
        "../configurations/cis_benchmarks/matched_all_level1.json",
    )
    
    with open(cis_benchmark_path) as f:
        cis_data = json.load(f)
    
    # Build CIS lookup index: setting_definition_id → {cis_entry, benchmark_policy_name}
    cis_index: dict[str, dict] = {}
    for benchmark_policy in cis_data["policies"]:
        policy_name = benchmark_policy["policy_metadata"]["intune_policy_name"]
        for item in benchmark_policy.get("matched_configurations", []):
            sid = item["setting_definition_id"]
            # First mapping wins if a setting appears in multiple benchmarks
            if sid not in cis_index:
                cis_index[sid] = {
                    "cis_item": item,
                    "benchmark_policy_name": policy_name,
                }
    
    # Check each input setting against the CIS index
    results = []
    for setting in settings_to_check:
        sid = setting.get("id")
        if not sid:
            continue
        
        configured_value = setting.get("configured_value")
        configured_label = setting.get("configured_value_label", str(configured_value))
        setting_name = setting.get("name", sid)
        policy_name = setting.get("policy_name", "")
        policy_id = setting.get("policy_id", "")
        
        cis_match = cis_index.get(sid)
        
        if cis_match is None:
            # Configured in tenant but no CIS coverage — note explicitly
            results.append({
                "setting_definition_id": sid,
                "setting_name": setting_name,
                "configured_value": configured_value,
                "configured_value_label": configured_label,
                "policy_name": policy_name,
                "policy_id": policy_id,
                "status": "not_in_benchmark",
                "cis_match": None,
                "notes": "Configured in tenant; no matching CIS recommendation found.",
            })
            continue
        
        cis_item = cis_match["cis_item"]
        rec = cis_item["cis_benchmark"]
        cis_raw = cis_item["raw_value"]
        operator = cis_item.get("operator", "==")
        assessment_status = rec.get("Assessment Status", "")
        
        # Settings CIS flags as Manual can't be automatically verified
        if assessment_status == "Manual":
            status = "manual_check_required"
            compliant = None
        else:
            compliant = check_compliance(configured_value, cis_raw, operator)
            status = "compliant" if compliant else "non_compliant"
        
        results.append({
            "setting_definition_id": sid,
            "setting_name": setting_name,
            "configured_value": configured_value,
            "configured_value_label": configured_label,
            "policy_name": policy_name,
            "policy_id": policy_id,
            "status": status,
            "cis_match": {
                "cis_id": cis_item["cis_recommendation_number"],
                "cis_title": rec["Title"],
                "assessment_status": assessment_status,
                "cis_reference_value": cis_raw,
                "cis_reference_value_display": cis_item["configured_value"],
                "value_type": cis_item["value_type"],
                "operator": operator,
                "rationale": rec.get("Rationale Statement", ""),
                "impact": rec.get("Impact Statement", ""),
                "remediation": rec.get("Remediation Procedure", ""),
                "benchmark_policy": cis_match["benchmark_policy_name"],
                "benchmark_source": cis_data.get("benchmark_csv_source", ""),
            },
        })
    
    summary = {
        "total_checked": len(results),
        "compliant": sum(1 for r in results if r["status"] == "compliant"),
        "non_compliant": sum(1 for r in results if r["status"] == "non_compliant"),
        "not_in_benchmark": sum(1 for r in results if r["status"] == "not_in_benchmark"),
        "manual_check_required": sum(1 for r in results if r["status"] == "manual_check_required"),
    }
    
    return json.dumps({
        "summary": summary,
        "results": results,
    }, indent=2)

cis_benchmark_agent = {
    "name": "cis_benchmark_agent",
    "description": (
        "Compares the tenant's relevant configured settings against CIS benchmarks. "
        "Reads relevant_configs.json written by config_agent and produces a compliance report."
    ),
    "system_prompt": (
        "You are a CIS Benchmark compliance analyst for Microsoft Intune. "
        "Your job is to compare the tenant's current configuration against "
        "CIS Benchmark recommendations and report gaps clearly.\n\n"

        "## Steps\n"
        "1. Call compare_relevant_settings_to_benchmark() — it reads relevant_configs.json "
        "automatically and returns compliance findings for those settings.\n"
        "   If a file-not-found error occurs, fall back to compare_to_cis_benchmark(query=<description>) "
        "for a full tenant scan.\n"
        "2. Present results as a compliance table with columns:\n"
        "   CIS ID | Benchmark Policy | Setting Name | Configured Value | "
        "   CIS Recommended | Status\n"
        "3. Use these status labels:\n"
        "   - COMPLIANT: configured value matches CIS recommendation\n"
        "   - NON-COMPLIANT: configured value deviates from recommendation\n"
        "   - NOT IN BENCHMARK: setting is not covered by CIS\n"
        "   - MANUAL CHECK REQUIRED: CIS marks this as a manual control\n\n"

        "## For each NON-COMPLIANT setting\n"
        "- State the current value and what the CIS recommendation is.\n"
        "- State the CIS rationale for why this value matters.\n"
        "- State the remediation path from the CIS benchmark data.\n\n"

        "## For each MANUAL CHECK REQUIRED setting\n"
        "- Note that this control cannot be verified programmatically.\n"
        "- State what CIS recommends and the audit procedure if available.\n\n"

        "## Important\n"
        "Only use data from tool output. "
        "Do not generate remediation steps from your own knowledge."
    ),
    "tools": [compare_relevant_settings_to_benchmark, compare_to_cis_benchmark],
}

benchmark_agent = create_deep_agent(
    middleware=[log_before_model, log_after_model, tool_logger],
    system_prompt=(
        "You are a CIS Benchmark compliance analyst for Microsoft Intune. "
        "Your job is to compare the tenant's current configuration against "
        "CIS Benchmark recommendations and report gaps clearly.\n\n"

        "## Steps\n"
        "1. Call compare_relevant_settings_to_benchmark() with NO arguments — "
        "it reads relevant_configs.json from the virtual filesystem automatically.\n"
        "   If a file-not-found error is returned, fall back to compare_to_cis_benchmark(query=<description>).\n"
        "2. Present results as a compliance table with columns:\n"
        "   CIS ID | Benchmark Policy | Setting Name | Configured Value | "
        "   CIS Recommended | Status\n"
        "3. Use these status labels:\n"
        "   - COMPLIANT: configured value matches CIS recommendation\n"
        "   - NON-COMPLIANT: configured value deviates from recommendation\n"
        "   - NOT IN BENCHMARK: setting is not covered by CIS\n"
        "   - MANUAL CHECK REQUIRED: CIS marks this as a manual control\n\n"

        "## For each NON-COMPLIANT setting\n"
        "- State the current value and what the CIS recommendation is.\n"
        "- State the CIS rationale for why this value matters.\n"
        "- State the remediation path from the CIS benchmark data.\n\n"

        "## For each MANUAL CHECK REQUIRED setting\n"
        "- Note that this control cannot be verified programmatically.\n"
        "- State what CIS recommends and the audit procedure if available.\n\n"

        "## Important\n"
        "Only use data from tool output. "
        "Do not generate remediation steps from your own knowledge."
    ),
    model=model,
    tools=[compare_relevant_settings_to_benchmark, compare_to_cis_benchmark])

def _file_data(path: str) -> dict:
    """Wrap a file's content in the FileData format deepagents expects."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    with open(path) as f:
        lines = f.read().splitlines()
    return {"content": lines, "created_at": now, "modified_at": now}

if __name__ == "__main__":
    result = benchmark_agent.invoke({
        "messages": [{"role": "user", "content": "What are the CIS benchmark compliance gaps for the following settings?"}],
        "files": {"relevant_configs.json": _file_data(os.path.join(os.path.dirname(__file__), "relevant_configs.json"))},
    })
    print(result["messages"][-1]["content"])