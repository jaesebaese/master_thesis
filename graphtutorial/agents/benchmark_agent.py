from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
import json
import os
from dotenv import load_dotenv
import logging
try:
    from .preprocessing_at_startup import build_cis_benchmark_vector_db
except ImportError:
    from preprocessing_at_startup import build_cis_benchmark_vector_db


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

def _resolve_operator(stored_operator: str, cis_title: str) -> str:
    """Upgrade a stored '==' operator to '>=' or '<=' when the CIS title implies a range.

    Many CIS titles say 'set to X or more character(s)' but the data file stores
    operator '==' because the exact threshold value is what was matched.  Reading
    the title avoids false non_compliant results when the tenant exceeds the minimum.
    """
    if stored_operator != "==":
        return stored_operator
    t = cis_title.lower()
    if any(p in t for p in ("or more", "or greater", "or higher", "or above", "min", "min.", "minimum")):
        return ">="
    if any(p in t for p in ("or fewer", "or less", "or lower", "or below", "max", "max.", "maximum")):
        return "<="
    return stored_operator


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
        # Settings CIS flags as Manual can't be automatically verified

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
    }
    
    return json.dumps({
        "summary": summary,
        "results": results,
    }, indent=2)

@tool
def search_benchmark(runtime: ToolRuntime, requirements: str = "") -> str:
    """Search the CIS benchmark vector DB for each requirement in a list.

    When called with no arguments, reads requirements.json automatically from the
    virtual filesystem (or disk).  Otherwise accepts the JSON output of
    policy_requirement_extractor (the full object with a "requirements" key, or a
    bare list of requirement objects).  For each requirement, performs a semantic
    similarity search and returns the top CIS benchmark controls that match.

    Args:
        requirements: JSON string — either {"requirements": [...]} or [...] where
                      each item has at least "requirement_id" and "source_text".
                      Omit to auto-load from policy_requirements.json.
    """
    files = runtime.state.get("files", {})
    file_entry = files.get("/policy_requirements.json") or files.get("policy_requirements.json")
    if file_entry is None:
        return json.dumps({"error": "policy_requirements.json not found. Ensure policy_agent has run first."})
    if isinstance(file_entry, dict):
        raw = file_entry.get("content", [])
        requirements = "\n".join(raw) if isinstance(raw, list) else str(raw)
    else:
        requirements = str(file_entry)

    try:
        parsed = json.loads(requirements)
    except json.JSONDecodeError:
        import ast
        parsed = ast.literal_eval(requirements)
    req_list: list[dict] = parsed.get("requirements", parsed) if isinstance(parsed, dict) else parsed

    collection = build_cis_benchmark_vector_db()

    # Batch all queries in one ChromaDB call for efficiency
    query_texts = [
        f"{r['source_text']} {r.get('control_intent', '')}".strip()
        for r in req_list
    ]

    batch_results = collection.query(
        query_texts=query_texts,
        n_results=5,
        include=["metadatas", "distances"],
    )

    output = []
    for req, metadatas, distances in zip(req_list, batch_results["metadatas"], batch_results["distances"]):
        matches = []
        for meta, distance in zip(metadatas, distances):
            score = round(1 - distance, 4)
            if score > 0.5:
                matches.append({
                    "cis_id": meta.get("cis_id"),
                    "cis_section": meta.get("cis_section"),
                    "cis_title": meta.get("cis_title"),
                    "setting_definition_id": meta.get("setting_definition_id"),
                    "configured_value": meta.get("configured_value"),
                    "raw_value": meta.get("raw_value"),
                    "policy_name": meta.get("policy_name"),
                    "similarity_score": score,
                })
        output.append({
            "requirement_id": req.get("requirement_id"),
            "source_text": req.get("source_text"),
            "security_domain": req.get("security_domain"),
            "control_intent": req.get("control_intent"),
            "cis_matches": matches,
        })

    return json.dumps(output, indent=2)


@tool
def compare_search_results_to_tenant( runtime: ToolRuntime ) -> str:
    """Compare CIS settings found by search_benchmark against tenant configurations.

    Takes the JSON output of search_benchmark, collects every unique
    setting_definition_id across all requirements, looks each one up in the
    tenant's configured policies (policies_and_settings_expand_assign.json),
    and reports whether it is compliant, non_compliant, or not_configured.

    Args:
        search_results: JSON string — the direct output of search_benchmark.
    """
    
    files = runtime.state.get("files", {})
    file_entry = files.get("/requirements_vs_benchmark.json") or files.get("requirements_vs_benchmark.json")
    if file_entry is None:
        return json.dumps({"error": "requirements_vs_benchmark.json not found. Run search_benchmark first."})
    if isinstance(file_entry, dict):
        raw = file_entry.get("content", [])
        requirements = "\n".join(raw) if isinstance(raw, list) else str(raw)
    else:
        requirements = str(file_entry)

    try:
        parsed = json.loads(requirements)
    except json.JSONDecodeError:
        import ast
        parsed = ast.literal_eval(requirements)

    # Collect unique setting_definition_ids from search results, merging requirement_ids
    # when the same setting appears across multiple requirement matches.
    cis_settings: dict[str, dict] = {}
    for req_result in parsed:
        req_id = req_result.get("requirement_id", "")
        for match in req_result.get("cis_matches", []):
            sid = match.get("setting_definition_id")
            if not sid:
                continue
            if sid not in cis_settings:
                cis_settings[sid] = {
                    "cis_id": match.get("cis_id", ""),
                    "cis_title": match.get("cis_title", ""),
                    "cis_reference_value_display": match.get("configured_value", ""),
                    "cis_raw_value": match.get("raw_value", ""),
                    "benchmark_policy": match.get("policy_name", ""),
                    "requirement_ids": [],
                }
            if req_id and req_id not in cis_settings[sid]["requirement_ids"]:
                cis_settings[sid]["requirement_ids"].append(req_id)

    if not cis_settings:
        return json.dumps({"error": "No setting_definition_ids found in search results."})

    base = os.path.dirname(__file__)

    # Enrich each CIS setting with full benchmark details from matched_all_level1.json
    cis_benchmark_path = os.path.join(base, "../configurations/cis_benchmarks/matched_all_level1.json")
    with open(cis_benchmark_path) as f:
        cis_data = json.load(f)
    cis_details: dict[str, dict] = {}
    for policy in cis_data.get("policies", []):
        policy_name = policy["policy_metadata"]["intune_policy_name"]
        for item in policy.get("matched_configurations", []):
            sid = item.get("setting_definition_id", "")
            if not sid or sid in cis_details:
                continue
            rec = item.get("cis_benchmark", {})
            cis_details[sid] = {
                "operator": _resolve_operator(item.get("operator", "=="), rec.get("Title", "")),
                "rationale": rec.get("Rationale Statement", ""),
                "impact": rec.get("Impact Statement", ""),
                "remediation": rec.get("Remediation Procedure", ""),
                "benchmark_policy_full": policy_name,
            }

    # Build tenant setting index from policies_and_settings_expand_assign.json
    policies_path = os.path.join(base, "../configurations/policies_and_settings_expand_assign.json")
    with open(policies_path) as f:
        policies = json.load(f)

    setting_index: dict[str, list[dict]] = {}
    for policy in policies:
        policy_settings: list[dict] = []
        for setting in policy.get("settings", []):
            _collect_settings(setting.get("settingInstance", {}), policy_settings)
        for s in policy_settings:
            sid = s["setting_definition_id"]
            setting_index.setdefault(sid, []).append({
                "policy_name": policy.get("name", ""),
                "policy_id": policy.get("id", ""),
                "raw_value": s["raw_value"],
                "value_type": s["value_type"],
            })

    # Compare each CIS setting against tenant configuration
    results = []
    for sid, cis in cis_settings.items():
        tenant_hits = setting_index.get(sid, [])
        details = cis_details.get(sid, {})
        operator = details.get("operator", "==")
        cis_raw = cis["cis_raw_value"]

        cis_match_detail = {
            "cis_id": cis["cis_id"],
            "cis_title": cis["cis_title"],
            "cis_reference_value": cis_raw,
            "cis_reference_value_display": cis["cis_reference_value_display"],
            "operator": operator,
            "rationale": details.get("rationale", ""),
            "impact": details.get("impact", ""),
            "remediation": details.get("remediation", ""),
            "benchmark_policy": details.get("benchmark_policy_full", cis["benchmark_policy"]),
        }

        cis_compliant_match = {
            "cis_id": cis["cis_id"],
            "cis_title": cis["cis_title"],
            "cis_reference_value": cis_raw,
            "cis_reference_value_display": cis["cis_reference_value_display"],
            "benchmark_policy": details.get("benchmark_policy_full", cis["benchmark_policy"]),
        }

        if not tenant_hits:
            status = "not_configured"
            tenant_configurations = []
        else:
            tenant_configurations = [
                {**hit, "compliant": check_compliance(hit["raw_value"], cis_raw, operator)}
                for hit in tenant_hits
            ]
            status = "compliant" if all(h["compliant"] for h in tenant_configurations) else "non_compliant"

        results.append({
            "setting_definition_id": sid,
            "matched_by_requirements": cis["requirement_ids"],
            "status": status,
            "cis_match": cis_compliant_match if status == "compliant" else cis_match_detail,
            "tenant_configurations": tenant_configurations,
        })

    status_order = {"non_compliant": 0, "not_configured": 1, "manual_check_required": 2, "compliant": 3}
    results.sort(key=lambda r: status_order.get(r["status"], 4))

    summary = {
        "total": len(results),
        "compliant": sum(1 for r in results if r["status"] == "compliant"),
        "non_compliant": sum(1 for r in results if r["status"] == "non_compliant"),
        "not_configured": sum(1 for r in results if r["status"] == "not_configured"),
    }
    return json.dumps({"summary": summary, "results": results}, indent=2)


@tool
def compare_requirements_results(runtime: ToolRuntime) -> str:
    """Unified compliance check merging both requirements search outputs.

    Reads requirements_vs_benchmark.json (output of search_benchmark) and
    relevant_configurations.json (output of analyze_requirements_against_tenant),
    collects the union of all unique setting_definition_ids, and for each setting:
    - Looks up its CIS benchmark reference value from matched_all_level1.json
    - Looks up its configured value(s) from the tenant's policies JSON
    - Determines compliance status and records which search(es) surfaced it

    Compliance statuses:
      compliant              - tenant value satisfies the CIS recommendation
      non_compliant          - tenant value deviates from the CIS recommendation
      not_configured         - setting has a CIS reference but is absent from all tenant policies
      no_benchmark_reference - setting is in tenant but has no CIS benchmark entry
    """
    files = runtime.state.get("files", {})

    def _load_entry(key: str):
        entry = files.get(f"/{key}") or files.get(key)
        if entry is None:
            return None
        if isinstance(entry, dict):
            raw = entry.get("content", [])
            s = "\n".join(raw) if isinstance(raw, list) else str(raw)
        else:
            s = str(entry)
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            import ast
            return ast.literal_eval(s)

    bench_data = _load_entry("requirements_vs_benchmark.json") or []
    tenant_req_data = _load_entry("relevant_configurations.json") or []

    if not bench_data and not tenant_req_data:
        return json.dumps({
            "error": (
                "Neither requirements_vs_benchmark.json nor relevant_configurations.json "
                "found. Run search_cis_benchmark and/or analyze_requirements_against_tenant first."
            )
        })

    # Collect unique setting IDs with provenance and requirement linkage
    settings: dict[str, dict] = {}

    for req in bench_data:
        req_id = req.get("requirement_id", "")
        for match in req.get("cis_matches", []):
            sid = match.get("setting_definition_id")
            if not sid:
                continue
            if sid not in settings:
                settings[sid] = {"requirement_ids": [], "found_via": set()}
            settings[sid]["found_via"].add("benchmark_search")
            if req_id and req_id not in settings[sid]["requirement_ids"]:
                settings[sid]["requirement_ids"].append(req_id)

    for req in tenant_req_data:
        req_id = req.get("requirement_id", "")
        for match in req.get("tenant_matches", []):
            sid = match.get("setting_id")  # config_agent uses "setting_id"
            if not sid:
                continue
            if sid not in settings:
                settings[sid] = {"requirement_ids": [], "found_via": set()}
            settings[sid]["found_via"].add("tenant_search")
            if req_id and req_id not in settings[sid]["requirement_ids"]:
                settings[sid]["requirement_ids"].append(req_id)

    if not settings:
        return json.dumps({"error": "No setting IDs found in either search output."})

    base = os.path.dirname(__file__)

    # Enrich with full CIS benchmark details
    cis_benchmark_path = os.path.join(base, "../configurations/cis_benchmarks/matched_all_level1.json")
    with open(cis_benchmark_path) as f:
        cis_data = json.load(f)
    cis_details: dict[str, dict] = {}
    for policy in cis_data.get("policies", []):
        policy_name = policy["policy_metadata"]["intune_policy_name"]
        for item in policy.get("matched_configurations", []):
            sid = item.get("setting_definition_id", "")
            if not sid or sid in cis_details:
                continue
            rec = item.get("cis_benchmark", {})
            cis_details[sid] = {
                "cis_id": rec.get("Recommendation #", ""),
                "cis_title": rec.get("Title", ""),
                "cis_reference_value_display": str(item.get("configured_value", "")),
                "cis_raw_value": str(item.get("raw_value", "")),
                "operator": _resolve_operator(item.get("operator", "=="), rec.get("Title", "")),
                "rationale": rec.get("Rationale Statement", ""),
                "impact": rec.get("Impact Statement", ""),
                "remediation": rec.get("Remediation Procedure", ""),
                "benchmark_policy": policy_name,
            }

    # Build tenant setting index
    policies_path = os.path.join(base, "../configurations/policies_and_settings_expand_assign.json")
    with open(policies_path) as f:
        policies = json.load(f)
    setting_index: dict[str, list[dict]] = {}
    for policy in policies:
        policy_settings: list[dict] = []
        for setting in policy.get("settings", []):
            _collect_settings(setting.get("settingInstance", {}), policy_settings)
        for s in policy_settings:
            sid = s["setting_definition_id"]
            setting_index.setdefault(sid, []).append({
                "policy_name": policy.get("name", ""),
                "policy_id": policy.get("id", ""),
                "raw_value": s["raw_value"],
                "value_type": s["value_type"],
            })

    results = []
    for sid, info in settings.items():
        found_via = sorted(info["found_via"])
        cis = cis_details.get(sid)
        tenant_hits = setting_index.get(sid, [])

        tenant_status = "configured" if tenant_hits else "not_configured"
        cis_status = "has_benchmark" if cis else "no_benchmark_reference"

        if not tenant_hits:
            compliance_status = "not_configured"
        elif not cis:
            compliance_status = "no_benchmark_reference"
        else:
            operator = cis["operator"]
            cis_raw = cis["cis_raw_value"]
            all_compliant = all(
                check_compliance(h["raw_value"], cis_raw, operator) for h in tenant_hits
            )
            compliance_status = "compliant" if all_compliant else "non_compliant"

        if cis and tenant_hits:
            operator = cis["operator"]
            cis_raw = cis["cis_raw_value"]
            tenant_configurations = [
                {**h, "compliant": check_compliance(h["raw_value"], cis_raw, operator)}
                for h in tenant_hits
            ]
        else:
            tenant_configurations = tenant_hits

        record: dict = {
            "setting_definition_id": sid,
            "matched_by_requirements": info["requirement_ids"],
            "found_via": found_via,
            "tenant_status": tenant_status,
            "cis_status": cis_status,
            "compliance_status": compliance_status,
        }

        if cis:
            cis_block: dict = {
                "cis_id": cis["cis_id"],
                "cis_title": cis["cis_title"],
                "cis_reference_value": cis["cis_raw_value"],
                "cis_reference_value_display": cis["cis_reference_value_display"],
                "operator": cis["operator"],
                "benchmark_policy": cis["benchmark_policy"],
            }
            if compliance_status == "non_compliant":
                cis_block["rationale"] = cis["rationale"]
                cis_block["impact"] = cis["impact"]
                cis_block["remediation"] = cis["remediation"]
            record["cis_benchmark"] = cis_block

        record["tenant_configurations"] = tenant_configurations
        results.append(record)

    status_order = {"non_compliant": 0, "not_configured": 1, "no_benchmark_reference": 2, "compliant": 3}
    results.sort(key=lambda r: status_order.get(r["compliance_status"], 4))

    summary = {
        "total": len(results),
        "compliant": sum(1 for r in results if r["compliance_status"] == "compliant"),
        "non_compliant": sum(1 for r in results if r["compliance_status"] == "non_compliant"),
        "not_configured": sum(1 for r in results if r["compliance_status"] == "not_configured"),
        "no_benchmark_reference": sum(1 for r in results if r["compliance_status"] == "no_benchmark_reference"),
        "found_via_benchmark_search_only": sum(1 for r in results if r["found_via"] == ["benchmark_search"]),
        "found_via_tenant_search_only": sum(1 for r in results if r["found_via"] == ["tenant_search"]),
        "found_via_both": sum(1 for r in results if len(r["found_via"]) == 2),
    }
    return json.dumps({"summary": summary, "results": results}, indent=2)


benchmark_agent = {
    "name": "benchmark_agent",
    "description": (
        "Compares the tenant's configured settings against CIS benchmarks. "
        "Always runs two steps: (1) search_cis_benchmark to find CIS controls matching "
        "each policy requirement, then (2) compare_requirements_results to check whether "
        "the tenant's configured values comply with those CIS recommendations. "
    ),
    "system_prompt": (
       "You are a CIS Benchmark compliance analyst for Microsoft Intune. "
        "Your job is to map security policy requirements to CIS Benchmark controls "
        "and check whether the tenant is compliant.\n\n"

        "## Workflow\n"
        "Complete these steps in order. Do not proceed to the next step until the current one is complete.\n\n"
        "Step 1: Call search_cis_benchmark() with no arguments — it reads policy_requirements.json "
        "from the virtual filesystem and returns matching CIS controls per requirement.\n"

        "Step 2: Write the exact result from search_cis_benchmark() to a file called "
        "'requirements_vs_benchmark.json' using the write_file tool.\n"

        "Step 3: Only after step 2 is complete, call compare_requirements_results() — "
        "it checks each matched CIS control against the tenant's configured policies "
        "and returns a compliance verdict per setting.\n"

        "Step 4: Write the exact result from compare_requirements_results() to a file called "
        "'tenant_configs_vs_benchmark.json' using the write_file tool.\n"

        "Step 5: Only after step 4 is complete, present the results as described below.\n"

        "## Output structure\n"
        "Use the data from 'tenant_configs_vs_benchmark.json' to present the compliance results. Do not use any other data source for the compliance status or configured values.\n\n"
        "Present results in this order:\n"
        "1. Compliance summary: total counts of COMPLIANT / NON-COMPLIANT / NOT CONFIGURED / NOT IN BENCHMARK\n"
        "2. Full compliance table with columns:\n"
        "   Setting ID | Configured Value | CIS Recommended Value | Status | Matched Requirements\n"
        "   - Setting ID: setting_definition_id\n"
        "   - Configured Value: raw_value in tenant_configurations\n"
        "   - CIS Recommended Value: cis_reference_value from the cis_benchmark\n"
        "   - Status: compliance_status (any from the Status labels below)\n"
        "   - Matched Requirements: list of matched_by_requirements"

        "3. Findings detail: one section per NON-COMPLIANT or NOT CONFIGURED setting (see below)\n\n"

        "## Status labels\n"
        "   - COMPLIANT: tenant value satisfies the CIS recommendation\n"
        "   - NON-COMPLIANT: tenant value deviates from the CIS recommendation\n"
        "   - NOT CONFIGURED: CIS recommends this setting but it is absent from all tenant policies\n"
        "   - NOT IN BENCHMARK: There is no CIS recommendation for the setting configured in the tenant"
        
        "## Findings detail\n"
        "For each NON-COMPLIANT setting:\n"
        "- Current tenant value and CIS recommended value\n"
        "- CIS rationale for why this value matters\n"
        "- Remediation path\n\n"
        "For each NOT CONFIGURED setting:\n"
        "- Flag as a gap: the control provides no protection\n"
        "- CIS recommended value, rationale, and remediation path\n\n"

        "## Grounding rules\n"
        "- Configured values and compliance verdicts must come from tool output only — never from memory.\n"
    ),
        "tools": [search_benchmark, compare_requirements_results],
    "model": model,
}

bench_agent = create_deep_agent(
    middleware=[log_before_model, log_after_model, tool_logger],
    system_prompt=(
        "You are a CIS Benchmark compliance analyst for Microsoft Intune. "
        "Your job is to map security policy requirements to CIS Benchmark controls "
        "and check whether the tenant is compliant.\n\n"

        "## Steps\n"
        "1. Call search_cis_benchmark() with NO arguments — it reads policy_requirements.json "
        "from the virtual filesystem automatically and returns matching CIS controls "
        "per requirement.\n"
        "2. Use the write_file tool and write the exact result fron search_cis_benchmark() into a file"
        "called 'requirements_vs_benchmark.json'"
        "3. Call to compare_search_results_to_tenant() — "
        "it checks each matched CIS control against the tenant's configured policies "
        "and returns a compliance verdict per setting.\n"
        "4. Use the write_file tool and write the exact result of compare_search_results_to_tenant()"
        " into a file called 'tenant_configs_vs_benchmark.json'"
        "5. Present results as a compliance table with columns:\n"
        "   Setting ID | Setting Name | CIS ID | Configured Value | "
        "   CIS Recommended Value | Status | Tenant Policy Name\n"
        "6. Use these status labels:\n"
        "   - COMPLIANT: tenant value satisfies the CIS recommendation\n"
        "   - NON-COMPLIANT: tenant value deviates from the CIS recommendation\n"
        "   - NOT CONFIGURED: CIS recommends this setting but it is absent from all tenant policies\n"

        "## For each NON-COMPLIANT setting\n"
        "- State the current tenant value and the CIS recommended value.\n"
        "- State the CIS rationale for why this value matters.\n"
        "- State the remediation path from the CIS benchmark data.\n\n"

        "## For each NOT CONFIGURED setting\n"
        "- Flag this as a gap: the control provides no protection.\n"
        "- State what CIS recommends, why it matters (rationale), and how to remediate.\n\n"

        "## Important\n"
        "Only use data from tool output. "
        "Do not generate remediation steps from your own knowledge."
    ),
    model=model,
    tools=[search_benchmark, compare_search_results_to_tenant, compare_requirements_results])

def _file_data(path: str) -> dict:
    """Wrap a file's content in the FileData format deepagents expects."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    with open(path) as f:
        lines = f.read().splitlines()
    return {"content": lines, "created_at": now, "modified_at": now}

if __name__ == "__main__":
    result = bench_agent.invoke({
        "messages": [{"role": "user", "content": "Check the CIS benchmark compliance for the security requirements in policy_requirements.json."}],
        "files": {"policy_requirements.json": _file_data(os.path.join(os.path.dirname(__file__), "policy_requirements.json"))},
    })
    