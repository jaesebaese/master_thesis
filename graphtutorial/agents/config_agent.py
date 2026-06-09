import logging

from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from typing import List
import json
import os
from benchmark_agent import _file_data
from rich_renderer import RichRenderer
from preprocessing_at_startup import build_tenant_collection, flatten_for_relevance, TENANT_SETTINGS_JSON, INTUNE_SETTINGS_JSON
from activity_stream import astream_activity
import asyncio

OPENAI_MODEL = "gpt-5.4-nano-2026-03-17"


# Initialize the model
model = init_chat_model(model=OPENAI_MODEL, model_provider="openai", temperature=0.0)


@tool
def find_configs_in_policies(runtime: ToolRuntime) -> str:
    """Check which configured settings appear in policies_and_settings_expand.json.

    When configs is omitted, reads policy_results.json from the virtual
    filesystem (written by policy_agent) or from disk as a fallback.
    When configs is provided explicitly, uses that list directly.

    Each item must have at least an "id" field matching a settingDefinitionId
    in the policy settings tree.

    Returns:
      {
        "found":   [{config, policy_name, policy_id}, ...],
        "missing": [config, ...]
      }
    """
    if not configs:
        files = runtime.state.get("files", {})
        file_entry = files.get("/policy_results.json") or files.get("policy_results.json")
        if file_entry is not None:
            if isinstance(file_entry, dict):
                raw = file_entry.get("content", [])
                content_str = "\n".join(raw) if isinstance(raw, list) else str(raw)
            else:
                content_str = str(file_entry)
            data = json.loads(content_str)
        else:
            disk_path = os.path.join(os.path.dirname(__file__), "policy_results.json")
            try:
                with open(disk_path) as f:
                    data = json.load(f)
            except FileNotFoundError:
                return json.dumps({"error": "policy_results.json not found. Run policy_agent first."})

        settings = data.get("settings", data) if isinstance(data, dict) else data
        configs = settings

    path = os.path.join(
        os.path.dirname(__file__), "../configurations/policies_and_settings_expand.json"
    )
    with open(path) as f:
        policies = json.load(f)

    def _collect_ids(instance: dict, out: set) -> None:
        sid = instance.get("settingDefinitionId")
        if sid:
            out.add(sid)
        for val in instance.values():
            if isinstance(val, dict):
                _collect_ids(val, out)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        _collect_ids(item, out)

    # Build map: settingDefinitionId → list of {policy_name, policy_id}
    id_to_policies: dict[str, list[dict]] = {}
    for policy in policies:
        policy_info = {"policy_name": policy.get("name", ""), "policy_id": policy.get("id", "")}
        setting_ids: set[str] = set()
        for setting in policy.get("settings", []):
            _collect_ids(setting.get("settingInstance", {}), setting_ids)
        for sid in setting_ids:
            id_to_policies.setdefault(sid, []).append(policy_info)

    found = []
    missing = []
    for config in configs:
        if hasattr(config, "id"):
            config_id = config.id
            config_dict = config.model_dump() if hasattr(config, "model_dump") else vars(config)
        elif isinstance(config, dict):
            config_id = config.get("id", "")
            config_dict = config
        else:
            # LLM passed a raw string (the setting definition ID itself)
            config_id = str(config)
            config_dict = {"id": config_id}
        matches = id_to_policies.get(config_id)
        if matches:
            for m in matches:
                found.append({**m, "config": config_dict})
        else:
            missing.append(config_dict)

    result = {"found": found, "missing": missing}
    relevant_configs_path = os.path.join(os.path.dirname(__file__), "relevant_configs.json")
    with open(relevant_configs_path, "w") as f:
        json.dump(result, f, indent=2)
    return json.dumps(result, indent=2)


@tool
def find_configs_in_tenant(runtime: ToolRuntime) -> str:
    """Semantically match each security requirement from policy_requirements.json
    against the tenant's configured settings via a single batched vector search.

    Mirrors search_cis_benchmark's approach: all requirement queries are sent to
    ChromaDB in one call, no LLM step. Returns a list — one entry per requirement
    (including those with no matches, which get an empty tenant_matches array) —
    each with a 'tenant_matches' array of the most similar tenant settings.

    Each match includes: setting_id, setting_name, configured_value_label,
    configured_value, policy_name, similarity_score.
    """
    with open(INTUNE_SETTINGS_JSON) as f:
        catalog = {s["id"]: s for s in json.load(f)}

    with open(TENANT_SETTINGS_JSON) as f:
        policies = json.load(f)

    tenant_flat = flatten_for_relevance(policies, catalog)
    tenant_index: dict[str, list[dict]] = {}
    for s in tenant_flat:
        tenant_index.setdefault(s["id"], []).append(s)

    files = runtime.state.get("files", {})
    file_entry = files.get("/policy_requirements.json") or files.get("policy_requirements.json")
    if file_entry is None:
        return json.dumps({"error": "policy_requirements.json not found. Ensure policy_agent has run first."})

    if isinstance(file_entry, dict):
        raw = file_entry.get("content", [])
        requirements_str = "\n".join(raw) if isinstance(raw, list) else str(raw)
    else:
        requirements_str = str(file_entry)

    try:
        parsed = json.loads(requirements_str)
    except json.JSONDecodeError:
        import ast
        parsed = ast.literal_eval(requirements_str)
    req_list: list[dict] = parsed.get("requirements", parsed) if isinstance(parsed, dict) else parsed

    tenant_collection = build_tenant_collection(platform="windows")  # ensure the collection is built and get the instance

    query_texts = [
        f"{r.get('source_text', '')} {r.get('control_intent', '')}".strip()
        for r in req_list
    ]
    batch_results = tenant_collection.query(
        query_texts=query_texts,
        n_results=5,
        include=["metadatas", "distances"],
    )

    output = []
    for req, metadatas, distances in zip(req_list, batch_results["metadatas"], batch_results["distances"]):
        matches = []
        for meta, dist in zip(metadatas, distances):
            score = round(1 - dist, 4)
            if score < 0.50:
                continue
            cid = meta.get("id", "")
            tenant_hits = tenant_index.get(cid, [])
            matches.append({
                "setting_id": cid,
                "setting_name": meta.get("name", cid),
                "configured_value_label": meta.get("configured_value_label", ""),
                "configured_value": tenant_hits[0].get("configured_value") if tenant_hits else None,
                "description": meta.get("description", ""),
                "policy_name": meta.get("policy_name", ""),
                "similarity_score": score,
            })
        output.append({
            "requirement_id": req.get("requirement_id"),
            "source_text": req.get("source_text"),
            "expected_value": req.get("expected_value"),
            "expected_unit": req.get("expected_unit"),
            "operator": req.get("operator"),
            "strength": req.get("strength"),
            "security_domain": req.get("security_domain"),
            "control_intent": req.get("control_intent"),
            "tenant_matches": matches,
        })

    return json.dumps(output, indent=2)


@tool
def evaluate_requirements_compliance(runtime: ToolRuntime) -> str:
    """Read requirements_analysrelevant_configurationsis_tenant.json (output of find_configs_in_tenant) and make a
    single LLM call to classify each requirement as satisfied, violated, or not_configured
    based on the tenant's Intune settings.
    """
    files = runtime.state.get("files", {})
    file_entry = (
        files.get("/relevant_configurations.json")
        or files.get("relevant_configurations.json")
    )

    if file_entry is None:
        disk_path = os.path.join(os.path.dirname(__file__), "relevant_configurations.json")
        try:
            with open(disk_path) as f:
                requirements_list = json.load(f)
        except FileNotFoundError:
            return json.dumps({"error": "relevant_configurations.json not found. Run find_configs_in_tenant first."})
    else:
        if isinstance(file_entry, dict):
            raw = file_entry.get("content", [])
            requirements_str = "\n".join(raw) if isinstance(raw, list) else str(raw)
        else:
            requirements_str = str(file_entry)
        requirements_list = json.loads(requirements_str, strict=False)

    user_prompt = (
        "Classify each requirement's compliance status based on the tenant settings.\n\n"
        f"REQUIREMENTS AND MATCHED TENANT SETTINGS:\n{json.dumps(requirements_list, indent=2)}"
    )

    system_prompt = """You are a security compliance analyst evaluating Microsoft Intune tenant settings against security policy requirements.

For each requirement, determine the compliance status using the matched tenant settings.

## Status Definitions

### satisfied

The tenant has at least one relevant setting that fulfills the requirement.

Rules:

* For minimum-style requirements ("at least", "minimum", "greater than or equal to"), the configured value must be greater than or equal to the expected value.
* For maximum-style requirements ("maximum", "no more than", "up to", "less than or equal to"), the configured value must be less than or equal to the expected value.
* For exact-match requirements ("exactly", "must be"), the configured value must equal the expected value.
* A stricter configuration than required should be considered satisfied when it still complies with the requirement intent.

### violated

The tenant has one or more relevant settings for the requirement, but none satisfy the requirement.

Examples:

* A numeric value does not meet the required constraint.
* A required security feature is disabled.
* A configured version, age, length, or threshold is weaker than required.

### partially_satisfied

* Relevant settings exist and contribute to the requirement, but none fully implement the predicate, OR the predicate can't be evaluated against the available settings.

### not_configured

The tenant does not have any relevant settings that implement the requirement.

Use this status when:

* tenant_matches is empty.
* No matched settings are semantically related to the requirement at all.

## Evaluation Rules

1. Use source_text, expected_value, expected_unit, operator, and control_intent to determine the requirement's intent.
2. When expected_unit represents a measurable quantity (for example: days, characters, attempts, versions), compare values numerically whenever possible.
3. Use the setting_name, configured_value, configured_value_label, and description for comparison.
4. Only consider settings that are semantically relevant to the requirement.
5. If multiple relevant settings exist:
   * If any direct setting exists: evaluate the predicate against direct settings only. 
   * If only indirect settings exist: return partially_satisfied with an explanation of what they cover and what they miss.
   * If only irrelevant settings exist: return not_configured.
6. If no relevant settings exist, return "not_configured". And specify in the explanation why none of the existing settings were relevant. 

## Severity Rules

* status = "violated" → severity = "finding"
* status = "satisfied" or "partially_satisfied" or "not_configured" → severity = "informational"

## Contributing Settings

* Include only the settings that influenced the decision.
* For "not_configured", return an empty array.

## Output Requirements

Return ONLY a valid JSON array.
Do not include explanations outside the JSON.
Do not include markdown fences.

Output schema:

[
   {
      "requirement_id": "REQ-001",
      "source_text": "Passwords of priviledged user accountsmust have a minimum length of twelve characters.",
      "expected_value": "12",
      "expected_unit": "characters",
      "operator": "minimum",
      "strength": "mandatory",
      "security_domain": "authentication",
      "control_intent": "enforce_password_min_length_and_charset_complexity",
      "tenant_matches": [
         {
            "setting_id": "device_vendor_msft_laps_policies_passwordlength",
            "setting_name": "Password Length ",
            "configured_value_label": "10",
            "configured_value": 10,
            "description": "Use this setting to configure the length of the password of the managed local administrator account.\n\nIf not specified, this setting will default to 14 characters.\n\nThis setting has a minimum allowed ",
            "policy_name": "CIS - Windows LAPS [L1] - Windows 11 - v4.0.0.0",
            "similarity_score": 0.5542
         },
         ...
      ],
      "status": "violated",
      "severity": "finding",  
      "explanation": "The 'Password Length' setting is configured to 10, which does not meet the minimum requirement of 12 characters, but it is a relevant setting that partially addresses the requirement."    
]
"""

    response = model.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ])

    content = response.content
    if isinstance(content, list):
        content = "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    raw = (content or "").strip()
    if raw.startswith("```"):
        raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON from model: {e}", "raw_output": raw[:200]})

    return json.dumps(result, indent=2)


config_agent = {
    "name": "config_agent",
    "description": (
        "Retrieves Intune tenant configurations and evaluates security policy compliance. "
        "Always runs two steps: (1) find_configs_in_tenant to match security requirements "
        "against tenant settings via vector search, then (2) evaluate_requirements_compliance "
        "to classify each requirement as satisfied, violated, or not_configured. "
    ),
    "system_prompt": (
    "You are a Microsoft Intune configuration analyst. "
    "Answer only from live tenant data — never from memory or assumptions.\n\n"

    "## Workflow\n"
    "You MUST always complete ALL five steps below in order, regardless of what the task description says. "
    "Do not skip any step. Do not proceed to the next step until the current one is done. "
    "Only after completing all steps, return the final summary.\n\n"

    "Step 1 — Retrieve tenant configurations:\n"
    "Call find_configs_in_tenant (no arguments). "
    "It reads policy_requirements.json and returns a list of requirements, each with a "
    "'tenant_matches' array. Each match contains: setting_id, setting_name, "
    "configured_value_label, configured_value, policy_name, similarity_score.\n\n"

    "Step 2 — Save the results:\n"
    "Write the full JSON response from find_configs_in_tenant to: relevant_configurations.json"
    "Make sure that the JSON is valid and properly formatted.\n\n"

    "Step 3 — Classify compliance:\n"
    "Only after Step 2 is complete, call evaluate_requirements_compliance (no arguments). "
    "It classifies each requirement as satisfied, violated, or not_configured.\n\n"

    "Step 4 — Summarize findings:\n"
    "Write the full JSON response of evaluate_requirements_compliance to: requirements_vs_tenant.json\n\n"

    "Step 5 — Return a summary table:\n"
    "Once all the files tools are finished and the files are written,"
    "produce a markdown table with these exact columns, in this order:\n"
    "| Requirement ID | Requirement Description | Expected Value | Tenant Configured Value | "
    "Setting Id | Setting Name | Setting Description | Policy Name | Compliance Status |\n\n"

    "Use all the information from requirements_vs_tenant.json to fill out the information in the table."

    "Column definitions:\n"
    "- Requirement ID: requirement_id \n"
    "- Requirement Description: the requirement's source_text field\n"
    "- Expected Value: the expected_value\n"
    "- Tenant Configured Value: configured_value_label from the tenant match\n"
    "- Setting Id: setting_id from the tenant match\n"
    "- Setting Name: setting_name from the tenant match\n"
    "- Setting Description: a one-sentence summary of the description from the tenant match\n"
    "- Policy Name: policy_name from the tenant match\n"
    "- Compliance Status: from requirements_vs_tenant.json\n\n"
    

    "After the table, if any requirements are violated or not_configured, "
    "add a brief bulleted list of recommended remediation steps. "
    "Do not re-explain findings already visible in the table.\n"
        ),
    "tools": [find_configs_in_tenant, evaluate_requirements_compliance],
    "model": model,
}

co_agent = create_deep_agent(system_prompt=config_agent["system_prompt"], tools=config_agent["tools"], model=config_agent["model"])

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    renderer = RichRenderer(logger=logger)

    #result = stream_agent_v2(agent, pending, config=run_config, on_interrupt=handle_interrupt)
    pending = {
        "messages": [{"role": "user", "content": "Check interdependecies in the security configurations"}],
        "files": {
            "policy_requirements.json": _file_data(os.path.join(os.path.dirname(__file__), "policy_requirements.json"))
        },
    }
    run_config = {"configurable": {"thread_id": "1"}}

    final_state = asyncio.run(
        astream_activity(co_agent, agent_input=pending, config=run_config, render=False, on_event=renderer)
    )
    print("\nFINAL STATE:\n" + json.dumps(final_state, indent=2))