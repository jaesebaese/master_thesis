from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from typing import List
import json
import os
from policy_agent import PolicyAgentResults, SETTINGS_JSON
from preprocessing_at_startup import build_tenant_collection, flatten_for_relevance

OPENAI_MODEL = "gpt-5.4-nano-2026-03-17"


# Initialize the model
model = init_chat_model(model=OPENAI_MODEL, model_provider="openai", temperature=0.0)


def explain_policy_settings(policy_name: str) -> str:
    """Look up a policy by name in policies_and_settings_expand.json, resolve each
    configured setting against the full Intune settings definition catalog, and
    return a structured JSON payload describing what every setting does and which
    value is currently configured.  Pass this payload to the LLM to get a plain-
    English explanation.

    Args:
        policy_name: Full or partial name of the policy (case-insensitive match).
    """

    config_path = os.path.join(
        os.path.dirname(__file__), "../configurations/policies_and_settings_expand.json"
    )
    defs_path = os.path.join(
        os.path.dirname(__file__),
        "../intune_configurations/intune_configuration_settings.json",
    )

    with open(config_path) as f:
        policies = json.load(f)

    # Build a lookup dict once: definition id → definition object
    with open(defs_path) as f:
        definitions: dict = {d["id"]: d for d in json.load(f)}

    # Find the first policy whose name contains policy_name (case-insensitive)
    policy = next(
        (p for p in policies if policy_name.lower() in p.get("name", "").lower()),
        None,
    )
    if not policy:
        available = [p["name"] for p in policies]
        return json.dumps({"error": f"No policy found matching '{policy_name}'.", "available_policies": available})

    # ------------------------------------------------------------------ #
    # Recursively walk the settingInstance tree and collect every leaf    #
    # ------------------------------------------------------------------ #
    extracted: list[dict] = []

    #TODO: make sure that the children don't count as separate settings in the final output - they should be nested under the parent setting they depend on, with an explanation that they only apply if the parent setting is configured a certain way.
    def _extract(instance: dict) -> None:
        dtype = instance.get("@odata.type", "")
        sid = instance.get("settingDefinitionId", "")

        if "ChoiceSettingInstance" in dtype:
            chosen_id = instance.get("choiceSettingValue", {}).get("value", "")
            extracted.append({"definitionId": sid, "chosenOptionId": chosen_id, "type": "choice"})
            # A choice can have dependent children (e.g. sub-settings unlocked by a specific option)
            for child in instance.get("choiceSettingValue", {}).get("children", []):
                _extract(child)

        elif "GroupSettingCollectionInstance" in dtype:
            extracted.append({"definitionId": sid, "chosenOptionId": None, "type": "group"})
            for group_val in instance.get("groupSettingCollectionValue", []):
                for child in group_val.get("children", []):
                    _extract(child)

        elif "GroupSettingInstance" in dtype:
            extracted.append({"definitionId": sid, "chosenOptionId": None, "type": "group"})
            for child in instance.get("groupSettingValue", {}).get("children", []):
                _extract(child)

        elif "SimpleSettingCollectionInstance" in dtype:
            values = [v.get("value") for v in instance.get("simpleSettingCollectionValue", [])]
            extracted.append({"definitionId": sid, "chosenOptionId": values, "type": "simpleCollection"})

        elif "SimpleSettingInstance" in dtype:
            value = instance.get("simpleSettingValue", {}).get("value", "")
            extracted.append({"definitionId": sid, "chosenOptionId": value, "type": "simple"})

        else:
            # Fallback: record what we have so nothing is silently dropped
            extracted.append({"definitionId": sid, "chosenOptionId": None, "type": "unknown"})

    for setting in policy.get("settings", []):
        _extract(setting.get("settingInstance", {}))

    # ------------------------------------------------------------------ #
    # Enrich each extracted item with human-readable metadata             #
    # ------------------------------------------------------------------ #
    explanations = []
    for item in extracted:
        defn = definitions.get(item["definitionId"])
        if defn is None:
            # Definition not found in the catalog – still surface it
            explanations.append({
                "setting_id": item["definitionId"],
                "setting_name": item["definitionId"],
                "description": "Definition not found in the local catalog.",
                "configured_value": item["chosenOptionId"],
                "configured_value_label": str(item["chosenOptionId"]),
            })
            continue

        entry: dict = {
            "setting_id": item["definitionId"],
            "setting_name": defn.get("displayName") or defn.get("name") or item["definitionId"],
            "description": defn.get("description") or defn.get("helpText") or "",
            "configured_value": item["chosenOptionId"],
            "configured_value_label": str(item["chosenOptionId"]),  # default
        }

        #print(entry)

        # For choice settings resolve the selected option's display name
        if item["type"] == "choice" and item["chosenOptionId"]:
            option_map = {
                o["itemId"]: (o.get("displayName") or o.get("name") or o["itemId"])
                for o in defn.get("options", [])
            }
            entry["configured_value_label"] = option_map.get(
                item["chosenOptionId"], item["chosenOptionId"]
            )
            # Also expose all available options so the LLM understands the full range
            entry["available_options"] = list(option_map.values())

        explanations.append(entry)

    result = {
        "policy_name": policy["name"],
        "description": policy.get("description", ""),
        "platform": policy.get("platforms", ""),
        "technologies": policy.get("technologies", ""),
        "settings_count": len(explanations),
        "settings": explanations,
    }
    return json.dumps(result, indent=2)

@tool
def find_configs_in_policies(runtime: ToolRuntime, configs: List[PolicyAgentResults] | None = None) -> str:
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

        # PolicyAgentResults returns {"settings": [...]}; support flat list too
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
def analyze_configs() -> str:
    """Retrieve and explain all currently configured security policies in the tenant.
    Returns structured JSON with every setting name, configured value, and available
    options for each policy."""

    path = os.path.join(os.path.dirname(__file__), "../configurations", "policies_and_settings_expand.json")
    with open(path, "r") as f:
        configurations = json.load(f)

    all_explanations = []
    for config in configurations:
        name = config.get("name", "")
        explanation_json = explain_policy_settings(name)
        all_explanations.append(json.loads(explanation_json))

    return json.dumps(all_explanations, indent=2)



@tool
def analyze_requirements_against_tenant(runtime: ToolRuntime) -> str:
    """Semantically match each security requirement from policy_requirements.json
    against the tenant's configured settings via a single batched vector search.

    Mirrors search_cis_benchmark's approach: all requirement queries are sent to
    ChromaDB in one call, no LLM step. Returns a list — one entry per requirement
    (including those with no matches, which get an empty tenant_matches array) —
    each with a 'tenant_matches' array of the most similar tenant settings.

    Each match includes: setting_id, setting_name, configured_value_label,
    configured_value, policy_name, similarity_score.
    """
    with open(SETTINGS_JSON) as f:
        catalog = {s["id"]: s for s in json.load(f)}

    tenant_path = os.path.join(
        os.path.dirname(__file__), "../configurations/policies_and_settings_expand_assign.json"
    )
    with open(tenant_path) as f:
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

    tenant_collection = build_tenant_collection(platform="windows")

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
                "policy_name": meta.get("policy_name", ""),
                "similarity_score": score,
            })
        output.append({
            "requirement_id": req.get("requirement_id"),
            "source_text": req.get("source_text"),
            "security_domain": req.get("security_domain"),
            "control_intent": req.get("control_intent"),
            "tenant_matches": matches,
        })

    return json.dumps(output, indent=2)


config_agent = {
    "name": "config_agent",
    "description": (
        "Retrieves and analyzes Intune configuration policies. "
        "Use analyze_requirements_against_tenant to semantically match security requirements "
        "against the tenant's configured settings via vector search."
    ),
    "system_prompt": (
        "You are a Microsoft Intune configuration analyst. "
        "Your job is to retrieve and explain what is currently configured "
        "in the tenant — never answer from memory.\n\n"

        "## Requirements compliance\n"
        "To check how well the tenant satisfies the security policy requirements, "
        "call analyze_requirements_against_tenant (no arguments). "
        "It reads policy_requirements.json from the virtual filesystem and returns a list — "
        "one entry per requirement — each with a 'tenant_matches' array of the most similar "
        "tenant settings. Each match has: setting_id, setting_name, configured_value_label, "
        "configured_value, policy_name, similarity_score.\n "
        "IMPORTANT: After the tool returns, write the full JSON result to a new file with the name: 'requirements_analysis_tenant.json'.\n"
        "For each requirement, report which settings matched and their configured values.\n\n"

    ),
    "tools": [analyze_requirements_against_tenant],
    "model": model,
}

if __name__ == "__main__":
    result = analyze_requirements_against_tenant.invoke({"messages": [{"role": "user"}]})
    print(result)