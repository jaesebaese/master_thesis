from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from typing import List
import json
import os
from policy_agent import PolicyAgentResults

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



config_agent = {
    "name": "config_agent",
    "description": (
        "Retrieves and analyzes Intune configuration policies. "
        "Use the analyze_configs tool to get detailed explanations of currently configured policies and settings in the tenant. "
    ),
    "system_prompt": (
        "You are a Microsoft Intune configuration analyst. "
        "Your job is to retrieve and explain what is currently configured "
        "in the tenant — never answer from memory.\n\n"

        "## Tool selection\n"
        "- To check which policy settings are configured in the tenant: "
        "call find_configs_in_policies() with NO arguments — it reads "
        "policy_results.json written by policy_agent automatically. "
        "Return 'found' items as CONFIGURED and 'missing' items as NOT CONFIGURED.\n"
        "- If the user asks for an OVERVIEW of all policies or wants to know "
        "what policies exist: call analyze_configs.\n"

        "## When explaining a policy\n"
        "For each setting in the returned JSON:\n"
        "1. State the setting name and what it controls.\n"
        "2. State the currently configured value and its security effect.\n"
        "3. If 'available_options' is present, note which option was chosen "
        "and why it matters from a security perspective.\n"
        "4. If 'depends_on' is present, note that this setting requires "
        "another setting to be active — flag this as a dependency.\n"
        "5. If 'activates' is present, note that this setting enables "
        "child settings — list them.\n\n"

        "## Output structure\n"
        "Group settings by category (e.g. BitLocker, Firewall, Defender). "
        "End with a one-paragraph security posture summary. "
        "Be precise but avoid unexplained acronyms.\n\n"

        "## Saving output\n"
        "After calling find_configs_in_policies(), call write_file with:\n"
        "  path: 'relevant_configs.json'\n"
        "  content: the raw JSON string returned by find_configs_in_policies\n"
        "Do this before summarising the results."
    ),
    "tools": [find_configs_in_policies, analyze_configs],
}

"""if __name__ == "__main__":
    result = analyze_configs.invoke()
    print(result)"""