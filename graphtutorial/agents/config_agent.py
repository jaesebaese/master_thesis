from langchain.chat_models import init_chat_model
from langchain.tools import tool
import json
import os

OLLAMA_MODEL = "llama3.1:latest"


# Initialize the model
model = init_chat_model(model=OLLAMA_MODEL, model_provider="ollama", temperature=0.0)


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
def analyze_configs(query: str) -> str:
    """Retrieve all security configuration policies. Returns policy names,
    descriptions, platforms and technologies so you can analyze which ones
    are relevant to the query."""

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
        "- If the user asks for an OVERVIEW of all policies or wants to know "
        "what policies exist: call analyze_configs and return explanations for each configuration.\n"

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
        "Be precise but avoid unexplained acronyms."
    ),
    "tools": [analyze_configs],
}

"""if __name__ == "__main__":
    result = analyze_configs.invoke("Explain all currently configured policies and settings in the tenant, with a focus on security implications.")
    print(result)"""