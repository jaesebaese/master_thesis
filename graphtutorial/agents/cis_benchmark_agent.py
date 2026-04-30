from langchain.chat_models import init_chat_model
from langchain.tools import tool
import json
import os


OLLAMA_MODEL = "llama3.1:latest"

# Initialize the model
model = init_chat_model(model=OLLAMA_MODEL, model_provider="ollama", temperature=0.0)


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
    in matched_device_lock_configurations.json and produces a per-setting compliance
    report. Each setting is marked compliant, non_compliant, or not_configured.

    Args:
        query: Optional context passed through for LLM use; not used for filtering.
    """
    base = os.path.dirname(__file__)

    cis_benchmark_path = os.path.join(
        base,
        "../configurations/cis_benchmarks/matched_device_lock_configurations.json",
    )
    policies_path = os.path.join(
        base, "../configurations/policies_and_settings_expand.json"
    )

    with open(cis_benchmark_path) as f:
        cis_benchmark = json.load(f)

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
    for item in cis_benchmark.get("matched_configurations", []):
        sid = item["setting_definition_id"]
        cis_raw = item["raw_value"]
        benchmark = item["cis_benchmark"]

        tenant_hits = setting_index.get(sid, [])

        if not tenant_hits:
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
            "cis_id": benchmark["cis_id"],
            "cis_title": benchmark["cis_title"],
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
    }

    return json.dumps({
        "benchmark": cis_benchmark["policy_metadata"],
        "summary": summary,
        "results": results,
    }, indent=2)


cis_benchmark_agent = {
    "name": "cis_benchmark_agent",
    "description": (
        "Retrieves and analyzes CIS benchmark policies. "
        "Uses the compare_to_cis_benchmark tool to scan tenant policies for compliance with the CIS benchmark and produces a detailed report."
    ),
    "system_prompt": (
        "You are a CIS Benchmark compliance analyst for Microsoft Intune. "
        "Your job is to compare the tenant's current configuration against "
        "CIS Benchmark recommendations and report gaps clearly.\n\n"

        "## Steps\n"
        "1. Call compare_to_cis_benchmark with the query.\n"
        "2. Present results as a compliance table with columns:\n"
        "   CIS ID | Setting Name | Configured Value | "
        "   CIS Recommended | Status\n"
        "3. Use these status labels:\n"
        "   - COMPLIANT: configured value matches CIS recommendation\n"
        "   - NON-COMPLIANT: configured value deviates from recommendation\n"
        "   - NOT CONFIGURED: setting is absent from all tenant policies\n\n"

        "## For each NON-COMPLIANT setting\n"
        "- State the current value and what the CIS recommendation is.\n"
        "- State the CIS rationale for why this value matters.\n"
        "- State the remediation path from the CIS benchmark data.\n"
        "- Flag if changing this setting may affect dependent settings "
        "(the supervisor will check interdependencies separately).\n\n"

        "## For each NOT CONFIGURED setting\n"
        "- Note that the setting is absent from all policies.\n"
        "- State what value CIS recommends and why.\n"
        "- This is higher priority than non-compliant — "
        "the setting provides no protection at all.\n\n"

        "## Important\n"
        "Only use data from the compare_to_cis_benchmark tool output. "
        "Do not generate remediation steps from your own knowledge. "
        "If the tool output does not include a remediation path, "
        "state that the remediation path was not available in the data."
    ),
    "tools": [compare_to_cis_benchmark],
}

if __name__ == "__main__":
    result = compare_to_cis_benchmark.invoke("Show how many of my policies are compliant with the CIS benchmark and which ones are not, with details.")
    print(result)