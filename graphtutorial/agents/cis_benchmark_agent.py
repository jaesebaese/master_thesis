from langchain.chat_models import init_chat_model
from langchain.tools import tool
import json
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_MODEL = "minimax-m2.5:cloud"
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
def compare_relevant_settings_to_cis_benchmark(settings_to_check: list[dict]) -> str:
    """Compare a list of configured tenant settings against CIS benchmarks.
    
    Takes the output of tenant_policy_agent (configured settings relevant to
    the current query) and checks each one against the pre-built CIS-to-Intune
    mapping. Settings not covered by any CIS benchmark are marked
    not_in_benchmark; settings marked as Manual in CIS are flagged for human
    review rather than automatic verification.
    
    Args:
        settings_to_check: A list of configured settings (from
            tenant_policy_agent). Each item must contain:
              - id: setting definition ID
              - configured_value: current configured value in tenant
              - configured_value_label (optional): human-readable label
              - name (optional): setting display name
              - policy_name (optional): policy that configures it
              - policy_id (optional): policy ID
    
    Returns:
        JSON string with summary stats and per-setting compliance findings.
    """
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
        "Retrieves and analyzes CIS benchmark policies. "
        "Uses the compare_to_cis_benchmark tool to scan tenant policies for compliance with the CIS benchmark and produces a detailed report."
    ),
    "system_prompt": (
        "You are a CIS Benchmark compliance analyst for Microsoft Intune. "
        "Your job is to compare the tenant's current configuration against "
        "CIS Benchmark recommendations and report gaps clearly.\n\n"

        "## Steps\n"
        "1. Call either compare_to_cis_benchmark or compare_relevant_settings_to_cis_benchmark with the query. \n"
        "   - Use compare_to_cis_benchmark for a comprehensive scan of all tenant policies against the CIS benchmark. This is useful for an overall compliance assessment.\n"
        "   - Use compare_relevant_settings_to_cis_benchmark to scan specific settings against the CIS benchmark. This is useful for targeted compliance checks.\n"
        "2. Present results as a compliance table with columns:\n"
        "   CIS ID | Benchmark Policy | Setting Name | Configured Value | "
        "   CIS Recommended | Status\n"
        "3. Use these status labels:\n"
        "   - COMPLIANT: configured value matches CIS recommendation\n"
        "   - NON-COMPLIANT: configured value deviates from recommendation\n"
        "   - NOT CONFIGURED: setting is absent from all tenant policies\n"
        "   - MANUAL CHECK REQUIRED: CIS marks this as a manual control that "
        "cannot be verified automatically\n\n"

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

        "## For each MANUAL CHECK REQUIRED setting\n"
        "- Note that this control cannot be verified programmatically.\n"
        "- State what CIS recommends and the audit procedure if available.\n\n"

        "## Important\n"
        "Only use data from the compare_to_cis_benchmark or compare_relevant_settings_to_cis_benchmark tool output. "
        "Do not generate remediation steps from your own knowledge. "
        "If the tool output does not include a remediation path, "
        "state that the remediation path was not available in the data."
    ),
    "tools": [compare_to_cis_benchmark, compare_relevant_settings_to_cis_benchmark],
}

if __name__ == "__main__":
    relevant_settings = [
      {
      "id": "device_vendor_msft_policy_config_credentialproviders_blockpicturepassword",
      "name": "Turn off picture password sign-in",
      "description": "This policy setting allows you to control whether a domain user can sign in using a picture password.\r\n\r\nIf you enable this policy setting, a domain user can't set up or sign in with a picture password. \r\n\r\nIf you disable or don't configure this policy setting, a domain user can set up and use a picture password.\r\n\r\nNote that the user's domain password will be cached in the system vault when using this feature.",
      "platform": "windows10",
      "type": "choice",
      "configured_value": "device_vendor_msft_policy_config_credentialproviders_blockpicturepassword_1",
      "configured_value_label": "Enabled",
      "parent_setting_id": "null",
      "parent_chain": [],
      "policy_name": "CIS - System - Restrictions [L1] - Windows 11 - v4.0.0.0",
      "policy_id": "dcffa355-f249-4bf8-b386-6be7f5c1945f",
      "available_options": [
        "Disabled",
        "Enabled"
      ]
    },
    {
      "id": "device_vendor_msft_policy_config_remotedesktopservices_donotallowpasswordsaving",
      "name": "Do not allow passwords to be saved",
      "description": "Controls whether passwords can be saved on this computer from Remote Desktop Connection.\r\n\r\nIf you enable this setting the password saving checkbox in Remote Desktop Connection will be disabled and users will no longer be able to save passwords. When a user opens an RDP file using Remote Desktop Connection and saves his settings, any password that previously existed in the RDP file will be deleted.\r\n\r\nIf you disable this setting or leave it not configured, the user will be able to save passwords using Remote Desktop Connection.",
      "platform": "windows10",
      "type": "choice",
      "configured_value": "device_vendor_msft_policy_config_remotedesktopservices_donotallowpasswordsaving_1",
      "configured_value_label": "Enabled",
      "parent_setting_id": "null",
      "parent_chain": [],
      "policy_name": "CIS - Windows Components - Restrictions [L1] - Windows 11 - v4.0.0.0",
      "policy_id": "c4dc50cf-85bf-49ad-a4d7-c594a7096e98",
      "available_options": [
        "Disabled",
        "Enabled"
      ]
    },
    {
      "id": "device_vendor_msft_policy_config_windowslogon_enablemprnotifications",
      "name": "Configure the transmission of the user's password in the content of MPR notifications sent by winlogon.",
      "description": "This policy controls whether the user's password is included in the content of MPR notifications sent by winlogon in the system.\r\n\r\nIf you disable this setting or do not configure it, winlogon sends MPR notifications with empty password fields of the user's authentication info.\r\n\r\nIf you enable this setting, winlogon sends MPR notifications containing the user's password in the authentication info.",
      "platform": "windows10",
      "type": "choice",
      "configured_value": "device_vendor_msft_policy_config_windowslogon_enablemprnotifications_0",
      "configured_value_label": "Disabled",
      "parent_setting_id": "null",
      "parent_chain": [],
      "policy_name": "CIS - Windows Components - Restrictions [L1] - Windows 11 - v4.0.0.0",
      "policy_id": "c4dc50cf-85bf-49ad-a4d7-c594a7096e98",
      "available_options": [
        "Disabled",
        "Enabled"
      ]
    },
    {
      "id": "device_vendor_msft_policy_config_localpoliciessecurityoptions_networksecurity_lanmanagerauthenticationlevel",
      "name": "Network Security LAN Manager Authentication Level",
      "description": "Network security LAN Manager authentication level  This security setting determines which challenge/response authentication protocol is used for network logons. This choice affects the level of authentication protocol used by clients, the level of session security negotiated, and the level of authentication accepted by servers as follows:  Send LM and NTLM responses: Clients use LM and NTLM authentication and never use NTLMv2 session security; domain controllers accept LM, NTLM, and NTLMv2 authentication.  Send LM and NTLM - use NTLMv2 session security if negotiated: Clients use LM and NTLM authentication and use NTLMv2 session security if the server supports it; domain controllers accept LM, NTLM, and NTLMv2 authentication.  Send NTLM response only: Clients use NTLM authentication only and use NTLMv2 session security if the server supports it; domain controllers accept LM, NTLM, and NTLMv2 authentication.  Send NTLMv2 response only: Clients use NTLMv2 authentication only and use NTLMv2 session security if the server supports it; domain controllers accept LM, NTLM, and NTLMv2 authentication.  Send NTLMv2 response only\\refuse LM: Clients use NTLMv2 authentication only and use NTLMv2 session security if the server supports it; domain controllers refuse LM (accept only NTLM and NTLMv2 authentication).  Send NTLMv2 response only\\refuse LM and NTLM: Clients use NTLMv2 authentication only and use NTLMv2 session security if the server supports it; domain controllers refuse LM and NTLM (accept only NTLMv2 authentication).  Important  This setting can affect the ability of computers running Windows 2000 Server, Windows 2000 Professional, Windows XP Professional, and the Windows Server 2003 family to communicate with computers running Windows NT 4.0 and earlier over the network. For example, at the time of this writing, computers running Windows NT 4.0 SP4 and earlier did not support NTLMv2. Computers running Windows 95 and Windows 98 did not support NTLM.  Default:  Windows 2000 and windows XP: send LM and NTLM responses  Windows Server 2003: Send NTLM response only  Windows Vista, Windows Server 2008, Windows 7, and Windows Server 2008 R2: Send NTLMv2 response only",
      "platform": "windows10",
      "type": "choice",
      "configured_value": "device_vendor_msft_policy_config_localpoliciessecurityoptions_networksecurity_lanmanagerauthenticationlevel_5",
      "configured_value_label": "Send NTLMv2 responses only. Refuse LM and NTLM",
      "parent_setting_id": "null",
      "parent_chain": [],
      "policy_name": "ADN - Add-on - Local Policies Security Options [L1] - Windows 11 - v4.0.0.0",
      "policy_id": "972e07f3-be73-4195-9f9e-8f573c07ddcd",
      "available_options": [
        "Send LM and NTLM responses",
        "Send LM and NTLM-use NTLMv2 session security if negotiated",
        "Send LM and NTLM responses only",
        "Send NTLMv2 responses only",
        "Send NTLMv2 responses only. Refuse LM",
        "Send NTLMv2 responses only. Refuse LM and NTLM"
      ]
    },
    {
      "id": "device_vendor_msft_policy_config_localpoliciessecurityoptions_networksecurity_minimumsessionsecurityforntlmsspbasedclients_537395200",
      "name": "Network Security Minimum Session Security For NTLMSSP Based Clients",
      "description": "Network security: Minimum session security for NTLM SSP based (including secure RPC) clients  This security setting allows a client to require the negotiation of 128-bit encryption and/or NTLMv2 session security. These values are dependent on the LAN Manager Authentication Level security setting value. The options are:  Require NTLMv2 session security: The connection will fail if NTLMv2 protocol is not negotiated. Require 128-bit encryption: The connection will fail if strong encryption (128-bit) is not negotiated.  Default:  Windows XP, Windows Vista, Windows 2000 Server, Windows Server 2003, and Windows Server 2008: No requirements.  Windows 7 and Windows Server 2008 R2: Require 128-bit encryption",
      "platform": "windows10",
      "type": "choice",
      "configured_value": "device_vendor_msft_policy_config_localpoliciessecurityoptions_networksecurity_minimumsessionsecurityforntlmsspbasedclients_537395200",
      "configured_value_label": "Require NTLM and 128-bit encryption",
      "parent_setting_id": "null",
      "parent_chain": [],
      "policy_name": "CIS - Local Policies Security Options [L1] - Windows 11 - v4.0.0.0",
      "policy_id": "6af60625-fa62-4a37-b748-a21f24ae6d06",
      "available_options": [
        "None",
        "Require NTLMv2 session security",
        "Require 128-bit encryption",
        "Require NTLM and 128-bit encryption"
      ]
    },
    {
      "id": "device_vendor_msft_policy_config_localpoliciessecurityoptions_networksecurity_minimumsessionsecurityforntlmsspbasedservers_537395200",
      "name": "Network Security Minimum Session Security For NTLMSSP Based Servers",
      "description": "Network security: Minimum session security for NTLM SSP based (including secure RPC) servers  This security setting allows a server to require the negotiation of 128-bit encryption and/or NTLMv2 session security. These values are dependent on the LAN Manager Authentication Level security setting value. The options are:  Require NTLMv2 session security: The connection will fail if message integrity is not negotiated. Require 128-bit encryption. The connection will fail if strong encryption (128-bit) is not negotiated.  Default:  Windows XP, Windows Vista, Windows 2000 Server, Windows Server 2003, and Windows Server 2008: No requirements.  Windows 7 and Windows Server 2008 R2: Require 128-bit encryption",
      "platform": "windows10",
      "type": "choice",
      "configured_value": "device_vendor_msft_policy_config_localpoliciessecurityoptions_networksecurity_minimumsessionsecurityforntlmsspbasedservers_537395200",
      "configured_value_label": "Require NTLM and 128-bit encryption",
      "parent_setting_id": "null",
      "parent_chain": [],
      "policy_name": "CIS - Local Policies Security Options [L1] - Windows 11 - v4.0.0.0",
      "policy_id": "6af60625-fa62-4a37-b748-a21f24ae6d06",
      "available_options": [
        "None",
        "Require NTLMv2 session security",
        "Require 128-bit encryption",
        "Require NTLM and 128-bit encryption"
      ]
    },
    {
      "id": "device_vendor_msft_policy_config_localpoliciessecurityoptions_microsoftnetworkclient_sendunencryptedpasswordtothirdpartysmbservers",
      "name": "Microsoft Network Client Send Unencrypted Password To Third Party SMB Servers",
      "description": "Microsoft network client: Send unencrypted password to connect to third-party SMB servers  If this security setting is enabled, the Server Message Block (SMB) redirector is allowed to send plaintext passwords to non-Microsoft SMB servers that do not support password encryption during authentication.  Sending unencrypted passwords is a security risk.  Default: Disabled.",
      "platform": "windows10",
      "type": "choice",
      "configured_value": "device_vendor_msft_policy_config_localpoliciessecurityoptions_microsoftnetworkclient_sendunencryptedpasswordtothirdpartysmbservers_0",
      "configured_value_label": "Disable",
      "parent_setting_id": "null",
      "parent_chain": [],
      "policy_name": "CIS - Local Policies Security Options [L1] - Windows 11 - v4.0.0.0",
      "policy_id": "6af60625-fa62-4a37-b748-a21f24ae6d06",
      "available_options": [
        "Enable",
        "Disable"
      ]
    },
    {
      "id": "device_vendor_msft_policy_config_credentialsui_disablepasswordreveal",
      "name": "Do not display the password reveal button",
      "description": "This policy setting allows you to configure the display of the password reveal button in password entry user experiences.\r\n\r\nIf you enable this policy setting, the password reveal button will not be displayed after a user types a password in the password entry text box.\r\n\r\nIf you disable or do not configure this policy setting, the password reveal button will be displayed after a user types a password in the password entry text box.\r\n\r\nBy default, the password reveal button is displayed after a user types a password in the password entry text box. To display the password, click the password reveal button.\r\n\r\nThe policy applies to all Windows components and applications that use the Windows system controls, including Internet Explorer.",
      "platform": "windows10",
      "type": "choice",
      "configured_value": "device_vendor_msft_policy_config_credentialsui_disablepasswordreveal_1",
      "configured_value_label": "Enabled",
      "parent_setting_id": "null",
      "parent_chain": [],
      "policy_name": "CIS - Windows Components - Restrictions [L1] - Windows 11 - v4.0.0.0",
      "policy_id": "c4dc50cf-85bf-49ad-a4d7-c594a7096e98",
      "available_options": [
        "Disabled",
        "Enabled"
      ]
    },
    {
      "id": "device_vendor_msft_policy_config_devicelock_devicepasswordexpiration",
      "name": "Device Password Expiration",
      "description": "Specifies when the password expires (in days). 0 - Passwords do not expire.",
      "platform": "windows10",
      "type": "simple",
      "configured_value": 500,
      "configured_value_label": "500",
      "parent_setting_id": "device_vendor_msft_policy_config_devicelock_devicepasswordenabled",
      "parent_chain": [
        "device_vendor_msft_policy_config_devicelock_devicepasswordenabled"
      ],
      "policy_name": "CIS - Device Lock [L1] - Windows 11 - v4.0.0.0",
      "policy_id": "76b42865-c293-47ff-b0be-6cce07766fea",
      "dependency_condition": {
        "parent_id": "device_vendor_msft_policy_config_devicelock_devicepasswordenabled",
        "parent_chosen_label": "Enabled",
        "active_unknown": "true"
      }
    },
    {
      "id": "device_vendor_msft_policy_config_devicelock_devicepasswordhistory",
      "name": "Device Password History",
      "description": "Specifies how many passwords can be stored in the history that can\u2019t be used.",
      "platform": "windows10",
      "type": "simple",
      "configured_value": 12,
      "configured_value_label": "12",
      "parent_setting_id": "device_vendor_msft_policy_config_devicelock_devicepasswordenabled",
      "parent_chain": [
        "device_vendor_msft_policy_config_devicelock_devicepasswordenabled"
      ],
      "policy_name": "CIS - Device Lock [L1] - Windows 11 - v4.0.0.0",
      "policy_id": "76b42865-c293-47ff-b0be-6cce07766fea",
      "dependency_condition": {
        "parent_id": "device_vendor_msft_policy_config_devicelock_devicepasswordenabled",
        "parent_chosen_label": "Enabled",
        "active_unknown": "true"
      }
    },
    {
      "id": "device_vendor_msft_policy_config_devicelock_maxdevicepasswordfailedattempts",
      "name": "Max Device Password Failed Attempts",
      "description": "On a desktop, when the user reaches the value set by this policy, it is not wiped. Instead, the desktop is put on BitLocker recovery mode, which makes the data inaccessible but recoverable. If BitLocker is not enabled, then the policy cannot be enforced.",
      "platform": "windows10",
      "type": "simple",
      "configured_value": 20,
      "configured_value_label": "20",
      "parent_setting_id": "device_vendor_msft_policy_config_devicelock_devicepasswordenabled",
      "parent_chain": [
        "device_vendor_msft_policy_config_devicelock_devicepasswordenabled"
      ],
      "policy_name": "CIS - Device Lock [L1] - Windows 11 - v4.0.0.0",
      "policy_id": "76b42865-c293-47ff-b0be-6cce07766fea",
      "dependency_condition": {
        "parent_id": "device_vendor_msft_policy_config_devicelock_devicepasswordenabled",
        "parent_chosen_label": "Enabled",
        "active_unknown": "true"
      }
    },
    {
      "id": "device_vendor_msft_policy_config_devicelock_maxinactivitytimedevicelock",
      "name": "Max Inactivity Time Device Lock",
      "description": "Specifies the maximum amount of time (in minutes) allowed after the device is idle that will cause the device to become PIN or password locked. Users can select any existing timeout value less than the specified maximum time in the Settings app. 0 - No timeout is defined",
      "platform": "windows10",
      "type": "simple",
      "configured_value": 15,
      "configured_value_label": "15",
      "parent_setting_id": "device_vendor_msft_policy_config_devicelock_devicepasswordenabled",
      "parent_chain": [
        "device_vendor_msft_policy_config_devicelock_devicepasswordenabled"
      ],
      "policy_name": "CIS - Device Lock [L1] - Windows 11 - v4.0.0.0",
      "policy_id": "76b42865-c293-47ff-b0be-6cce07766fea",
      "dependency_condition": {
        "parent_id": "device_vendor_msft_policy_config_devicelock_devicepasswordenabled",
        "parent_chosen_label": "Enabled",
        "active_unknown": "true"
      }
    },
    {
      "id": "device_vendor_msft_policy_config_devicelock_mindevicepasswordlength",
      "name": "Min Device Password Length",
      "description": "Specifies the minimum number or characters required in the PIN or password.",
      "platform": "windows10",
      "type": "simple",
      "configured_value": 3,
      "configured_value_label": "3",
      "parent_setting_id": "device_vendor_msft_policy_config_devicelock_devicepasswordenabled",
      "parent_chain": [
        "device_vendor_msft_policy_config_devicelock_devicepasswordenabled"
      ],
      "policy_name": "CIS - Device Lock [L1] - Windows 11 - v4.0.0.0",
      "policy_id": "76b42865-c293-47ff-b0be-6cce07766fea",
      "dependency_condition": {
        "parent_id": "device_vendor_msft_policy_config_devicelock_devicepasswordenabled",
        "parent_chosen_label": "Enabled",
        "active_unknown": "true"
      }
    },
    {
      "id": "device_vendor_msft_policy_config_devicelock_mindevicepasswordcomplexcharacters",
      "name": "Min Device Password Complex Characters",
      "description": "The number of complex element types (uppercase and lowercase letters, numbers, and punctuation) required for a strong PIN or password.",
      "platform": "windows10",
      "type": "choice",
      "configured_value": "device_vendor_msft_policy_config_devicelock_mindevicepasswordcomplexcharacters_2",
      "configured_value_label": "Digits and lowercase letters are required",
      "parent_setting_id": "device_vendor_msft_policy_config_devicelock_alphanumericdevicepasswordrequired",
      "parent_chain": [
        "device_vendor_msft_policy_config_devicelock_devicepasswordenabled",
        "device_vendor_msft_policy_config_devicelock_alphanumericdevicepasswordrequired"
      ],
      "policy_name": "CIS - Device Lock [L1] - Windows 11 - v4.0.0.0",
      "policy_id": "76b42865-c293-47ff-b0be-6cce07766fea",
      "available_options": [
        "Digits only",
        "Digits and lowercase letters are required",
        "Digits lowercase letters and uppercase letters are required. Not supported in desktop Microsoft accounts and domain accounts",
        "Digits lowercase letters uppercase letters and special characters are required. Not supported in desktop"
      ],
      "dependency_condition": {
        "parent_id": "device_vendor_msft_policy_config_devicelock_alphanumericdevicepasswordrequired",
        "parent_chosen_label": "Password or Alphanumeric PIN required.",
        "active_unknown": "true"
      }
    },
    {
      "id": "device_vendor_msft_bitlocker_systemdrivesrequirestartupauthentication",
      "name": "Require additional authentication at startup",
      "description": "This policy setting allows you to configure whether BitLocker requires additional authentication each time the computer starts and whether you are using BitLocker with or without a Trusted Platform Module (TPM). This policy setting is applied when you turn on BitLocker.\r\n\r\nNote: Only one of the additional authentication options can be required at startup, otherwise a policy error occurs.\r\n\r\nIf you want to use BitLocker on a computer without a TPM, select the \"Allow BitLocker without a compatible TPM\" check box. In this mode either a password or a USB drive is required for start-up. When using a startup key, the key information used to encrypt the drive is stored on the USB drive, creating a USB key. When the USB key is inserted the access to the drive is authenticated and the drive is accessible. If the USB key is lost or unavailable or if you have forgotten the password then you will need to use one of the BitLocker recovery options to access the drive.\r\n\r\nOn a computer with a compatible TPM, four types of authentication methods can be used at startup to provide added protection for encrypted data. When the computer starts, it can use only the TPM for authentication, or it can also require insertion of a USB flash drive containing a startup key, the entry of a 6-digit to 20-digit personal identification number (PIN), or both.\r\n\r\nIf you enable this policy setting, users can configure advanced startup options in the BitLocker setup wizard.\r\n\r\nIf you disable or do not configure this policy setting, users can configure only basic options on computers with a TPM.\r\n\r\nNote: If you want to require the use of a startup PIN and a USB flash drive, you must configure BitLocker settings using the command-line tool manage-bde instead of the BitLocker Drive Encryption setup wizard.\r\n\r\n",
      "platform": "windows10",
      "type": "choice",
      "configured_value": "device_vendor_msft_bitlocker_systemdrivesrequirestartupauthentication_1",
      "configured_value_label": "Enabled",
      "parent_setting_id": "null",
      "parent_chain": [],
      "policy_name": "CIS - BitLocker - Operating System Drives [BL] - Windows 11 - v4.0.0.0",
      "policy_id": "baf9b2f7-84ba-4326-8700-47b49029515f",
      "available_options": [
        "Disabled",
        "Enabled"
      ]
    },
    {
      "id": "device_vendor_msft_policy_config_windowslogon_allowautomaticrestartsignon",
      "name": "Sign-in and lock last interactive user automatically after a restart",
      "description": "This policy setting controls whether a device will automatically sign in and lock the last interactive user after the system restarts or after a shutdown and cold boot.\r\n\r\nThis only occurs if the last interactive user didn\u2019t sign out before the restart or shutdown.\u200b\r\n\r\nIf the device is joined to Active Directory or Azure Active Directory, this policy only applies to Windows Update restarts. Otherwise, this will apply to both Windows Update restarts and user-initiated restarts and shutdowns.\u200b\r\n\r\nIf you don\u2019t configure this policy setting, it is enabled by default. When the policy is enabled, the user is automatically signed in and the session is automatically locked with all lock screen apps configured for that user after the device boots.\u200b\r\n\r\nAfter enabling this policy, you can configure its settings through the ConfigAutomaticRestartSignOn policy, which configures the mode of automatically signing in and locking the last interactive user after a restart or cold boot\u200b.\r\n\r\nIf you disable this policy setting, the device does not configure automatic sign in. The user\u2019s lock screen apps are not restarted after the system restarts.",
      "platform": "windows10",
      "type": "choice",
      "configured_value": "device_vendor_msft_policy_config_windowslogon_allowautomaticrestartsignon_0",
      "configured_value_label": "Disabled",
      "parent_setting_id": "null",
      "parent_chain": [],
      "policy_name": "CIS - Windows Components - Restrictions [L1] - Windows 11 - v4.0.0.0",
      "policy_id": "c4dc50cf-85bf-49ad-a4d7-c594a7096e98",
      "available_options": [
        "Disabled",
        "Enabled"
      ]
    }
  ]
    result = compare_relevant_settings_to_cis_benchmark.invoke({"settings_to_check": relevant_settings})
    print(result)