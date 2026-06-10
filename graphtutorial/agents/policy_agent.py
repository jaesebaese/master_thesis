from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
import json
from dotenv import load_dotenv
from preprocessing_at_startup import build_tenant_collection


load_dotenv()

OLLAMA_MODEL = "minimax-m2.5:cloud"

OPENAI_API_MODEL = "gpt-5.4-nano-2026-03-17"

# Initialize the model -> if not used it will be inherited by the supervisor_agent
#model = init_chat_model(model=OLLAMA_MODEL, model_provider="ollama", temperature=0.0)
model = init_chat_model(model=OPENAI_API_MODEL, model_provider="openai", temperature=0.0)
""" model = init_chat_model(
    model=OLLAMA_MODEL,
    model_provider="ollama",
    base_url="https://ollama.com",
    client_kwargs={"headers": {"Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"}},
    temperature=0.0,
)
 """


@tool
def policy_requirement_extractor(runtime: ToolRuntime) -> str:
    """
    Extract structured security requirements from free-text policy text.

    This tool only extracts requirements. It does not map them to Intune
    settings, check compliance, or generate remediation advice.
    """
    files = runtime.state.get("files", {})
    
    file_entry = files.get("/security_policy.txt") or files.get("security_policy.txt")

    if file_entry is None:
         return json.dumps({"error": "security_policy.txt not found."})
    if isinstance(file_entry, dict):
        raw = file_entry.get("content", [])
        policy_file = "\n\n".join(raw) if isinstance(raw, list) else str(raw)
    else:
        policy_file = str(file_entry)

    system_prompt = """
You are a security policy requirement extraction engine for Microsoft Intune
MDM/device-configuration deployment.

Your task: extract the requirements from free-text policy that can be enforced
through a device configuration, compliance, or account-protection policy on a
managed endpoint.

CRITICAL FILTER — apply before extracting anything:
A requirement qualifies ONLY IF it maps to an enforceable device/OS/account
setting that Intune can push to or evaluate on a managed device. If the
requirement depends on human behavior, manual process, or organizational
governance, you MUST exclude it — even if it is mandatory in the policy.
If a requirement is partially enforceable, include only the enforceable
component and limit source_text to that clause.

DO NOT:
- map requirements to Intune settings
- check compliance
- recommend remediation
- add security best practices that are not in the text
- invent values or infer technical configuration settings
- summarize the policy generally
- include any output outside the JSON object

Return a single JSON object. The output must be parseable by json.loads().

Use this exact schema:

{
  "requirements": [
    {
      "requirement_id": "REQ-001",
      "source_text": "verbatim or near-verbatim clause from the input policy that establishes this requirement — do not paraphrase or interpret",
      "security_domain": "security_policy | encryption | firewall | antivirus | authentication | update_management | device_compliance | data_protection | access_control | other",
      "control_intent": "short_snake_case_description_of_the_control_goal",
      "expected_value": "the required value as a string: a number (\"12\"), \"enabled\", \"disabled\", or null if not specified",
      "operator": "minimum | maximum | exactly | null",
      "expected_unit": "characters | days | attempts | versions | enabled_disabled | other | null",
      "applicability": "who or what the requirement applies to, or null",
      "strength": "mandatory | recommended | prohibited | informational",
      "confidence": 0.0
    }
  ]
}

Rules:
- requirement_id must start at REQ-001 and increment sequentially.
- source_text must be verbatim or near-verbatim from the input — do not paraphrase or interpret.
- security_domain must use one of the listed categories
- control_intent must be short, specific, and in snake_case (e.g., minimum_password_length, require_bitlocker_encryption).
- expected_value must be null if no concrete value is stated. Do not infer or estimate.
    - Use "enabled" or "disabled" for toggle settings
    - Use a number as a string for numeric thresholds (e.g., "12", "90").
- When expected_value is "enabled" or "disabled", expected_unit must be "enabled_disabled".
- operator must be null if the requirement is not a measurable threshold.
- confidence reflects how clearly the source text maps to an enforceable Intune setting:
    - 1.0 = explicit, specific, and unambiguous
    - 0.7 = implied but reasonably certain
    - 0.4 = requires interpretation; edge case
    - below 0.4 = likely excluded by the CRITICAL FILTER

Examples:

Numeric threshold —
{
  "requirement_id": "REQ-001",
  "source_text": "Passwords must be at least 12 characters in length.",
  "security_domain": "authentication",
  "control_intent": "minimum_password_length",
  "expected_value": "12",
  "operator": "minimum",
  "expected_unit": "characters",
  "applicability": "all managed devices",
  "strength": "mandatory",
  "confidence": 0.95
}

Toggle setting —
{
  "requirement_id": "REQ-002",
  "source_text": "BitLocker encryption must be enabled on all managed devices.",
  "security_domain": "encryption",
  "control_intent": "require_bitlocker_encryption",
  "expected_value": "enabled",
  "operator": null,
  "expected_unit": "enabled_disabled",
  "applicability": "all managed devices",
  "strength": "mandatory",
  "confidence": 0.98
}
"""
    user_prompt = f"""
    Extract structured security requirements from this policy text:
    {policy_file}
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
    raw_output = (content or "").strip()

    # Local models sometimes wrap JSON in ```json fences
    if raw_output.startswith("```"):
        raw_output = raw_output.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        return json.dumps({
            "requirements": [],
            "error": "Model returned invalid JSON",
            "raw_output": raw_output,
        }, indent=2)

    return json.dumps(parsed, indent=2)


@tool
def check_security_policy(runtime: ToolRuntime) -> str:
    """
    Check if the given security policy fulfils the format requirements.

    This tool receives the security policy text.

    The tool returns a JSON object containing only the settings from the input array that are relevant to the policy.
    No new settings should be invented, and unrelated settings should be excluded. All fields from the input must be preserved verbatim.
    """
    files = runtime.state.get("files", {})
        
    file_entry = files.get("/security_policy.txt") or files.get("security_policy.txt")

    if file_entry is None:
        return json.dumps({"error": "policy_requirements.json not found. Ensure policy_agent has run first."})
    if isinstance(file_entry, dict):
        raw = file_entry.get("content", [])
        policy_file = "\n\n".join(raw) if isinstance(raw, list) else str(raw)
    else:
        policy_file = str(file_entry)

policy_agent = {
    "name": "policy_agent",
    "description": (
        "Extracts structured security requirements from the security policy document. "
        "Calls policy_requirement_extractor to parse the policy and returns a JSON array "
        "of requirements saved to policy_requirements.json. Use this as the first step "
        "before any tenant or benchmark analysis."
    ),
    "system_prompt": ("""\
You are a security policy analyst. Your job is to extract structured requirements \
from a security policy document provided by the user.

Only extract requirements from the provided document — do not infer, assume, \
or supplement from memory or training data.

## Step 1 — Extract requirements
Call policy_requirement_extractor (no arguments). It will read the security policy \
and return a JSON array of requirements.

## Step 2 — Save to file
Write the tool's return value to the virtual filesystem:
  path: policy_requirements.json
  content: the exact string returned by the tool — do not reformat, \
pretty-print, add markdown fences, or modify it in any way.

## Step 3 — Output a summary
After saving, produce a summary table with one row per requirement:

| Requirement ID | Source Text | Security Domain |

Keep the source text brief (one sentence max). Do not add prose outside the table \
unless requirements could not be extracted.
"""
    ),
    "tools": [policy_requirement_extractor],
    "model": model,
}


if __name__ == "__main__":
    build_tenant_collection(force_rebuild=False)
    policy = "All USB storage devices must be blocked on organizational endpoints to prevent unauthorized data transfer and mitigate the risk of malware infections. This policy applies to all employees, contractors, and third-party users who access organizational systems and data. The use of USB storage devices is prohibited unless explicitly authorized by the IT department for specific business needs. Exceptions may be granted on a case-by-case basis, but only after a thorough risk assessment and implementation of appropriate security controls. Users must not attempt to bypass this policy by using alternative methods of data transfer, such as personal email accounts or cloud storage services, without prior approval. Violations of this policy may result in disciplinary action, up to and including termination of employment or contract. The organization will implement technical controls to enforce this policy, such as endpoint security solutions that block USB storage device access and monitor for any attempts to connect unauthorized devices."
    
    print("\n--- Extracted policy requirements ---\n")
    #requirements_json = policy_requirement_extractor.invoke({"policy_input": policy})
    #print(requirements_json)

    result = policy_requirement_extractor.invoke({
    "security_policy": policy, "platform": "windows10"})
    print(result)
