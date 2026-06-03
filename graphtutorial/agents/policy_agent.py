import os
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
import json
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
from preprocessing_at_startup import flatten_for_relevance, build_tenant_collection


load_dotenv()

OLLAMA_MODEL = "minimax-m2.5:cloud"

OPENAI_API_MODEL = "gpt-5.4-nano-2026-03-17"

SETTINGS_JSON = os.path.join(os.path.dirname(__file__), "../intune_configurations/intune_configuration_settings.json")
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "../intune_configurations/chroma_db")
TENANT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../configurations/policies_and_settings_expand.json")

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

def build_intune_vector_db(force_rebuild: bool = False):
    """Embed all settings from intune_configuration_settings.json into a
    persistent ChromaDB collection.

    Each document contains the setting name, description, and keywords so the
    collection supports semantic search.  Metadata stores the id, name,
    description (truncated), and platform for display without re-fetching.

    The DB is written to intune_configurations/chroma_db/ next to the JSON
    file so it only needs to be built once.  Pass force_rebuild=True to
    drop and recreate the collection.

    Returns the ChromaDB Collection object.
    """


    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    ef = DefaultEmbeddingFunction()

    existing = [c.name for c in client.list_collections()]

    if "intune_settings" in existing:
        if not force_rebuild:
            print("Collection already exists. Loading from disk.")
            return client.get_collection("intune_settings", embedding_function=ef)
        print("force_rebuild=True — dropping existing collection.")
        client.delete_collection("intune_settings")

    collection = client.create_collection(
        name="intune_settings",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"Loading settings from {SETTINGS_JSON} ...")
    with open(SETTINGS_JSON) as f:
        settings = json.load(f)

    ids, documents, metadatas = [], [], []

    for s in settings:
        setting_id = s.get("id", "")
        if not setting_id:
            continue

        name = s.get("displayName") or s.get("name") or ""
        description = (s.get("description") or "")[:500]
        keywords = ", ".join(s.get("keywords") or [])
        platform = s.get("applicability", {}).get("platform", "")

        # Text that gets embedded — rich enough for semantic search
        document = f"{name}. {description} Keywords: {keywords}".strip()

        ids.append(setting_id)
        documents.append(document)
        metadatas.append({
            "id": setting_id,
            "name": name,
            "description": description[:200],  # ChromaDB metadata has size limits
            "platform": platform,
        })

    BATCH_SIZE = 500
    total = len(ids)
    for i in range(0, total, BATCH_SIZE):
        collection.add(
            ids=ids[i : i + BATCH_SIZE],
            documents=documents[i : i + BATCH_SIZE],
            metadatas=metadatas[i : i + BATCH_SIZE],
        )
        print(f"  Embedded {min(i + BATCH_SIZE, total)}/{total} settings...")

    print(f"Done. {total} settings stored in {CHROMA_DB_PATH}")
    return collection


def _get_intune_collection():
    """Load the persisted ChromaDB collection. Raises if the DB hasn't been built yet."""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    ef = DefaultEmbeddingFunction()
    return client.get_collection("intune_settings", embedding_function=ef)


@tool
def policy_analyzer(runtime: ToolRuntime, security_policy: str, platform: str ) -> str:
    """Search the Intune settings vector database for configuration settings
    that are semantically similar to the query.  Returns the top matches with
    their id, name, description, and platform.

    Args:
        query: A natural-language description of security requirements
                e.g. 'BitLocker recovery password options' or
               'block executable content from email'.
    """

    files = runtime.state.get("files", {})
    
    file_entry = files.get("/security_policy.txt") or files.get("security_policy.txt")

    if file_entry is not None:
        if isinstance(file_entry, dict):
            raw = file_entry.get("content", [])
            policy_file = "\n\n".join(raw) if isinstance(raw, list) else str(raw)
        else:
            policy_file = str(file_entry)
    elif os.path.join(os.path.dirname(__file__), "security_policy.txt"):
        print("FILE NOT FOUND")
        #TODO: add the file in the local path
        disk_path = os.path.join(os.path.dirname(__file__), "security_policy.txt")
        try:
            with open(disk_path) as f:
                policy_file = f.read()
        except FileNotFoundError:
            return json.dumps({
                "settings": [],
                "error": "security_policy.txt not found in virtual filesystem or on disk.",
            })
    else:
        policy_file = security_policy
            
    _PLATFORM_MAP = {
        "windows": "windows10",
        "windows10": "windows10",
        "windows 10": "windows10",
        "windows 11": "windows10",
        "windows11": "windows10",
        "macos": "macOS",
        "mac": "macOS",
        "ios": "iOS",
        "android": "android",
    }
    normalized_platform = _PLATFORM_MAP.get(platform.lower().strip(), platform) if platform else None

    collection = _get_intune_collection()
    where = {"platform": normalized_platform} if normalized_platform else None

    paragraphs = [p.strip() for p in policy_file.split("\n\n") if len(p.strip()) > 20]
    if not paragraphs:
        paragraphs = [policy_file.strip()]

    best: dict[str, dict] = {}

    for para in paragraphs:
        results = collection.query(
            query_texts=[para],
            n_results=10,
            where=where,
            include=["metadatas", "distances", "documents"],
        )
        for meta, distance in zip(results["metadatas"][0], results["distances"][0]):
            score = round(1 - distance, 4)
            sid = meta.get("id")
            if score > 0.5 and (sid not in best or score > best[sid]["similarity_score"]):
                best[sid] = {
                    "id": sid,
                    "name": meta.get("name"),
                    "description": meta.get("description"),
                    "platform": meta.get("platform"),
                    "similarity_score": score,
                    "matched_paragraph": para,
                }

    hits = sorted(best.values(), key=lambda h: h["similarity_score"], reverse=True)
    return json.dumps(hits, indent=2)

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
         return json.dumps({"error": "policy_requirements.json not found. Ensure policy_agent has run first."})
    if isinstance(file_entry, dict):
        raw = file_entry.get("content", [])
        policy_file = "\n\n".join(raw) if isinstance(raw, list) else str(raw)
    else:
        policy_file = str(file_entry)

    system_prompt = """
        You are a security policy requirement extraction engine for Microsoft Intune
        MDM/device-configuration deployment.

        Your task: extract the requirements from free-text policy that can be
        enforced through a  device configuration, compliance, or account-protection 
        policy on a managed endpoint.

        CRITICAL FILTER — apply before extracting anything:
        A requirement qualifies ONLY IF it maps to an enforceable device/OS/account
        setting that Intune can push to or evaluate on a managed device. If the
        requirement depends on human behavior, manual process, or organizational governance, 
        you MUST exclude it — even if it is mandatory in the policy.

        DO NOT:
        - map requirements to Intune settings
        - check compliance
        - recommend remediation
        - add security best practices that are not in the text
        - invent values
        - summarize the policy generally

        Return only valid JSON.

        Use this exact schema:

        {
        "requirements": [
            {
            "requirement_id": "REQ-001",
            "source_text": "A sentence that describes what the requirement is and for what it is",
            "security_domain": "security_policy | encryption | firewall | antivirus | authentication | update_management | device_compliance | data_protection | access_control | other",
            "control_intent": "short_snake_case_description_of_the_control_goal",
            "expected_value": "specific required value, boolean, number, or null if not specified",
            "operator": "minimum | maximum | exactly | null (if not a measurable quantity)",
            "expected_unit": "characters | days | attempts | versions | enabled_disabled | other | null",
            "applicability": "who or what the requirement applies to, or null",
            "strength": "mandatory | recommended | prohibited | informational",
            "confidence": 0.0
            }
        ]
        }

        Rules:
        - requirement_id must start at REQ-001 and increment sequentially.
        - source_text must summarize the relevant sentence or clause from the input.
        - security_domain must use one of the listed categories.
        - control_intent must be short, specific, and written in snake_case.
        - expected_value must be null if no concrete value is stated.
        - Do not infer technical configuration settings.
        - Do not include explanations outside the JSON.
        - The output must be parseable by json.loads().
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


RELEVANCE_SYSTEM_PROMPT = """\
You are an Intune settings discovery specialist. Your role is to identify which
of the tenant's CURRENTLY CONFIGURED Intune settings are relevant to a given
security policy or topic.

You will receive:
1. A security policy (free text).
2. A JSON array of settings currently configured in the tenant. Each entry has
   at minimum an id, name, description, platform, and configured value.

Your task: return ONLY the settings from the input array that are relevant to
the policy. Do not invent settings. Do not include settings unrelated to the
policy. Preserve every field from the input verbatim — id, name, description,
platform, configured value — exactly as given.

Output format: a JSON object of the form
{
  "settings": [
    { ...verbatim entry from input... },
    ...
  ]
}

Rules:
- Use only IDs that appear in the input.
- If no settings are relevant, return {"settings": []}.
- Do not add explanations outside the JSON.
- The output must be parseable by json.loads().
"""

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
        "Searches the tenant's configuration settings using semantic similarity. "
        "Use this to find which configuration settings relate to a given topic "
        "such as 'BitLocker', 'firewall inbound rules', or 'password complexity'."
    ),
    "system_prompt": ("""\
You are a security policy specialist.
Call the policy_requirement_extractor to extract the requirements out of the
provided security policy.
## Saving output\n
    After calling find_configs_in_policies(), call write_file with:
        path: 'policy_requirements.json'
        content: the raw JSON string returned by policy_requirement_extractor\n"
    Do this before summarising the results.
    
"""
    ),
    "tools": [policy_requirement_extractor],
    "model": model,
}


if __name__ == "__main__":
    build_intune_vector_db(force_rebuild=False)
    policy =""" 
All passwords must be created in a way that ensures they are difficult to guess or compromise. Users must ensure that passwords are unique to each system and are not reused across different services. Passwords must remain confidential at all times and must not be shared with any other individual, including IT personnel. Furthermore, passwords must not be stored in plain text, whether digitally or physically, unless they are protected by approved secure storage mechanisms such as password managers.

4.2 Complexity Requirements
Passwords must meet a minimum length of twelve characters and include a combination of uppercase letters, lowercase letters, numbers, and special characters. Users must avoid using easily guessable information such as names, usernames, dates of birth, or common words found in dictionaries. The intent is to ensure that passwords are resistant to brute-force and dictionary-based attacks.

4.3 Passphrases
Where systems allow, users are encouraged to create passphrases instead of traditional passwords. A passphrase should consist of at least sixteen characters and be composed of multiple unrelated words combined with numbers or special characters. This approach increases memorability while maintaining a high level of security.

4.4 Password Reuse
To reduce the risk of compromise, users must not reuse previous passwords. The organization will enforce controls to prevent the reuse of at least the last ten passwords. In addition, users should ensure that passwords used within the organization are not reused for personal accounts or external services.

4.5 Password Expiry
Passwords must be changed periodically to reduce the risk of long-term exposure. Standard user accounts must update their passwords at least every ninety days, while privileged accounts must be updated every sixty days. In all cases, passwords must be changed immediately if there is any suspicion that they have been compromised.

5. Multi-Factor Authentication
To enhance security beyond passwords alone, multi-factor authentication is required for access to sensitive systems, remote access services, and all privileged accounts. This additional layer of verification may include authenticator applications, hardware tokens, or biometric factors where appropriate. The use of multi-factor authentication significantly reduces the risk of unauthorized access even if a password is compromised.

6. Account Protection
6.1 Failed Login Attempts
To protect against unauthorized access attempts, accounts will be automatically locked after a defined number of unsuccessful login attempts. Specifically, after five failed attempts, the account will be locked for a minimum of fifteen minutes or until it is reset by authorized personnel. This measure helps mitigate brute-force attacks.

6.2 Session Management
User sessions must be managed to reduce the risk of unauthorized access due to unattended devices. Systems will automatically terminate sessions after fifteen minutes of inactivity, requiring users to re-authenticate to regain access. This ensures that access is limited to active and authorized users only.

7. Storage and Transmission
Passwords must be handled securely both at rest and in transit. Under no circumstances may passwords be stored in plain text. Instead, they must be protected using strong cryptographic hashing algorithms such as bcrypt or Argon2. Additionally, passwords must only be transmitted over secure channels that provide encryption, ensuring that they cannot be intercepted or read by unauthorized parties.

8. User Responsibilities
All users are responsible for maintaining the confidentiality and security of their passwords. This includes selecting strong passwords, not sharing them with others, and promptly reporting any suspected compromise. Users are also expected to use only organization-approved tools, such as password managers, for storing and managing their credentials. Care must be taken to ensure that passwords used for corporate systems are not reused in personal contexts.

9. Privileged Accounts
Accounts with elevated privileges require additional safeguards due to their increased access to sensitive systems and data. These accounts must use stronger passwords, with a minimum length of sixteen characters, and must always be protected by multi-factor authentication. The use of privileged accounts must be strictly controlled, monitored, and logged. Shared use of privileged accounts is not permitted, as it prevents accountability and traceability.


10. Enforcement

10.1 Technical Enforcement
The organization will implement technical controls to enforce this policy consistently across all systems. These controls include enforcing password complexity requirements, preventing password reuse, requiring periodic password changes, and enabling account lockout mechanisms. Multi-factor authentication will be enforced where required, and authentication events will be logged and monitored to detect suspicious activity.

10.2 Administrative Enforcement
In addition to technical measures, the organization will conduct regular audits and access reviews to ensure compliance with this policy. Access rights will be reviewed periodically and adjusted as necessary based on role changes or termination of employment. Non-compliance will be addressed through appropriate administrative actions.10.3 Non-Compliance
Failure to comply with this policy may result in disciplinary measures, which can include restriction of access rights, formal warnings, or termination of employment or contractual agreements, depending on the severity of the violation.   
"""
    #policy_2 = All USB storage devices must be blocked on organizational endpoints to prevent unauthorized data transfer and mitigate the risk of malware infections. This policy applies to all employees, contractors, and third-party users who access organizational systems and data. The use of USB storage devices is prohibited unless explicitly authorized by the IT department for specific business needs. Exceptions may be granted on a case-by-case basis, but only after a thorough risk assessment and implementation of appropriate security controls. Users must not attempt to bypass this policy by using alternative methods of data transfer, such as personal email accounts or cloud storage services, without prior approval. Violations of this policy may result in disciplinary action, up to and including termination of employment or contract. The organization will implement technical controls to enforce this policy, such as endpoint security solutions that block USB storage device access and monitor for any attempts to connect unauthorized devices.
    
    print("\n--- Extracted policy requirements ---\n")
    #requirements_json = policy_requirement_extractor.invoke({"policy_input": policy})
    #print(requirements_json)

    result = policy_requirement_extractor.invoke({
    "security_policy": policy, "platform": "windows10"})
    print(result)
