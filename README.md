# Microsoft Graph Security Posture Agent

This tool ingests a security policy document, compares it against your Microsoft Intune tenant configuration and the CIS Microsoft Intune for Windows 11 Benchmark, and reports where your tenant's configuration is compliant, non-compliant, or missing relative to both.

## Prerequisites

- **Python 3.11+** and pip.
- A Microsoft work/school account with access to a tenant that has Intune configuration policies, and permission to grant an app registration `DeviceManagementConfiguration.Read.All` (delegated). If you don't have one, you can [sign up for the Microsoft 365 Developer Program](https://developer.microsoft.com/microsoft-365/dev-program) for a free test tenant.
- An **OpenAI API key** — all agents (`supervisor_agent.py`, `policy_agent.py`, `config_agent.py`, `benchmark_agent.py`, `interdepedency_agent.py`, `search_agent.py`) currently call OpenAI models via LangChain's `init_chat_model("openai:...")`.
- A **Tavily API key** — used by `search_agent.py` for web search.
- The **CIS Microsoft Intune for Windows 11 Benchmark v4.0.0** spreadsheet (`.xlsx`). This is licensed content from the [Center for Internet Security](https://www.cisecurity.org/) and is **not included in this repo** — you must obtain it yourself and place it at:
    ```
    graphtutorial/preprocessing/CIS_Microsoft_Intune_for_Windows_11_Benchmark_v4.0.0.xlsx
    ```
- The CIS-published Level 1 Intune configuration policy exports (`.json`), placed under:
    ```
    graphtutorial/configurations/cis_benchmarks/Settings Catalog/Level 1/
    ```
    These are matched against the benchmark spreadsheet in the step below.

## Install dependencies

```Shell
python3 -m pip install -r graphtutorial/requirements.txt
python3 -m pip install typer pdfplumber python-docx pandas openpyxl rich chromadb \
    langchain langchain-core langchain-openai langgraph deepagents tavily-python python-dotenv
```

`graphtutorial/pyproject.toml` lists some of the agent dependencies but isn't wired up as an installable package for this repo, and there's no single lockfile covering everything the `agents/` code imports — the second command above fills in the gap.

## Configure credentials

- `graphtutorial/config.cfg` — Azure app registration used for the Graph device-code sign-in:
    ```ini
    [azure]
    clientId = <your app registration's client ID>
    tenantId = <your tenant ID>
    graphUserScopes = User.Read Mail.Read Mail.Send DeviceManagementConfiguration.Read.All
    ```
- `graphtutorial/.env`:
    ```
    OPENAI_API_KEY=<your OpenAI key>
    TAVILY_API_KEY=<your Tavily key>
    ```
    (An Ollama-based model path exists commented out in the agent files if you'd rather run a local/hosted Ollama model instead of OpenAI.)

Both files are gitignored — they won't be committed.

## Set up the data the agent needs (run once, from the repo root)

1. **Add the CIS benchmark files** described in Prerequisites above.
2. **Match the CIS benchmark against the Intune policy exports:**
    ```Shell
    python3 graphtutorial/preprocessing/match_cis_configurations.py
    ```
    Writes `graphtutorial/configurations/cis_benchmarks/matched_all_level1.json`.
3. **Fetch your tenant's current Intune configuration:**
    ```Shell
    python3 graphtutorial/fetch_tenant_settings.py
    ```
    This runs a device-code sign-in and calls Microsoft Graph for your tenant's configuration policies.

## Run the tool

```Shell
python3 graphtutorial/agents/main.py /path/to/security_policy.pdf
```

Accepts `.pdf`, `.docx`, `.xlsx`, `.csv`, or plain text. This will:

- Convert the policy document to text and save it to `graphtutorial/agents/security_policy.txt`.
- Build/refresh the CIS benchmark and tenant configuration vector databases (ChromaDB).
- Launch the supervisor agent, which compares your tenant's Intune configuration against both the uploaded policy and the CIS benchmark and reports the results.
