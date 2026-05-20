from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from typing import List
import json
import os
from policy_agent import PolicyAgentResults, flatten_for_relevance, _get_intune_collection, SETTINGS_JSON
from deepagents import create_deep_agent
import logging
import time
from contextvars import ContextVar
from langchain.agents.middleware import before_model, after_model, wrap_tool_call


OPENAI_MODEL = "gpt-5.4-nano-2026-03-17"
OLLAMA_MODEL = "mistral-nemo:latest"
#model = init_chat_model(model=OLLAMA_MODEL, model_provider="ollama", temperature=0.0)


# Initialize the model
model = init_chat_model(model=OPENAI_MODEL, model_provider="openai", temperature=0.0)

_start = ContextVar("model_start", default=None)

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s", 
    handlers=[
        logging.FileHandler("agent.log", mode='w'),  # overwrite log file on each run
    ],)

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


def log_chunk(chunk: dict) -> None:
    if not isinstance(chunk, dict):
        logger.info("Chunk: %s", str(chunk)[:1000])
        return

    for node, payload in chunk.items():
        if payload is None:
            logger.info("[%s] (no output)", node)
            continue

        # Payload might be an Overwrite/Append wrapper, a dict, or something else
        if not isinstance(payload, dict):
            logger.info("[%s] %s", node, str(payload)[:1000])
            continue

        # Drop noisy keys
        payload = {k: v for k, v in payload.items() if k != "files"}

        messages = payload.get("messages")
        if messages is None:
            # No messages field — just log keys touched
            logger.info("[%s] keys=%s", node, list(payload.keys()))
            continue

        # messages might also be an Overwrite wrapper, not a list
        if not isinstance(messages, list):
            logger.info("[%s] messages=%s", node, str(messages)[:300])
            continue

        for msg in messages:
            role = type(msg).__name__
            content = getattr(msg, "content", "")
            if not isinstance(content, str):
                content = str(content)
            preview = content[:300] + ("..." if len(content) > 300 else "")
            logger.info("[%s] %s: %s", node, role, preview)


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


@tool
def find_interdependencies_in_configurations(runtime: ToolRuntime) -> str:
    """Check which configured settings appear in policies_and_settings_expand.json.

    For each setting in {configured_settings ∪ benchmark_findings ∪ remediation_candidates}:
  
  1. catalog_search(query=setting.name + setting.description, n=10)
     → top 10 semantically related catalog entries
  
  2. tenant_lookup(retrieved_ids)
     → which of those are also configured in the tenant?
  
  3. classify_relationships(setting, retrieved + their tenant status)
     → for each candidate, classify:
        - parent_child (one gates the other)
        - conflict (configured to inconsistent values)
        - alternative (achieves same control via different mechanism)
        - prerequisite (one must be configured for the other to take effect)
        - paired_control (only meaningful together)
        - unrelated (drop)
  
  4. for relationships marked conflict or unmet-prerequisite:
     flag as a finding requiring user attention
  
  5. for relationships marked alternative or paired_control:
     surface as informational
    """
    files = runtime.state.get("files", {})
    file_entry = files.get("/relevant_configs.json") or files.get("relevant_configs.json")
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
    if isinstance(data, dict) and "found" in data:
        configs = [
            entry["config"]
            for entry in data["found"]
            if isinstance(entry, dict) and "config" in entry
        ]
    elif isinstance(data, dict) and "settings" in data:
        configs = data["settings"]
    elif isinstance(data, list):
        configs = data
    else:
        configs = []

    if not configs:
        return json.dumps({
            "error": "No settings found in relevant_configs.json. Check the file shape."
        })

    path = os.path.join(
        os.path.dirname(__file__), "../configurations/policies_and_settings_expand_assign.json"
    )
    with open(path) as f:
        policies = json.load(f)

    # --- 1. Build a flat index of every setting configured across the whole tenant ---
    with open(SETTINGS_JSON) as f:
        catalog = {s["id"]: s for s in json.load(f)}

    # Build policy_id -> set of assigned group IDs
    policy_groups: dict[str, set[str]] = {}
    for policy in policies:
        pid = policy.get("id", "")
        policy_groups[pid] = {
            a["target"]["groupId"]
            for a in policy.get("assignments", [])
            if a.get("target", {}).get("groupId")
        }

    all_tenant_flat = flatten_for_relevance(policies, catalog)
    tenant_index: dict[str, list[dict]] = {}
    for s in all_tenant_flat:
        tenant_index.setdefault(s["id"], []).append(s)

    # --- 2. Detect cross-policy conflicts (same setting ID, different values) ---
    structural_conflicts: list[dict] = []
    for sid, entries in tenant_index.items():
        if len(entries) < 2:
            continue
        unique_values = {str(e.get("configured_value")) for e in entries}
        if len(unique_values) <= 1:
            continue

        # For each group, collect the distinct values it receives
        group_values: dict[str, set[str]] = {}
        for e in entries:
            val = str(e.get("configured_value"))
            for gid in policy_groups.get(e.get("policy_id", ""), set()):
                group_values.setdefault(gid, set()).add(val)

        conflicting_groups = [gid for gid, vals in group_values.items() if len(vals) > 1]

        occurrences = [
            {
                "policy_name": e["policy_name"],
                "configured_value": e["configured_value"],
                "configured_value_label": e.get("configured_value_label", ""),
                "group_ids": sorted(policy_groups.get(e.get("policy_id", ""), set())),
            }
            for e in entries
        ]

        if conflicting_groups:
            structural_conflicts.append({
                "type": "conflict",
                "setting_id": sid,
                "setting_name": entries[0].get("name", sid),
                "conflicting_groups": sorted(conflicting_groups),
                "occurrences": occurrences,
                "severity": "finding",
                "reason": "Same setting configured with conflicting values targeting the same group(s).",
            })
        else:
            structural_conflicts.append({
                "type": "different_group_value",
                "setting_id": sid,
                "setting_name": entries[0].get("name", sid),
                "occurrences": occurrences,
                "severity": "info",
                "reason": "Different values configured for different groups: intentional differentiation, not a conflict.",
            })
    print("CONFLICTS: " + json.dumps(structural_conflicts, indent=2, default=list))
    # --- 3. Detect unmet prerequisites from parent_chain ---
    configured_ids = {s["id"] for s in configs if isinstance(s, dict)}
    unmet_prerequisites: list[dict] = []
    seen_unmet: set[tuple] = set()
    for setting in configs:
        if not isinstance(setting, dict):
            continue
        for parent_id in setting.get("parent_chain", []):
            key = (setting["id"], parent_id)
            if key in seen_unmet or parent_id in configured_ids:
                continue
            seen_unmet.add(key)
            unmet_prerequisites.append({
                "type": "unmet_prerequisite",
                "setting_id": setting["id"],
                "setting_name": setting.get("name", setting["id"]),
                "missing_parent_id": parent_id,
                "severity": "finding",
                "reason": f"Depends on '{parent_id}' which is absent from the policy-relevant configured settings.",
            })

    # --- 4. Semantic interdependency analysis via ChromaDB + LLM ---
    CLASSIFY_SYSTEM = """\
You are an Intune security policy interdependency analyst.

You will receive:
1. A "focal" setting currently configured in the tenant.
2. A list of "candidates" — semantically similar settings, each flagged with whether it is configured in the tenant.

For each candidate classify its relationship to the focal setting using exactly one label:
- parent_child   – one gates or depends on the other in the Intune settings tree
- conflict       – configured to contradictory or mutually exclusive values
- alternative    – achieves the same control via a different mechanism
- prerequisite   – one must be configured before the other takes effect
- paired_control – neither alone achieves the control; both are needed together
- unrelated      – no meaningful security dependency (omit from output)

Return a JSON array. Each element must have:
{
  "candidate_id": "...",
  "candidate_name": "...",
  "relationship": "<label>",
  "tenant_configured": true|false,
  "severity": "finding" or "informational",
  "explanation": "<one concise sentence>"
}

Use severity="finding" for conflict and prerequisite relationships.
Use severity="informational" for parent_child, alternative, and paired_control.
If no meaningful relationships exist, return [].
Output must be valid JSON parseable by json.loads().
"""

    semantic_relationships: list[dict] = []
    try:
        collection = _get_intune_collection()

        for setting in configs[:20]:  # cap to stay within LLM token budget
            if not isinstance(setting, dict):
                continue

            query = f"{setting.get('name', '')}. {setting.get('description', '')}"
            try:
                results = collection.query(
                    query_texts=[query],
                    n_results=11,  # +1 because self often appears in results
                    include=["metadatas", "distances"],
                )
            except Exception:
                continue

            candidates = []
            for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
                cid = meta.get("id", "")
                if cid == setting["id"]:
                    continue
                score = round(1 - dist, 4)
                if score < 0.65:
                    continue
                if cid in tenant_index:
                    # Attach the first configured value (or all if you want to handle multi-policy)
                    configured_value_label = tenant_index[cid][0].get("configured_value_label")
                    configured_value = tenant_index[cid][0].get("configured_value")
                candidates.append({
                    "id": cid,
                    "name": meta.get("name", cid),
                    "description": meta.get("description", ""),
                    "similarity_score": score,
                    "tenant_configured": cid in tenant_index,
                    "configured_value_label": configured_value_label,
                    "configured_value": configured_value
                })

            if not candidates:
                continue

            focal_summary = {
                "id": setting["id"],
                "name": setting.get("name"),
                "description": setting.get("description"),
                "configured_value_label": setting.get("configured_value_label"),
                "policy_name": setting.get("policy_name"),
            }
            user_prompt = (
                f"FOCAL SETTING:\n{json.dumps(focal_summary, indent=2)}\n\n"
                f"CANDIDATES:\n{json.dumps(candidates, indent=2)}\n\n"
                "Classify each candidate's relationship to the focal setting."
            )

            response = model.invoke([
                {"role": "system", "content": CLASSIFY_SYSTEM},
                {"role": "user", "content": user_prompt},
            ])
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.replace("```json", "").replace("```", "").strip()

            try:
                classified = json.loads(raw)
            except json.JSONDecodeError:
                continue

            for rel in classified:
                rel["focal_setting_id"] = setting["id"]
                rel["focal_setting_name"] = setting.get("name", setting["id"])
                rel["focal_policy_name"] = setting.get("policy_name", "")
                semantic_relationships.append(rel)

    except Exception as e:
        print(f"WARNING: ChromaDB unavailable, skipping semantic analysis: {e}")
        collection = None

    # --- 5. Partition into findings vs informational and return ---
    semantic_findings = [r for r in semantic_relationships if r.get("severity") == "finding"]
    informational = [r for r in semantic_relationships if r.get("severity") == "informational"]

    findings = structural_conflicts + unmet_prerequisites + semantic_findings

    result = {
        "summary": {
            "settings_analyzed": len(configs),
            "structural_conflicts": len(structural_conflicts),
            "unmet_prerequisites": len(unmet_prerequisites),
            "semantic_findings": len(semantic_findings),
            "informational_relationships": len(informational),
        },
        "findings": findings,
        "informational": informational,
    }
    return json.dumps(result, indent=2)

interdependency_agent = {
    "name": "interdependency_agent",
    "description": (
        "Retrieves and analyzes Intune configuration policies. "
        "Use the find_interdependencies_in_configurations tool to get detailed explanations of currently configured policies and settings in the tenant. "
    ),
    "system_prompt": ("""\
        You are an Intune security policy interdependency analyst.

        You will receive:
        1. A "focal" setting currently configured in the tenant.
        2. A list of "candidates" — semantically similar settings, each flagged with whether it is configured in the tenant.

        For each candidate classify its relationship to the focal setting using exactly one label:
        - parent_child   – one gates or depends on the other in the Intune settings tree
        - conflict       – configured to contradictory or mutually exclusive values
        - alternative    – achieves the same control via a different mechanism
        - prerequisite   – one must be configured before the other takes effect
        - paired_control – neither alone achieves the control; both are needed together
        - unrelated      – no meaningful security dependency (omit from output)

        Return a JSON array. Each element must have:
        {
        "candidate_id": "...",
        "candidate_name": "...",
        "relationship": "<label>",
        "tenant_configured": true|false,
        "severity": "finding" or "informational",
        "explanation": "<one concise sentence>"
        }

        Use severity="finding" for conflict and prerequisite relationships.
        Use severity="informational" for parent_child, alternative, and paired_control.
        If no meaningful relationships exist, return [].
        Output must be valid JSON parseable by json.loads().
        """
    ),
    "tools": [find_interdependencies_in_configurations],
}

int_agent_main = create_deep_agent(
    model=model,
    system_prompt="""You are an Intune security policy interdependency analyst.
    Your job is to find possible interdependencies and conflicts between Intune security configurations.    
    """,
    tools=[find_interdependencies_in_configurations]
)

def _file_data(path: str) -> dict:
    """Wrap a file's content in the FileData format deepagents expects."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    with open(path) as f:
        lines = f.read().splitlines()
    return {"content": lines, "created_at": now, "modified_at": now}


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    result = int_agent_main.invoke({"messages": [{"role": "user", "content": "What are the interdependencies between the configurations?"}],
        "files": {"relevant_configs.json": _file_data(os.path.join(os.path.dirname(__file__), "relevant_configs.json"))},
   })
    last_message = result["messages"][-1]
    print(last_message.content)