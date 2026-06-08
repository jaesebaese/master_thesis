import asyncio

from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
import json
import os
from activity_stream import astream_activity
from rich_renderer import RichRenderer
from policy_agent import _get_intune_collection, SETTINGS_JSON
from preprocessing_at_startup import flatten_for_relevance
from deepagents import create_deep_agent
import logging



OPENAI_MODEL = "gpt-5.4-nano-2026-03-17"
OLLAMA_MODEL = "mistral-nemo:latest"
#model = init_chat_model(model=OLLAMA_MODEL, model_provider="ollama", temperature=0.0)


# Initialize the model
model = init_chat_model(model=OPENAI_MODEL, model_provider="openai", temperature=0.0)

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

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s", 
    handlers=[
        logging.FileHandler("agent.log", mode='w'),  # overwrite log file on each run
    ],)


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
            content = response.content
            if isinstance(content, list):
                content = "".join(
                    b.get("text", "") if isinstance(b, dict) else str(b) for b in content
                )
            raw = (content or "").strip()
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

@tool
def find_catalog_interdependencies(runtime: ToolRuntime) -> str:
    """Look up benchmark settings in the Microsoft Intune catalog, resolve their
    catalog dependentOn parents, then run two structural analyses scoped to those
    settings and their parents:
      - Conflict detection: same setting configured with different values across policies/groups
      - Unmet prerequisites: catalog-declared parents that are absent from the tenant

    Fast and LLM-free. Call this first; use analyze_requirements_against_tenant for
    deeper semantic analysis.

    Args:
        benchmark_output: Output from compare_search_results_to_tenant with a
                          'results' list, each having 'setting_definition_id'.
    """
    files = runtime.state.get("files", {})
    file_entry = files.get("/tenant_configs_vs_benchmark.json") or files.get("tenant_configs_vs_benchmark.json")
    
    if file_entry is None:
        return json.dumps({"error": "tenant_configs_vs_benchmark.json not found. Ensure policy_agent has run first."})
    if isinstance(file_entry, dict):
        raw = file_entry.get("content", [])
        content_str = "\n".join(raw) if isinstance(raw, list) else str(raw)
    else:
        content_str = str(file_entry)

    benchmark_output = json.loads(content_str)

    results = benchmark_output.get("results", [])
    setting_ids = [r["setting_definition_id"] for r in results if r.get("setting_definition_id")]

    if not setting_ids:
        print("No setting_definition_ids found in benchmark_output.")
        return json.dumps({"error": "No setting_definition_ids found."})

    # Load catalog
    with open(SETTINGS_JSON) as f:
        catalog = {s["id"]: s for s in json.load(f)}

    # Catalog lookup — collect parent→child mapping via dependentOn
    catalog_hits: dict[str, dict] = {}
    parent_to_children: dict[str, set[str]] = {}

    for sid in setting_ids:
        entry = catalog.get(sid)
        if not entry:
            continue
        catalog_hits[sid] = entry
        for dep in entry.get("dependentOn", []):
            parent_id = dep.get("parentSettingId") or dep.get("dependentOn")
            if parent_id:
                parent_to_children.setdefault(parent_id, set()).add(sid)

    parent_ids = set(parent_to_children.keys())
    new_parent_ids = parent_ids - set(setting_ids)
    parent_catalog_hits = {pid: catalog[pid] for pid in new_parent_ids if pid in catalog}

    # Load tenant, build flat list and index
    tenant_path = os.path.join(
        os.path.dirname(__file__), "../configurations/policies_and_settings_expand_assign.json"
    )
    with open(tenant_path) as f:
        policies = json.load(f)

    tenant_flat = flatten_for_relevance(policies, catalog)
    tenant_index: dict[str, list[dict]] = {}
    for s in tenant_flat:
        tenant_index.setdefault(s["id"], []).append(s)

    policy_groups: dict[str, set[str]] = {}
    for policy in policies:
        pid = policy.get("id", "")
        policy_groups[pid] = {
            a["target"]["groupId"]
            for a in policy.get("assignments", [])
            if a.get("target", {}).get("groupId")
        }

    # --- Block A: Structural conflict detection (scoped to benchmark + parent IDs) ---
    structural_conflicts: list[dict] = []
    scoped_ids = set(setting_ids) | new_parent_ids

    for sid, entries in tenant_index.items():
        if sid not in scoped_ids:
            continue
        if len(entries) < 2:
            continue
        unique_values = {str(e.get("configured_value")) for e in entries}
        if len(unique_values) <= 1:
            continue

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
                "severity": "informational",
                "reason": "Different values configured for different groups: intentional differentiation, not a conflict.",
            })

    # --- Block B: Unmet catalog prerequisites ---
    unmet_catalog_prerequisites: list[dict] = []
    seen_unmet: set[tuple] = set()

    for pid in new_parent_ids:
        if tenant_index.get(pid):
            continue
        cat_entry = parent_catalog_hits.get(pid, {})
        parent_name = cat_entry.get("displayName") or cat_entry.get("name") or pid
        for child_sid in parent_to_children.get(pid, set()):
            key = (child_sid, pid)
            if key in seen_unmet:
                continue
            seen_unmet.add(key)
            child_cat = catalog_hits.get(child_sid, {})
            child_name = child_cat.get("displayName") or child_cat.get("name") or child_sid
            unmet_catalog_prerequisites.append({
                "type": "unmet_catalog_prerequisite",
                "setting_id": child_sid,
                "setting_name": child_name,
                "missing_parent_id": pid,
                "missing_parent_name": parent_name,
                "severity": "finding",
                "reason": (
                    f"Catalog dependentOn: '{child_name}' depends on parent "
                    f"'{parent_name}' ({pid}) which is not configured in the tenant."
                ),
            })

    # Partition
    conflict_findings = [c for c in structural_conflicts if c.get("severity") == "finding"]
    conflict_informational = [c for c in structural_conflicts if c.get("severity") == "informational"]
    all_findings = conflict_findings + unmet_catalog_prerequisites

    # Build per-setting catalog & tenant status records
    settings_catalog_status = []
    for sid in setting_ids:
        cat = catalog_hits.get(sid, {})
        parents = [
            dep.get("parentSettingId") or dep.get("dependentOn")
            for dep in cat.get("dependentOn", [])
            if dep.get("parentSettingId") or dep.get("dependentOn")
        ]
        hits = tenant_index.get(sid, [])
        settings_catalog_status.append({
            "setting_id": sid,
            "name": cat.get("displayName") or cat.get("name") or "(not in catalog)",
            "description": (cat.get("description") or "")[:120],
            "in_catalog": bool(cat),
            "depends_on": parents,
            "tenant_status": "configured" if hits else "not_configured",
            "tenant_occurrences": [
                {
                    "policy_name": h.get("policy_name"),
                    "configured_value_label": h.get("configured_value_label") or h.get("configured_value"),
                }
                for h in hits
            ],
        })

    parent_catalog_status = []
    for pid in sorted(new_parent_ids):
        cat = parent_catalog_hits.get(pid, {})
        hits = tenant_index.get(pid, [])
        parent_catalog_status.append({
            "setting_id": pid,
            "name": cat.get("displayName") or cat.get("name") or "(not in catalog)",
            "tenant_status": "configured" if hits else "not_configured",
            "tenant_occurrences": [
                {
                    "policy_name": h.get("policy_name"),
                    "configured_value_label": h.get("configured_value_label") or h.get("configured_value"),
                }
                for h in hits
            ],
        })

    return json.dumps({
        "summary": {
            "settings_analyzed": len(setting_ids),
            "structural_conflicts": len(conflict_findings),
            "unmet_catalog_prerequisites": len(unmet_catalog_prerequisites),
            "informational_relationships": len(conflict_informational),
        },
        "settings_catalog_status": settings_catalog_status,
        "parent_catalog_status": parent_catalog_status,
        "findings": all_findings,
        "informational": conflict_informational,
    }, indent=2, default=list)


interdependency_agent = {
    "name": "interdependency_agent",
    "description": (
        "Analyzes Intune security policy interdependencies. "
        "Calls find_catalog_interdependencies to detect structural conflicts, unmet prerequisites, "
        "and parent-child relationships among Intune settings and their catalog parents. "
    ),
    "system_prompt": ("""\
You are an Intune security policy interdependency analyst embedded in a multi-agent security review workflow.

Your job is to surface real configuration errors and gaps in the tenant by running the structural analysis.

## Step 1 — Structural analysis (call find_catalog_interdependencies)

Call find_catalog_interdependencies with no arguments. It reads tenant_configs_vs_benchmark.json from the virtual filesystem and returns:
- structural_conflicts: settings configured to contradictory values across policies or groups
- unmet_catalog_prerequisites: CIS benchmark settings whose catalog-defined parent is not present in the tenant

After the tool returns, write the full JSON result to a file named "catalog_interdependencies.json" with the write_file tool.

## Step 2 — Consolidated report

Only after the file has been written, produce a structured findings report with the following sections:

### Structural Conflicts
List every entry from the structural_conflicts block. For each, state the setting name,
the conflicting policies/groups, the differing values, and the severity.
If the block is empty, write "No structural conflicts identified."

### Unmet Prerequisites
List every entry from the unmet_catalog_prerequisites block. For each, state which benchmark
setting depends on the missing parent and what the missing parent is.
If the block is empty, write "No unmet prerequisites identified."

### Informational
List any entries with severity="informational" here as a brief summary.
If none, omit this section.
                      
### Parent-Child Relationships
Using settings_catalog_status[].depends_on and parent_catalog_status[], build a
dependency summary showing each benchmark setting and its catalog parent(s):

parent_name
  <setting_name> → depends on → <parent_name> (<parent tenant_status>)

Group by parent. If a parent is not_configured, mark it NOT CONFIGURED.
If all parents are configured, mark them CONFIGURED.
Omit settings with no depends_on entries.

## Rules
- Do not invent findings. Only report what the tool returns.
- If a tool returns an error, report it clearly and do not proceed to the report.
- severity="finding" entries belong in the findings sections above.
- severity="informational" entries belong in the Informational section.
- Any other severity value should be treated as a finding.
"""
    ),
    "tools": [find_catalog_interdependencies],
    "model": model,
}

int_agent_main = create_deep_agent(
    model=model,
    system_prompt=interdependency_agent["system_prompt"],
    tools=[find_catalog_interdependencies],
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
    renderer = RichRenderer(logger=logger)

    #result = stream_agent_v2(agent, pending, config=run_config, on_interrupt=handle_interrupt)
    pending = {
        "messages": [{"role": "user", "content": "Check interdependecies in the security configurations"}],
        "files": {
            "tenant_configs_vs_benchmark.json": _file_data(os.path.join(os.path.dirname(__file__), "tenant_configs_vs_benchmark.json"))
        },
    }
    run_config = {"configurable": {"thread_id": "1"}}

    final_state = asyncio.run(
        astream_activity(int_agent_main, agent_input=pending, config=run_config, render=False, on_event=renderer)
    )
    print("\nFINAL STATE:\n" + json.dumps(final_state, indent=2))