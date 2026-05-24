from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from typing import List
import json
import os
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
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

REQUIREMENT_CLASSIFY_SYSTEM = """\
You are an Intune security policy compliance analyst.

You will receive:
1. A "requirement" — a security policy requirement with its intent and expected controls.
2. A list of "candidates" — Intune settings currently configured in the tenant that are
   semantically related to this requirement.

For each candidate classify its relationship to the requirement using exactly one label:
- satisfies    – the setting directly fulfills or strongly supports the requirement
- conflicts    – the setting is configured in a way that contradicts or undermines the requirement
- partial      – the setting partially addresses the requirement but is insufficient alone
- prerequisite – the setting must be in place for the requirement to take effect (and it is)
- unrelated    – no meaningful relationship (omit from output)

Return a JSON array. Each element must have:
{
  "candidate_id": "...",
  "candidate_name": "...",
  "configured_value_label": "...",
  "relationship": "<label>",
  "severity": "finding" | "informational",
  "explanation": "<one concise sentence>"
}

Use severity="finding" for conflicts only.
Use severity="informational" for satisfies, partial, and prerequisite.
If no meaningful relationships exist, return [].
Output must be valid JSON parseable by json.loads().
"""

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


def _build_tenant_collection(tenant_flat: list[dict]):
    """Build a throwaway in-memory ChromaDB collection from the tenant's flat settings.

    Uses EphemeralClient so nothing is persisted to disk. Deduplicates by setting ID,
    keeping the first occurrence. Returns the collection ready for querying.
    """
    client = chromadb.EphemeralClient()
    ef = DefaultEmbeddingFunction()
    col = client.create_collection(
        "tenant_settings_tmp",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    seen: set[str] = set()
    ids, docs, metas = [], [], []
    for rec in tenant_flat:
        sid = rec.get("id", "")
        if not sid or sid in seen:
            continue
        seen.add(sid)
        name = rec.get("name", sid)
        desc = rec.get("description", "")
        ids.append(sid)
        docs.append(f"{name}. {desc}".strip())
        metas.append({
            "id": sid,
            "name": name,
            "description": desc[:200],
            "configured_value_label": str(rec.get("configured_value_label") or ""),
            "policy_name": rec.get("policy_name", ""),
        })
    if ids:
        col.add(ids=ids, documents=docs, metadatas=metas)
    return col


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
    if file_entry is not None:
        if isinstance(file_entry, dict):
            raw = file_entry.get("content", [])
            content_str = "\n".join(raw) if isinstance(raw, list) else str(raw)
        else:
            content_str = str(file_entry)
    else:
        disk_path = os.path.join(os.path.dirname(__file__), "tenant_configs_vs_benchmark.json")
        try:
            with open(disk_path) as f:
                content_str = f.read()
        except FileNotFoundError:
            return json.dumps({"error": "tenant_configs_vs_benchmark.json not found in virtual filesystem or on disk."})

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

    # Print benchmark settings
    print("\n=== Benchmark Settings — Catalog & Tenant Status ===\n")
    for sid in setting_ids:
        cat = catalog_hits.get(sid, {})
        parents = [
            dep.get("parentSettingId") or dep.get("dependentOn")
            for dep in cat.get("dependentOn", [])
            if dep.get("parentSettingId") or dep.get("dependentOn")
        ]
        hits = tenant_index.get(sid, [])
        print(f"[BENCHMARK SETTING]  {sid}")
        print(f"  Name:        {cat.get('displayName') or cat.get('name', '(not in catalog)')}")
        print(f"  Description: {(cat.get('description') or '')[:120]}")
        print(f"  Depends on:  {parents if parents else 'none'}")
        if hits:
            for h in hits:
                print(f"  Tenant:      policy='{h.get('policy_name')}' | value='{h.get('configured_value_label') or h.get('configured_value')}'")
        else:
            print("  Tenant:      NOT CONFIGURED")
        print()

    if new_parent_ids:
        print("--- Parent Settings ---\n")
        for pid in sorted(new_parent_ids):
            cat = parent_catalog_hits.get(pid, {})
            hits = tenant_index.get(pid, [])
            status = "CONFIGURED" if hits else "NOT CONFIGURED — unmet prerequisite"
            print(f"[PARENT SETTING]  {pid}")
            print(f"  Name:        {cat.get('displayName') or cat.get('name', '(not in catalog)')}")
            print(f"  Tenant:      {status}")
            if hits:
                for h in hits:
                    print(f"             policy='{h.get('policy_name')}' | value='{h.get('configured_value_label') or h.get('configured_value')}'")
            print()

    if conflict_findings:
        print(f"\n--- Structural Conflicts ({len(conflict_findings)}) ---")
        for f in conflict_findings:
            print(f"  CONFLICT: {f['setting_name']} ({f['setting_id']}) — {f['reason']}")

    if unmet_catalog_prerequisites:
        print(f"\n--- Unmet Catalog Prerequisites ({len(unmet_catalog_prerequisites)}) ---")
        for f in unmet_catalog_prerequisites:
            print(f"  UNMET PREREQ: {f['setting_name']} missing parent '{f['missing_parent_name']}' ({f['missing_parent_id']})")

    return json.dumps({
        "summary": {
            "settings_analyzed": len(setting_ids),
            "structural_conflicts": len(conflict_findings),
            "unmet_catalog_prerequisites": len(unmet_catalog_prerequisites),
            "informational_relationships": len(conflict_informational),
        },
        "findings": all_findings,
        "informational": conflict_informational,
    }, indent=2, default=list)


@tool
def analyze_requirements_against_tenant(runtime: ToolRuntime) -> str:
    """Semantically match each security requirement from policy_requirements.json
    against the tenant's configured settings and classify the relationship via LLM.

    For each requirement, searches an in-memory ChromaDB collection built from the
    tenant's own settings (avoiding catalog ID format mismatches), then asks the LLM
    to classify each candidate as: satisfies, conflicts, partial, or prerequisite.

    Only 'conflicts' are emitted as findings; the rest are informational.

    Call this after find_catalog_interdependencies when deeper semantic analysis
    of requirement coverage is needed.
    """
    # Load catalog and tenant independently
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

    semantic_relationships: list[dict] = []
    requirements: list[dict] = []

    files = runtime.state.get("files", {})
    file_entry = files.get("/policy_requirements.json") or files.get("policy_requirements.json")
    if file_entry is not None:
        if isinstance(file_entry, dict):
            raw = file_entry.get("content", [])
            requirements = "\n".join(raw) if isinstance(raw, list) else str(raw)
        else:
            requirements = str(file_entry)
    else:
        disk_path = os.path.join(os.path.dirname(__file__), "policy_requirements.json")
        print("TAKE REQUIREMENTS FROM DISK")
        try:
            with open(disk_path) as f:
                requirements = f.read()
        except FileNotFoundError:
            return json.dumps({"error": "requirements.json not found in virtual filesystem or on disk."})

    try:
        parsed = json.loads(requirements)
    except json.JSONDecodeError:
        import ast
        parsed = ast.literal_eval(requirements)
    req_list: list[dict] = parsed.get("requirements", parsed) if isinstance(parsed, dict) else parsed
    

    tenant_collection = _build_tenant_collection(tenant_flat)

    for req in req_list[:30]:
        query = f"{req.get('source_text', '')} {req.get('control_intent', '')}".strip()
        try:
            sem_results = tenant_collection.query(
                query_texts=[query],
                n_results=11,
                include=["metadatas", "distances"],
            )
        except Exception:
            continue

        candidates = []
        for meta, dist in zip(sem_results["metadatas"][0], sem_results["distances"][0]):
            cid = meta.get("id", "")
            score = round(1 - dist, 4)
            if score < 0.65:
                continue
            tenant_hits_cid = tenant_index.get(cid, [])
            candidates.append({
                "id": cid,
                "name": meta.get("name", cid),
                "description": meta.get("description", ""),
                "similarity_score": score,
                "configured_value_label": tenant_hits_cid[0].get("configured_value_label") if tenant_hits_cid else None,
                "configured_value": tenant_hits_cid[0].get("configured_value") if tenant_hits_cid else None,
                "policy_name": meta.get("policy_name", ""),
            })

        if not candidates:
            continue

        requirement_summary = {
            "requirement_id": req.get("requirement_id"),
            "source_text": req.get("source_text"),
            "security_domain": req.get("security_domain"),
            "control_intent": req.get("control_intent"),
            "expected_value": req.get("expected_value"),
            "expected_unit": req.get("expected_unit"),
        }
        user_prompt = (
            f"REQUIREMENT:\n{json.dumps(requirement_summary, indent=2)}\n\n"
            f"TENANT SETTINGS:\n{json.dumps(candidates, indent=2)}\n\n"
            "Classify each tenant setting's relationship to fulfilling this requirement."
        )

        response = model.invoke([
            {"role": "system", "content": REQUIREMENT_CLASSIFY_SYSTEM},
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
            rel["requirement_id"] = req.get("requirement_id")
            rel["source_text"] = req.get("source_text")
            rel["security_domain"] = req.get("security_domain")
            semantic_relationships.append(rel)

    semantic_findings = [r for r in semantic_relationships if r.get("severity") == "finding"]
    semantic_informational = [r for r in semantic_relationships if r.get("severity") == "informational"]

    if semantic_findings:
        print(f"\n--- Requirement Conflicts ({len(semantic_findings)}) ---")
        for f in semantic_findings:
            print(f"  [CONFLICTS] {f.get('requirement_id')} ({f.get('security_domain', '')}) ↔ {f.get('candidate_name')}: {f.get('explanation', '')}")

    return json.dumps({
        "summary": {
            "requirements_analyzed": len(requirements[:20]),
            "semantic_findings": len(semantic_findings),
            "informational_relationships": len(semantic_informational),
        },
        "findings": semantic_findings,
        "informational": semantic_informational,
    }, indent=2, default=list)


interdependency_agent = {
    "name": "interdependency_agent",
    "description": (
        "Analyzes Intune security policy interdependencies and requirement compliance. "
        "First calls find_catalog_interdependencies to detect structural conflicts and unmet prerequisites "
        "among CIS benchmark settings and their catalog parents. "
        "Then calls analyze_requirements_against_tenant to classify how tenant settings satisfy, "
        "conflict with, or partially address each security requirement in policy_requirements.json. "
        "Writes both result sets to files and produces a consolidated findings report."
    ),
    "system_prompt": ("""\
You are an Intune security policy interdependency analyst embedded in a multi-agent security review workflow.

Your job is to surface real configuration errors, gaps, and conflicts in the tenant by running two focused analyses in sequence.

## Step 1 — Structural analysis (call find_catalog_interdependencies)

Call find_catalog_interdependencies with no arguments. It reads tenant_configs_vs_benchmark.json from the virtual filesystem and returns:
- structural_conflicts: settings configured to contradictory values across policies or groups
- unmet_catalog_prerequisites: CIS benchmark settings whose catalog-defined parent is not present in the tenant

After the tool returns, write the full JSON result to a file named "catalog_interdependencies.json".

## Step 2 — Requirements compliance analysis (call analyze_requirements_against_tenant)

Call analyze_requirements_against_tenant with no arguments. It reads policy_requirements.json and the tenant's own policies independently, then classifies each tenant setting's relationship to each security requirement:
- conflicts: tenant setting is configured in a way that contradicts the requirement (severity=finding)
- satisfies / partial / prerequisite: setting supports the requirement (severity=informational)

After the tool returns, write the full JSON result to a file named "requirements_analysis_tenant.json".

## Step 3 — Consolidated report

After both tools complete, produce a structured findings report with three sections:

### Structural Conflicts
List every entry from the structural_conflicts block. For each, state the setting name, the conflicting policies/groups, the differing values, and the severity.

### Unmet Prerequisites
List every entry from the unmet_catalog_prerequisites block. For each, state which benchmark setting depends on the missing parent and what the missing parent is.

### Requirement Conflicts
List every entry from the requirements analysis where relationship="conflicts". For each, state the requirement ID, the source text, the conflicting tenant setting, its configured value, and the explanation.

### Informational
Briefly summarise how many requirements are satisfied, partially addressed, or have prerequisites in place. No need to enumerate them individually unless the count is low (≤5).

## Rules
- Always run both tools, even if Step 1 returns no findings — the requirement compliance check is independent.
- Do not invent findings. Only report what the tools return.
- If a tool returns an error, report it clearly and continue to the next step.
- Severity="finding" entries belong in the findings sections above. Severity="informational" entries belong in the Informational summary.
"""
    ),
    "tools": [find_catalog_interdependencies, analyze_requirements_against_tenant],
}

int_agent_main = create_deep_agent(
    model=model,
    system_prompt="""You are an Intune security policy interdependency analyst.
    Your job is to find possible interdependencies and conflicts between Intune security configurations.
    1. Call find_catalog_interdependencies tool to find interdependencies from the different settings
    2. Use write file to write the results from find_catalog_interdependencies.
    3. Call analyze_requirements_against_tenant to identify which tenant
    settings conflict with or satisfy the security requirements.
    4.  Use write file to write the results from analyze_requirements_against_tenant.
    """,
    tools=[find_catalog_interdependencies, analyze_requirements_against_tenant]
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

    benchmark_output = {
  "summary": {
    "total": 10,
    "compliant": 10,
    "non_compliant": 0,
    "not_configured": 0
  },
  "results": [
    {
      "setting_definition_id": "device_vendor_msft_policy_config_devicelock_minimumpasswordage",
      "matched_by_requirements": [
        "REQ-001",
        "REQ-002",
        "REQ-010",
        "REQ-011",
        "REQ-012"
      ],
      "status": "compliant",
      "cis_match": {
        "cis_id": "26.9",
        "cis_title": "(L1) Ensure 'Minimum Password Age' is set to '1 or more day(s)'",
        "cis_reference_value": "1",
        "cis_reference_value_display": "1",
        "benchmark_policy": "CIS (L1) Device Lock & WHFB - Windows 11 Intune 4.0.0 "
      },
      "tenant_configurations": [
        {
          "policy_name": "CIS - Device Lock [L1] - Windows 11 - v4.0.0.0",
          "policy_id": "76b42865-c293-47ff-b0be-6cce07766fea",
          "raw_value": 1,
          "value_type": "integer",
          "compliant": "true"
        }
      ]
    },
    {
      "setting_definition_id": "device_vendor_msft_policy_config_devicelock_devicepasswordexpiration",
      "matched_by_requirements": [
        "REQ-001",
        "REQ-002",
        "REQ-012"
      ],
      "status": "compliant",
      "cis_match": {
        "cis_id": "26.4",
        "cis_title": "(L1) Ensure 'Device Password Enabled: Device Password Expiration' is set to '365 or fewer days, but not 0'",
        "cis_reference_value": "365",
        "cis_reference_value_display": "365",
        "benchmark_policy": "CIS (L1) Device Lock & WHFB - Windows 11 Intune 4.0.0 "
      },
      "tenant_configurations": [
        {
          "policy_name": "CIS - Device Lock [L1] - Windows 11 - v4.0.0.0",
          "policy_id": "76b42865-c293-47ff-b0be-6cce07766fea",
          "raw_value": 365,
          "value_type": "integer",
          "compliant": "true"
        }
      ]
    },
    {
      "setting_definition_id": "device_vendor_msft_policy_config_devicelock_devicepasswordhistory",
      "matched_by_requirements": [
        "REQ-001",
        "REQ-012"
      ],
      "status": "compliant",
      "cis_match": {
        "cis_id": "26.5",
        "cis_title": "(L1) Ensure 'Device Password Enabled: Device Password History' is set to '24 or more password(s)'",
        "cis_reference_value": "24",
        "cis_reference_value_display": "24",
        "benchmark_policy": "CIS (L1) Autopilot - Windows 11 Intune 4.0.0"
      },
      "tenant_configurations": [
        {
          "policy_name": "CIS - Device Lock [L1] - Windows 11 - v4.0.0.0",
          "policy_id": "76b42865-c293-47ff-b0be-6cce07766fea",
          "raw_value": 24,
          "value_type": "integer",
          "compliant": "true"
        }
      ]
    },
    {
      "setting_definition_id": "device_vendor_msft_policy_config_audit_accountlogonlogoff_auditaccountlockout",
      "matched_by_requirements": [
        "REQ-004",
        "REQ-005"
      ],
      "status": "compliant",
      "cis_match": {
        "cis_id": "6.2",
        "cis_title": "(L1) Ensure 'Account Logon Logoff Audit Account Lockout' is set to include 'Failure'",
        "cis_reference_value": "device_vendor_msft_policy_config_audit_accountlogonlogoff_auditaccountlockout_2",
        "cis_reference_value_display": "2",
        "benchmark_policy": "CIS (L1) Auditing (6) - Windows 11 Intune 4.0.0"
      },
      "tenant_configurations": [
        {
          "policy_name": "CIS - Auditing [L1] - Windows 11 - v4.0.0",
          "policy_id": "bce5b2df-6dd8-4aab-9747-12bdd24739fc",
          "raw_value": "device_vendor_msft_policy_config_audit_accountlogonlogoff_auditaccountlockout_2",
          "value_type": "choice",
          "compliant": "true"
        }
      ]
    },
    {
      "setting_definition_id": "device_vendor_msft_policy_config_devicelock_maxdevicepasswordfailedattempts",
      "matched_by_requirements": [
        "REQ-004",
        "REQ-005"
      ],
      "status": "compliant",
      "cis_match": {
        "cis_id": "26.6",
        "cis_title": "(L1) Ensure 'Device Password Enabled: Max Device Password Failed Attempts' is set to '5 or fewer failed attempt(s), but not 0'",
        "cis_reference_value": "5",
        "cis_reference_value_display": "5",
        "benchmark_policy": "CIS (L1) Device Lock & WHFB - Windows 11 Intune 4.0.0 "
      },
      "tenant_configurations": [
        {
          "policy_name": "CIS - Device Lock [L1] - Windows 11 - v4.0.0.0",
          "policy_id": "76b42865-c293-47ff-b0be-6cce07766fea",
          "raw_value": 5,
          "value_type": "integer",
          "compliant": "true"
        }
      ]
    },
    {
      "setting_definition_id": "device_vendor_msft_policy_config_devicelock_mindevicepasswordcomplexcharacters",
      "matched_by_requirements": [
        "REQ-010",
        "REQ-011"
      ],
      "status": "compliant",
      "cis_match": {
        "cis_id": "26.3",
        "cis_title": "(L1) Ensure 'Device Password Enabled: Min Device Password Complex Characters' is set to 'Digits and lowercase letters are required'",
        "cis_reference_value": "device_vendor_msft_policy_config_devicelock_mindevicepasswordcomplexcharacters_2",
        "cis_reference_value_display": "2",
        "benchmark_policy": "CIS (L1) Device Lock & WHFB - Windows 11 Intune 4.0.0 "
      },
      "tenant_configurations": [
        {
          "policy_name": "CIS - Device Lock [L1] - Windows 11 - v4.0.0.0",
          "policy_id": "76b42865-c293-47ff-b0be-6cce07766fea",
          "raw_value": "device_vendor_msft_policy_config_devicelock_mindevicepasswordcomplexcharacters_2",
          "value_type": "choice",
          "compliant": "true"
        }
      ]
    },
    {
      "setting_definition_id": "device_vendor_msft_policy_config_devicelock_mindevicepasswordlength",
      "matched_by_requirements": [
        "REQ-010",
        "REQ-011"
      ],
      "status": "compliant",
      "cis_match": {
        "cis_id": "26.8",
        "cis_title": "(L1) Ensure 'Device Password Enabled: Min Device Password Length' is set to '14 or more character(s)'",
        "cis_reference_value": "14",
        "cis_reference_value_display": "14",
        "benchmark_policy": "CIS (L1) Device Lock & WHFB - Windows 11 Intune 4.0.0 "
      },
      "tenant_configurations": [
        {
          "policy_name": "CIS - Device Lock [L1] - Windows 11 - v4.0.0.0",
          "policy_id": "76b42865-c293-47ff-b0be-6cce07766fea",
          "raw_value": 14,
          "value_type": "integer",
          "compliant": "true"
        }
      ]
    },
    {
      "setting_definition_id": "device_vendor_msft_policy_config_devicelock_alphanumericdevicepasswordrequired",
      "matched_by_requirements": [
        "REQ-011"
      ],
      "status": "compliant",
      "cis_match": {
        "cis_id": "26.2",
        "cis_title": "(L1) Ensure 'Device Password Enabled: Alphanumeric Device Password Required' is set to 'Password or Alphanumeric PIN required'",
        "cis_reference_value": "device_vendor_msft_policy_config_devicelock_alphanumericdevicepasswordrequired_0",
        "cis_reference_value_display": "0",
        "benchmark_policy": "CIS (L1) Device Lock & WHFB - Windows 11 Intune 4.0.0 "
      },
      "tenant_configurations": [
        {
          "policy_name": "CIS - Device Lock [L1] - Windows 11 - v4.0.0.0",
          "policy_id": "76b42865-c293-47ff-b0be-6cce07766fea",
          "raw_value": "device_vendor_msft_policy_config_devicelock_alphanumericdevicepasswordrequired_0",
          "value_type": "choice",
          "compliant": "true"
        }
      ]
    },
    {
      "setting_definition_id": "device_vendor_msft_policy_config_remotedesktopservices_donotallowpasswordsaving",
      "matched_by_requirements": [
        "REQ-012"
      ],
      "status": "compliant",
      "cis_match": {
        "cis_id": "4.11.36.3.2",
        "cis_title": "(L1) Ensure 'Do not allow passwords to be saved' is set to 'Enabled'",
        "cis_reference_value": "device_vendor_msft_policy_config_remotedesktopservices_donotallowpasswordsaving_1",
        "cis_reference_value_display": "1",
        "benchmark_policy": "CIS (L1) Admin Templates - Windows Components (4.11) - Windows 11 Intune 4.0.0"
      },
      "tenant_configurations": [
        {
          "policy_name": "CIS - Windows Components - Restrictions [L1] - Windows 11 - v4.0.0.0",
          "policy_id": "c4dc50cf-85bf-49ad-a4d7-c594a7096e98",
          "raw_value": "device_vendor_msft_policy_config_remotedesktopservices_donotallowpasswordsaving_1",
          "value_type": "choice",
          "compliant": "true"
        }
      ]
    },
    {
      "setting_definition_id": "device_vendor_msft_policy_config_webthreatdefense_notifypasswordreuse",
      "matched_by_requirements": [
        "REQ-012"
      ],
      "status": "compliant",
      "cis_match": {
        "cis_id": "76.1.2",
        "cis_title": "(L1) Ensure 'Notify Password Reuse' is set to 'Enabled'",
        "cis_reference_value": "device_vendor_msft_policy_config_webthreatdefense_notifypasswordreuse_1",
        "cis_reference_value_display": "1",
        "benchmark_policy": "CIS (L1) Section 15 - 104 - Windows 11 Intune 4.0.0"
      },
      "tenant_configurations": [
        {
          "policy_name": "CIS - Enhanced Phishing Protection [L1] - Windows 11 - v4.0.0.0",
          "policy_id": "a90b4ff2-a29d-4e5d-b138-4e2a19c0309d",
          "raw_value": "device_vendor_msft_policy_config_webthreatdefense_notifypasswordreuse_1",
          "value_type": "choice",
          "compliant": "true"
        }
      ]
    }
  ]
}
    
    result = int_agent_main.invoke({
        "messages": [{"role": "user", "content": "Check interdependecies and compliance with the security policy"}],
        "files": {
            "policy_requirements.json": _file_data(os.path.join(os.path.dirname(__file__), "policy_requirements.json")),
            "tenant_configs_vs_benchmark.json": _file_data(os.path.join(os.path.dirname(__file__), "tenant_configs_vs_benchmark.json"))
        },
    })