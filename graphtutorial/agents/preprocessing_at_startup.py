import os
import json
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

CIS_BENCHMARK_JSON = os.path.join(
    os.path.dirname(__file__),
    "../configurations/cis_benchmarks/matched_all_level1.json",
)
CIS_CHROMA_DB_PATH = os.path.join(
    os.path.dirname(__file__),
    "../configurations/cis_benchmarks/chroma_db",
)

TENANT_SETTINGS_JSON = os.path.join(
    os.path.dirname(__file__),
    "../configurations/policies_and_settings_expand_assign.json",
)

INTUNE_SETTINGS_JSON = os.path.join(os.path.dirname(__file__), "../intune_configurations/intune_configuration_settings.json")




def _resolve_choice_label(defn: dict, chosen_id: str) -> tuple[str, list[str]]:
    options = defn.get("options", []) if defn else []
    if not options:
        return chosen_id, []
    option_map = {
        o["itemId"]: (o.get("displayName") or o.get("name") or o["itemId"])
        for o in options
    }
    label = option_map.get(chosen_id, chosen_id)
    return label, list(option_map.values())


def _walk_setting_instance(
    instance: dict,
    policy_info: dict,
    catalog: dict,
    parent_chain: list,
    parent_chosen_label: str | None,
    out: list,
) -> None:
    sid = instance.get("settingDefinitionId", "")
    dtype = instance.get("@odata.type", "")
    defn = catalog.get(sid)

    chosen_id = None
    setting_type = "unknown"
    available_options: list[str] = []

    if "ChoiceSettingInstance" in dtype:
        chosen_id = instance.get("choiceSettingValue", {}).get("value", "") or None
        setting_type = "choice"
    elif "SimpleSettingInstance" in dtype:
        chosen_id = instance.get("simpleSettingValue", {}).get("value", "")
        setting_type = "simple"
    elif "SimpleSettingCollectionInstance" in dtype:
        chosen_id = [
            v.get("value")
            for v in instance.get("simpleSettingCollectionValue", [])
        ]
        setting_type = "simpleCollection"
    elif "GroupSettingCollectionInstance" in dtype or "GroupSettingInstance" in dtype:
        chosen_id = None
        setting_type = "group"

    label = str(chosen_id)
    if setting_type == "choice" and chosen_id:
        label, available_options = _resolve_choice_label(defn or {}, chosen_id)

    if sid:
        record = {
            "id": sid,
            "name": (defn.get("displayName") or defn.get("name") or sid) if defn else sid,
            "description": (defn.get("description") or defn.get("helpText") or "") if defn else "",
            "platform": (defn.get("applicability", {}).get("platform", "") if defn else ""),
            "type": setting_type,
            "configured_value": chosen_id,
            "configured_value_label": label,
            "parent_setting_id": parent_chain[-1] if parent_chain else None,
            "parent_chain": list(parent_chain),
            "policy_name": policy_info["policy_name"],
            "policy_id": policy_info["policy_id"],
        }
        if available_options:
            record["available_options"] = available_options
        if parent_chain:
            record["dependency_condition"] = {
                "parent_id": parent_chain[-1],
                "parent_chosen_label": parent_chosen_label,
                # `active` could be enriched later by interdependency_agent
                # using catalog metadata about which parent values gate which children
                "active_unknown": True,
            }
        out.append(record)

    new_chain = parent_chain + [sid] if sid else parent_chain
    new_parent_label = label if setting_type == "choice" else parent_chosen_label

    if "ChoiceSettingInstance" in dtype:
        for child in instance.get("choiceSettingValue", {}).get("children", []) or []:
            _walk_setting_instance(child, policy_info, catalog, new_chain, new_parent_label, out)
    elif "GroupSettingCollectionInstance" in dtype:
        for group_val in instance.get("groupSettingCollectionValue", []) or []:
            for child in group_val.get("children", []) or []:
                _walk_setting_instance(child, policy_info, catalog, new_chain, new_parent_label, out)
    elif "GroupSettingInstance" in dtype:
        for child in instance.get("groupSettingValue", {}).get("children", []) or []:
            _walk_setting_instance(child, policy_info, catalog, new_chain, new_parent_label, out)


def flatten_for_relevance(raw_policies: list, catalog: dict) -> list:
    """Walk every policy's settings tree and produce a flat list of enriched
    records — one per configured setting (parent and child alike).
    """
    flat = []
    for policy in raw_policies:
        policy_info = {
            "policy_name": policy.get("name", ""),
            "policy_id": policy.get("id", ""),
        }
        for setting in policy.get("settings", []):
            _walk_setting_instance(
                setting.get("settingInstance", {}),
                policy_info,
                catalog,
                parent_chain=[],
                parent_chosen_label=None,
                out=flat,
            )
    return flat


def build_tenant_collection(platform: str | None = None):
    """Build a throwaway in-memory ChromaDB collection from the tenant's flat settings.

    Uses EphemeralClient so nothing is persisted to disk. Deduplicates by setting ID,
    keeping the first occurrence. Returns the collection ready for querying.

    Args:
        platform: Optional case-insensitive substring filter on the 'platform' field
                  (e.g. "windows" to include only Windows settings). None = all platforms.
    """
    with open(INTUNE_SETTINGS_JSON) as f:
        catalog = {s["id"]: s for s in json.load(f)}

    with open(TENANT_SETTINGS_JSON) as f:
        policies = json.load(f)

    tenant_flat = flatten_for_relevance(policies, catalog)
    client = chromadb.EphemeralClient()
    ef = DefaultEmbeddingFunction()
    col = client.create_collection(
        "tenant_settings_tmp",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    seen: set[str] = set()
    ids, docs, metas = [], [], []
    platform_lower = platform.lower() if platform else None
    for rec in tenant_flat:
        if platform_lower and platform_lower not in (rec.get("platform") or "").lower():
            continue
        sid = rec.get("id", "")
        if not sid or sid in seen:
            continue
        seen.add(sid)
        name = rec.get("name", sid)
        desc = rec.get("description", "")
        value_label = str(rec.get("configured_value_label") or "")
        policy_name = rec.get("policy_name", "")
        # Richer document text improves recall against natural-language requirement queries
        doc_parts = [f"{name}."]
        if desc:
            doc_parts.append(desc)
        if value_label:
            doc_parts.append(f"Configured value: {value_label}.")
        if policy_name:
            doc_parts.append(f"Policy: {policy_name}.")
        ids.append(sid)
        docs.append(" ".join(doc_parts).strip())
        metas.append({
            "id": sid,
            "name": name,
            "description": desc[:200],
            "configured_value_label": value_label,
            "policy_name": policy_name,
        })
    if ids:
        col.add(ids=ids, documents=docs, metadatas=metas)
    return col



def build_cis_benchmark_vector_db(force_rebuild: bool = False):
    """Embed all CIS benchmark controls from matched_all_level1.json into a
    persistent ChromaDB collection.

    Flattens all policies → matched_configurations and deduplicates by
    setting_definition_id (first occurrence wins). Documents embed the CIS
    title, description, and rationale so the collection supports semantic
    search against natural-language security requirements.

    Pass force_rebuild=True to drop and recreate the collection.

    Returns the ChromaDB Collection object.
    """
    client = chromadb.PersistentClient(path=CIS_CHROMA_DB_PATH)
    ef = DefaultEmbeddingFunction()

    existing = [c.name for c in client.list_collections()]

    if "cis_benchmarks" in existing:
        if not force_rebuild:
            print("CIS benchmark collection already exists. Loading from disk.")
            return client.get_collection("cis_benchmarks", embedding_function=ef)
        print("force_rebuild=True — dropping existing collection.")
        client.delete_collection("cis_benchmarks")

    collection = client.create_collection(
        name="cis_benchmarks",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"Loading CIS benchmarks from {CIS_BENCHMARK_JSON} ...")
    with open(CIS_BENCHMARK_JSON) as f:
        data = json.load(f)

    ids, documents, metadatas = [], [], []
    seen: set[str] = set()

    for policy in data.get("policies", []):
        policy_meta = policy.get("policy_metadata", {})
        policy_name = policy_meta.get("intune_policy_name", "")
        policy_id = policy_meta.get("intune_policy_id", "")

        for item in policy.get("matched_configurations", []):
            sid = item.get("setting_definition_id", "")
            if not sid or sid in seen:
                continue
            seen.add(sid)

            cis = item.get("cis_benchmark", {})
            title = cis.get("Title", "")
            description = (cis.get("Description") or "")[:500]
            rationale = (cis.get("Rationale Statement") or "")[:500]

            document = f"{title}. {description} Rationale: {rationale}".strip()

            ids.append(sid)
            documents.append(document)
            metadatas.append({
                "setting_definition_id": sid,
                "cis_id": cis.get("Recommendation #", ""),
                "cis_section": cis.get("Section #", ""),
                "cis_title": title[:200],
                "configured_value": str(item.get("configured_value", "")),
                "raw_value": str(item.get("raw_value", "")),
                "value_type": item.get("value_type", ""),
                "policy_name": policy_name,
                "policy_id": policy_id,
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

    print(f"Done. {total} CIS benchmark controls stored in {CIS_CHROMA_DB_PATH}")
    return collection


if __name__ == "__main__":
    build_cis_benchmark_vector_db(force_rebuild=True)
