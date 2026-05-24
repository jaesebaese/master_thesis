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
