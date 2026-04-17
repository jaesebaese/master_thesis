import os

from langchain.chat_models import init_chat_model
from langchain.tools import tool
import json
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

OLLAMA_MODEL = "mistral-nemo:latest"
SETTINGS_JSON = os.path.join(os.path.dirname(__file__), "../intune_configurations/intune_configuration_settings.json")
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "../intune_configurations/chroma_db")

# Initialize the model
model = init_chat_model(model=OLLAMA_MODEL, model_provider="ollama", temperature=0.0)


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


def _get_collection():
    """Load the persisted ChromaDB collection. Raises if the DB hasn't been built yet."""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    ef = DefaultEmbeddingFunction()
    return client.get_collection("intune_settings", embedding_function=ef)


@tool
def policy_analyzer(query: str) -> str:
    """Search the Intune settings vector database for configuration settings
    that are semantically similar to the query.  Returns the top matches with
    their id, name, description, and platform.

    Args:
        query: A natural-language description of the settings you are looking
               for, e.g. 'BitLocker recovery password options' or
               'block executable content from email'.
    """
    collection = _get_collection()

    results = collection.query(
        query_texts=[query],
        n_results=10,
        include=["metadatas", "distances", "documents"],
    )

    hits = []
    for meta, distance, _ in zip(
        results["metadatas"][0],
        results["distances"][0],
        results["documents"][0],
    ):
        hits.append({
            "id": meta.get("id"),
            "name": meta.get("name"),
            "description": meta.get("description"),
            "platform": meta.get("platform"),
            "similarity_score": round(1 - distance, 4),
        })

    return json.dumps(hits, indent=2)


policy_agent = {
    "name": "policy_agent",
    "description": (
        "Searches the Intune settings catalog using semantic similarity. "
        "Use this to find which configuration settings relate to a given topic "
        "such as 'BitLocker', 'firewall inbound rules', or 'password complexity'."
    ),
    "system_prompt": (
        "You are a helpful Intune policy expert. "
        "Use the policy_analyzer tool to find relevant Intune configuration settings "
        "for the user's query. Present the results clearly, grouped by platform, "
        "and explain what each setting does based on its name and description."
    ),
    "tools": [policy_analyzer],
}


if __name__ == "__main__":
    build_intune_vector_db(force_rebuild=False)
    print("\n--- Similarity search: 'Password Requirements' ---\n")
    policy = """4. Password Requirements
        4.1 General Principles
        All passwords must be created in a way that ensures they are difficult to guess or compromise. Users must ensure that passwords are unique to each system and are not reused across different services. Passwords must remain confidential at all times and must not be shared with any other individual, including IT personnel. Furthermore, passwords must not be stored in plain text, whether digitally or physically, unless they are protected by approved secure storage mechanisms such as password managers.
        4.2 Complexity Requirements
        Passwords must meet a minimum length of twelve characters and include a combination of uppercase letters, lowercase letters, numbers, and special characters. Users must avoid using easily guessable information such as names, usernames, dates of birth, or common words found in dictionaries. The intent is to ensure that passwords are resistant to brute-force and dictionary-based attacks.
        4.3 Passphrases
        Where systems allow, users are encouraged to create passphrases instead of traditional passwords. A passphrase should consist of at least sixteen characters and be composed of multiple unrelated words combined with numbers or special characters. This approach increases memorability while maintaining a high level of security.
        4.4 Password Reuse
        To reduce the risk of compromise, users must not reuse previous passwords. The organization will enforce controls to prevent the reuse of at least the last ten passwords. In addition, users should ensure that passwords used within the organization are not reused for personal accounts or external services.
        4.5 Password Expiry
        Passwords must be changed periodically to reduce the risk of long-term exposure. Standard user accounts must update their passwords at least every ninety days, while privileged accounts must be updated every sixty days. In all cases, passwords must be changed immediately if there is any suspicion that they have been compromised."""
    print(policy_analyzer.invoke(policy))