import json
import numpy as np
from sentence_transformers import SentenceTransformer

# ============================================================
# Load embeddings
# ============================================================

with open("wifi_embeddings.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Loaded {len(data)} embedded chunks.")


# ============================================================
# Load embedding model
# ============================================================

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)


# ============================================================
# Exclude metadata / non-evidence sections
# ============================================================

excluded_sections = {
    "What was retained",
    "What was removed",
    "Recommended RAG chunk structure",
    "Source map",
    "Research-use note",

    # Currently only headings / insufficient evidence
    "Core symptom taxonomy",
    "Selected Wi-Fi-specific legacy cases"
}

retrieval_data = [
    item
    for item in data
    if item["section"] not in excluded_sections
]

print(f"Retrieval candidates: {len(retrieval_data)}")


# ============================================================
# Build embedding matrix
# ============================================================

document_embeddings = np.array(
    [item["embedding"] for item in retrieval_data]
)


# ============================================================
# Query
# ============================================================

query = input("\nEnter Wi-Fi problem: ")

query_embedding = model.encode_query(
    query,
    normalize_embeddings=True
)


# ============================================================
# Similarity
# ============================================================

scores = document_embeddings @ query_embedding


# ============================================================
# Top K
# ============================================================

top_k = min(3, len(retrieval_data))

top_indices = np.argsort(scores)[::-1][:top_k]


# ============================================================
# Display
# ============================================================

print("\n" + "=" * 75)
print("TOP RETRIEVED EVIDENCE")
print("=" * 75)

for rank, index in enumerate(top_indices, start=1):

    item = retrieval_data[index]

    print(f"\nRank {rank}")
    print(f"Chunk ID: {item['chunk_id']}")
    print(f"Section: {item['section']}")
    print(f"Pages: {item['pages']}")
    print(f"Similarity: {scores[index]:.4f}")

    print("\nEvidence:")
    print(item["text"][:1500])

    print("-" * 75)