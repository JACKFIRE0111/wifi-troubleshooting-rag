import json
from sentence_transformers import SentenceTransformer

# =========================
# 1. Load section-aware chunks
# =========================

with open("wifi_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks.")

# =========================
# 2. Load embedding model
# =========================

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

print("Embedding model loaded.")

# =========================
# 3. Prepare document texts
# =========================

texts = [chunk["text"] for chunk in chunks]

# =========================
# 4. Generate document embeddings
# =========================

embeddings = model.encode_document(
    texts,
    normalize_embeddings=True
)

print("Embeddings generated.")

# =========================
# 5. Save embeddings + metadata
# =========================

output = []

for chunk, embedding in zip(chunks, embeddings):

    output.append({
        "chunk_id": chunk["chunk_id"],
        "section": chunk["section"],
        "pages": chunk["pages"],
        "text": chunk["text"],
        "embedding": embedding.tolist()
    })


with open("wifi_embeddings.json", "w", encoding="utf-8") as f:

    json.dump(
        output,
        f,
        ensure_ascii=False
    )

# =========================
# 6. Verify
# =========================

print(f"Embedded chunks: {len(output)}")
print(f"Embedding dimension: {len(output[0]['embedding'])}")
print("Saved to: wifi_embeddings.json")