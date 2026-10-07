import json
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# Load test cases
# ============================================================

with open("wifi_test_cases.json", "r", encoding="utf-8") as f:
    test_cases = json.load(f)


# ============================================================
# Load ground truth
# IMPORTANT:
# Ground truth is ONLY used for evaluation, never retrieval.
# ============================================================

with open("wifi_ground_truth.json", "r", encoding="utf-8") as f:
    ground_truth = json.load(f)

ground_truth_by_id = {
    item["case_id"]: item
    for item in ground_truth
}


# ============================================================
# Load embeddings
# ============================================================

with open("wifi_embeddings.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Loaded {len(test_cases)} test cases.")
print(f"Loaded {len(data)} embedded chunks.")


# ============================================================
# Exclude metadata sections
# ============================================================

excluded_sections = {
    "What was retained",
    "What was removed",
    "Recommended RAG chunk structure",
    "Source map",
    "Research-use note",
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
# Load embedding model
# ============================================================

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

document_embeddings = np.array(
    [item["embedding"] for item in retrieval_data]
)


# ============================================================
# Retrieval evaluation
# ============================================================

results = []

hit_at_1 = 0
hit_at_3 = 0


for case in test_cases:

    case_id = case["case_id"]
    query = case["user_problem"]

    print("\n" + "=" * 70)
    print(f"Case: {case_id}")
    print(f"Query: {query}")

    # --------------------------------------------------------
    # Query embedding
    # --------------------------------------------------------

    query_embedding = model.encode_query(
        query,
        normalize_embeddings=True
    )

    # --------------------------------------------------------
    # Similarity
    # --------------------------------------------------------

    scores = document_embeddings @ query_embedding

    top_k = min(3, len(retrieval_data))

    top_indices = np.argsort(scores)[::-1][:top_k]

    retrieved_sections = []

    for rank, index in enumerate(top_indices, start=1):

        item = retrieval_data[index]

        retrieved_sections.append({
            "rank": rank,
            "chunk_id": item["chunk_id"],
            "section": item["section"],
            "similarity": float(scores[index])
        })

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    gt = ground_truth_by_id[case_id]

    expected_evidence = gt.get(
        "expected_evidence",
        []
    )

    expected_concepts = gt.get(
        "expected_concepts",
        []
    )

    retrieved_text = " ".join(
        item["section"].lower()
        for item in retrieved_sections
    )

    # --------------------------------------------------------
    # Simple evidence/concept matching
    # --------------------------------------------------------

    evidence_matches = []

    for evidence in expected_evidence:

        if evidence.lower() in retrieved_text:

            evidence_matches.append(evidence)

    concept_matches = []

    for concept in expected_concepts:

        if concept.lower() in retrieved_text:

            concept_matches.append(concept)

    # --------------------------------------------------------
    # Hit@1
    # --------------------------------------------------------

    rank1_section = retrieved_sections[0]["section"].lower()

    hit1 = any(
        evidence.lower() in rank1_section
        or rank1_section in evidence.lower()
        for evidence in expected_evidence
    )

    # --------------------------------------------------------
    # Hit@3
    # --------------------------------------------------------

    hit3 = len(evidence_matches) > 0

    if hit1:
        hit_at_1 += 1

    if hit3:
        hit_at_3 += 1

    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    result = {
        "case_id": case_id,
        "user_problem": query,
        "retrieved": retrieved_sections,
        "expected_evidence": expected_evidence,
        "expected_concepts": expected_concepts,
        "evidence_matches": evidence_matches,
        "concept_matches": concept_matches,
        "hit_at_1": hit1,
        "hit_at_3": hit3
    }

    results.append(result)

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print("\nTop 3:")

    for item in retrieved_sections:

        print(
            f"{item['rank']}. "
            f"{item['section']} "
            f"({item['similarity']:.4f})"
        )


# ============================================================
# Summary
# ============================================================

n = len(test_cases)

hit1_rate = hit_at_1 / n
hit3_rate = hit_at_3 / n

print("\n" + "=" * 70)
print("RETRIEVAL EVALUATION")
print("=" * 70)

print(f"Number of cases: {n}")
print(f"Hit@1: {hit_at_1}/{n} = {hit1_rate:.2%}")
print(f"Hit@3: {hit_at_3}/{n} = {hit3_rate:.2%}")


# ============================================================
# Save results
# ============================================================

output = {
    "num_cases": n,
    "hit_at_1": hit1_rate,
    "hit_at_3": hit3_rate,
    "results": results
}

with open(
    "retrieval_results.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output,
        f,
        ensure_ascii=False,
        indent=2
    )

print("\nSaved to: retrieval_results.json")