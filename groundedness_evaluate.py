import json
import re
import statistics
from sentence_transformers import SentenceTransformer, util


BASELINE_FILE = "baseline_results.json"
RAG_FILE = "rag_results.json"
GROUND_TRUTH_FILE = "wifi_ground_truth.json"
CHUNKS_FILE = "wifi_chunks.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Initial pilot threshold.
# This is a proxy, not a human-validated final threshold.
SUPPORT_THRESHOLD = 0.45


def split_into_units(text):
    """Split an answer into small claim/step-like units."""
    text = re.sub(r"\r\n?", "\n", text)

    units = []

    for line in text.split("\n"):
        line = line.strip()

        if not line:
            continue

        line = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s*", "", line).strip()

        if line:
            units.append(line)

    expanded = []

    for unit in units:
        parts = re.split(r"(?<=[.!?])\s+", unit)
        expanded.extend(
            [p.strip() for p in parts if p.strip()]
        )

    return expanded


def get_text(item):
    """Support common retrieved-evidence representations."""
    if isinstance(item, str):
        return item

    if isinstance(item, dict):
        for key in ("text", "content", "evidence", "chunk_text"):
            if key in item and isinstance(item[key], str):
                return item[key]

    return str(item)


def max_support(answer_units, source_texts, model):
    """
    For each answer unit, calculate its best semantic similarity
    against the supplied source texts.
    """
    if not answer_units:
        return 0.0, []

    source_texts = [
        get_text(x).strip()
        for x in source_texts
        if get_text(x).strip()
    ]

    if not source_texts:
        return 0.0, [
            {
                "claim": unit,
                "best_similarity": 0.0,
                "supported": False,
                "matched_source": ""
            }
            for unit in answer_units
        ]

    answer_embeddings = model.encode(
        answer_units,
        normalize_embeddings=True,
        convert_to_tensor=True
    )

    source_embeddings = model.encode(
        source_texts,
        normalize_embeddings=True,
        convert_to_tensor=True
    )

    similarities = util.cos_sim(
        answer_embeddings,
        source_embeddings
    )

    details = []
    supported_count = 0

    for i, claim in enumerate(answer_units):
        best_index = int(similarities[i].argmax())
        best_score = float(similarities[i][best_index])
        supported = best_score >= SUPPORT_THRESHOLD

        if supported:
            supported_count += 1

        details.append({
            "claim": claim,
            "best_similarity": round(best_score, 4),
            "supported": supported,
            "matched_source": source_texts[best_index]
        })

    support_rate = supported_count / len(answer_units)

    return support_rate, details


# ---------------------------------------------------------
# LOAD FILES
# ---------------------------------------------------------

with open(BASELINE_FILE, "r", encoding="utf-8") as f:
    baseline = json.load(f)

with open(RAG_FILE, "r", encoding="utf-8") as f:
    rag = json.load(f)

with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
    ground_truth = json.load(f)

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)


if isinstance(ground_truth, list):
    gt_by_id = {
        item["case_id"]: item
        for item in ground_truth
    }
else:
    gt_by_id = ground_truth


baseline_by_id = {
    item["case_id"]: item
    for item in baseline
}

rag_by_id = {
    item["case_id"]: item
    for item in rag
}


# ---------------------------------------------------------
# PREPARE FULL CORPUS
# ---------------------------------------------------------

full_corpus = [
    chunk["text"]
    for chunk in chunks
    if isinstance(chunk, dict) and "text" in chunk
]

print("=" * 70)
print("GROUNDEDNESS / UNSUPPORTED CLAIMS EVALUATION")
print("=" * 70)
print()
print(f"Embedding model: {MODEL_NAME}")
print(f"Support threshold: {SUPPORT_THRESHOLD}")
print(f"Full corpus chunks: {len(full_corpus)}")
print()

model = SentenceTransformer(
    MODEL_NAME,
    device="cpu"
)

print("Embedding model loaded.")
print()


# ---------------------------------------------------------
# EVALUATE
# ---------------------------------------------------------

results = []

for case_id in sorted(gt_by_id.keys()):

    if case_id not in baseline_by_id:
        print(f"WARNING: Missing baseline result for {case_id}")
        continue

    if case_id not in rag_by_id:
        print(f"WARNING: Missing RAG result for {case_id}")
        continue

    baseline_answer = baseline_by_id[case_id]["answer"]
    rag_answer = rag_by_id[case_id]["llm_answer"]

    baseline_units = split_into_units(baseline_answer)
    rag_units = split_into_units(rag_answer)

    # Both conditions use the same full source corpus.
    baseline_support_rate, baseline_details = max_support(
        baseline_units,
        full_corpus,
        model
    )

    rag_support_rate, rag_details = max_support(
        rag_units,
        full_corpus,
        model
    )

    # Additional RAG-specific measure: support from the evidence
    # actually retrieved for that case.
    retrieved_evidence = rag_by_id[case_id].get(
        "retrieved_evidence",
        []
    )

    rag_retrieval_support_rate, rag_retrieval_details = max_support(
        rag_units,
        retrieved_evidence,
        model
    )

    results.append({
        "case_id": case_id,

        "baseline_support_rate": baseline_support_rate,
        "rag_support_rate": rag_support_rate,

        "baseline_unsupported_rate":
            1.0 - baseline_support_rate,

        "rag_unsupported_rate":
            1.0 - rag_support_rate,

        "rag_retrieval_support_rate":
            rag_retrieval_support_rate,

        "rag_retrieval_unsupported_rate":
            1.0 - rag_retrieval_support_rate,

        "baseline_claims": baseline_details,
        "rag_claims": rag_details,
        "rag_retrieval_claims": rag_retrieval_details
    })


# ---------------------------------------------------------
# CASE RESULTS
# ---------------------------------------------------------

for result in results:

    print("-" * 70)
    print(result["case_id"])

    print(
        f"Corpus support rate: "
        f"No-RAG={result['baseline_support_rate']:.2f} | "
        f"RAG={result['rag_support_rate']:.2f}"
    )

    print(
        f"Unsupported claim rate: "
        f"No-RAG={result['baseline_unsupported_rate']:.2f} | "
        f"RAG={result['rag_unsupported_rate']:.2f}"
    )

    print(
        f"RAG retrieved-evidence support: "
        f"{result['rag_retrieval_support_rate']:.2f}"
    )


# ---------------------------------------------------------
# OVERALL
# ---------------------------------------------------------

def mean_or_zero(values):
    return statistics.mean(values) if values else 0.0


baseline_support = mean_or_zero([
    r["baseline_support_rate"]
    for r in results
])

rag_support = mean_or_zero([
    r["rag_support_rate"]
    for r in results
])

baseline_unsupported = mean_or_zero([
    r["baseline_unsupported_rate"]
    for r in results
])

rag_unsupported = mean_or_zero([
    r["rag_unsupported_rate"]
    for r in results
])

rag_retrieval_support = mean_or_zero([
    r["rag_retrieval_support_rate"]
    for r in results
])

rag_retrieval_unsupported = mean_or_zero([
    r["rag_retrieval_unsupported_rate"]
    for r in results
])


print()
print("=" * 70)
print("OVERALL GROUNDEDNESS RESULTS")
print("=" * 70)

print(
    f"No-RAG corpus support rate: "
    f"{baseline_support:.3f}"
)

print(
    f"RAG corpus support rate:    "
    f"{rag_support:.3f}"
)

print()

print(
    f"No-RAG unsupported claim rate: "
    f"{baseline_unsupported:.3f}"
)

print(
    f"RAG unsupported claim rate:    "
    f"{rag_unsupported:.3f}"
)

print()

print(
    f"RAG retrieved-evidence support: "
    f"{rag_retrieval_support:.3f}"
)

print(
    f"RAG retrieved-evidence unsupported: "
    f"{rag_retrieval_unsupported:.3f}"
)

print()

print(
    f"Corpus support improvement: "
    f"{rag_support - baseline_support:+.3f}"
)

print(
    f"Unsupported claim change: "
    f"{rag_unsupported - baseline_unsupported:+.3f}"
)


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

output = {
    "metric": "semantic_groundedness_proxy",
    "embedding_model": MODEL_NAME,
    "support_threshold": SUPPORT_THRESHOLD,
    "n_cases": len(results),

    "method_note": (
        "Both No-RAG and RAG answers are compared against the "
        "same full source corpus. An additional RAG-specific "
        "score measures support from the evidence actually "
        "retrieved for each case. These are semantic similarity "
        "proxies and require human validation before being "
        "treated as final labels."
    ),

    "overall": {
        "baseline_corpus_support": baseline_support,
        "rag_corpus_support": rag_support,

        "baseline_unsupported_rate":
            baseline_unsupported,

        "rag_unsupported_rate":
            rag_unsupported,

        "rag_retrieved_evidence_support":
            rag_retrieval_support,

        "rag_retrieved_evidence_unsupported":
            rag_retrieval_unsupported,

        "support_improvement":
            rag_support - baseline_support,

        "unsupported_claim_change":
            rag_unsupported - baseline_unsupported
    },

    "cases": results
}


with open(
    "groundedness_evaluation.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        output,
        f,
        ensure_ascii=False,
        indent=2
    )


print()
print("=" * 70)
print("Groundedness evaluation complete.")
print("Saved to: groundedness_evaluation.json")
print("=" * 70)
