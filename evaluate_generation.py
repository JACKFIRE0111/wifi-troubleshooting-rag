import json
import re
import statistics


BASELINE_FILE = "baseline_results.json"
RAG_FILE = "rag_results.json"
GROUND_TRUTH_FILE = "wifi_ground_truth.json"


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text


def phrase_present(answer, phrase):
    answer = normalize(answer)
    phrase = normalize(phrase)
    return phrase in answer


def evaluate_answer(answer, ground_truth):
    """
    Automatic proxy evaluation.

    Evidence coverage:
        How many expected evidence items are mentioned?

    Concept coverage:
        How many expected troubleshooting concepts are mentioned?
    """

    expected_evidence = ground_truth.get("expected_evidence", [])
    expected_concepts = ground_truth.get("expected_concepts", [])

    evidence_hits = []
    concept_hits = []

    for item in expected_evidence:
        hit = phrase_present(answer, item)
        evidence_hits.append(hit)

    for item in expected_concepts:
        hit = phrase_present(answer, item)
        concept_hits.append(hit)

    evidence_score = (
        sum(evidence_hits) / len(evidence_hits)
        if evidence_hits else 0
    )

    concept_score = (
        sum(concept_hits) / len(concept_hits)
        if concept_hits else 0
    )

    return {
        "evidence_score": evidence_score,
        "concept_score": concept_score,
        "evidence_hits": evidence_hits,
        "concept_hits": concept_hits,
    }


with open(BASELINE_FILE, "r", encoding="utf-8") as f:
    baseline = json.load(f)

with open(RAG_FILE, "r", encoding="utf-8") as f:
    rag = json.load(f)

with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
    ground_truth = json.load(f)


# Ground truth may be a list or a dictionary keyed by case ID.
if isinstance(ground_truth, list):
    gt_by_id = {
        item["case_id"]: item
        for item in ground_truth
    }
else:
    gt_by_id = ground_truth


results = []


print("=" * 70)
print("RAG vs NO-RAG EVALUATION")
print("=" * 70)


for base_item, rag_item in zip(baseline, rag):

    case_id = base_item["case_id"]

    gt = gt_by_id[case_id]

    baseline_answer = base_item["answer"]
    rag_answer = rag_item["llm_answer"]

    baseline_scores = evaluate_answer(
        baseline_answer,
        gt
    )

    rag_scores = evaluate_answer(
        rag_answer,
        gt
    )

    results.append({
        "case_id": case_id,

        "baseline_evidence": baseline_scores["evidence_score"],
        "rag_evidence": rag_scores["evidence_score"],

        "baseline_concepts": baseline_scores["concept_score"],
        "rag_concepts": rag_scores["concept_score"],
    })


print()


# ---------------------------------------------------------
# CASE-BY-CASE RESULTS
# ---------------------------------------------------------

for result in results:

    print("-" * 70)
    print(result["case_id"])

    print(
        f"Evidence coverage: "
        f"No-RAG={result['baseline_evidence']:.2f} | "
        f"RAG={result['rag_evidence']:.2f}"
    )

    print(
        f"Concept coverage:  "
        f"No-RAG={result['baseline_concepts']:.2f} | "
        f"RAG={result['rag_concepts']:.2f}"
    )


# ---------------------------------------------------------
# OVERALL RESULTS
# ---------------------------------------------------------

baseline_evidence = [
    r["baseline_evidence"]
    for r in results
]

rag_evidence = [
    r["rag_evidence"]
    for r in results
]

baseline_concepts = [
    r["baseline_concepts"]
    for r in results
]

rag_concepts = [
    r["rag_concepts"]
    for r in results
]


print()
print("=" * 70)
print("OVERALL RESULTS")
print("=" * 70)

print(
    f"No-RAG Evidence Coverage: "
    f"{statistics.mean(baseline_evidence):.3f}"
)

print(
    f"RAG Evidence Coverage:    "
    f"{statistics.mean(rag_evidence):.3f}"
)

print()

print(
    f"No-RAG Concept Coverage:  "
    f"{statistics.mean(baseline_concepts):.3f}"
)

print(
    f"RAG Concept Coverage:     "
    f"{statistics.mean(rag_concepts):.3f}"
)


evidence_improvement = (
    statistics.mean(rag_evidence)
    - statistics.mean(baseline_evidence)
)

concept_improvement = (
    statistics.mean(rag_concepts)
    - statistics.mean(baseline_concepts)
)


print()
print(
    f"Evidence improvement: "
    f"{evidence_improvement:+.3f}"
)

print(
    f"Concept improvement:  "
    f"{concept_improvement:+.3f}"
)


# ---------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------

with open(
    "generation_evaluation.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        ensure_ascii=False,
        indent=2
    )


print()
print("=" * 70)
print("Evaluation complete.")
print("Saved to: generation_evaluation.json")
print("=" * 70)