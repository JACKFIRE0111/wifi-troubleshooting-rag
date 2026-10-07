import json
import statistics


INPUT_FILE = "groundedness_evaluation.json"
OUTPUT_FILE = "threshold_sensitivity.json"

THRESHOLDS = [0.40, 0.45, 0.50, 0.55]


with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)


cases = data["cases"]


def support_rate(claims, threshold):
    if not claims:
        return 0.0

    supported = sum(
        1
        for claim in claims
        if claim["best_similarity"] >= threshold
    )

    return supported / len(claims)


results = []


for threshold in THRESHOLDS:

    baseline_rates = []
    rag_rates = []

    for case in cases:

        baseline_rate = support_rate(
            case["baseline_claims"],
            threshold
        )

        rag_rate = support_rate(
            case["rag_claims"],
            threshold
        )

        baseline_rates.append(baseline_rate)
        rag_rates.append(rag_rate)

    baseline_mean = statistics.mean(baseline_rates)
    rag_mean = statistics.mean(rag_rates)

    results.append({
        "threshold": threshold,
        "baseline_support": baseline_mean,
        "rag_support": rag_mean,
        "improvement": rag_mean - baseline_mean
    })


print("=" * 70)
print("THRESHOLD SENSITIVITY ANALYSIS")
print("=" * 70)
print()

print(
    f"{'Threshold':<12}"
    f"{'No-RAG':<15}"
    f"{'RAG':<15}"
    f"{'Improvement':<15}"
)

print("-" * 57)

for row in results:

    print(
        f"{row['threshold']:<12.2f}"
        f"{row['baseline_support']:<15.3f}"
        f"{row['rag_support']:<15.3f}"
        f"{row['improvement']:+.3f}"
    )


rag_wins = sum(
    1
    for row in results
    if row["rag_support"] > row["baseline_support"]
)

print()
print(
    f"RAG higher than No-RAG at "
    f"{rag_wins}/{len(results)} thresholds."
)


output = {
    "input": INPUT_FILE,
    "thresholds": THRESHOLDS,
    "results": results,
    "interpretation_note": (
        "This sensitivity analysis reuses the per-claim semantic "
        "similarities already computed by groundedness_evaluation.py. "
        "It tests whether the direction of the RAG-vs-No-RAG result "
        "changes when the support threshold changes."
    )
}


with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(
        output,
        f,
        ensure_ascii=False,
        indent=2
    )


print()
print("=" * 70)
print("Sensitivity analysis complete.")
print(f"Saved to: {OUTPUT_FILE}")
print("=" * 70)
