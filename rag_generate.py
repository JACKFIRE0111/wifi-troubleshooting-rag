import json
import numpy as np
import urllib.request
import urllib.error

from sentence_transformers import SentenceTransformer


# ============================================================
# Configuration
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "gpt-oss:120b-cloud"

TOP_K = 3

OUTPUT_FILE = "rag_results.json"


# ============================================================
# Load test cases
# ============================================================

with open("wifi_test_cases.json", "r", encoding="utf-8") as f:
    test_cases = json.load(f)


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
# Retrieval function
# ============================================================

def retrieve_evidence(query):

    query_embedding = model.encode_query(
        query,
        normalize_embeddings=True
    )

    scores = document_embeddings @ query_embedding

    top_indices = np.argsort(scores)[::-1][:TOP_K]

    results = []

    for rank, index in enumerate(top_indices, start=1):

        item = retrieval_data[index]

        results.append({
            "rank": rank,
            "chunk_id": item["chunk_id"],
            "section": item["section"],
            "pages": item["pages"],
            "similarity": float(scores[index]),
            "text": item["text"]
        })

    return results


# ============================================================
# Generate answer with Ollama
# ============================================================

def generate_with_ollama(user_problem, evidence):

    evidence_text = ""

    for item in evidence:

        evidence_text += (
            f"\n\n--- Evidence {item['rank']} ---\n"
            f"Section: {item['section']}\n"
            f"Pages: {item['pages']}\n"
            f"Evidence:\n{item['text']}\n"
        )

    system_prompt = """
You are a Windows Wi-Fi troubleshooting assistant.

Your task is to generate a practical troubleshooting procedure
for the user's Wi-Fi problem.

IMPORTANT RULES:

1. Use the provided evidence as the primary source.
2. Do not invent unsupported technical claims.
3. Do not mention the retrieval process.
4. Do not mention hidden ground truth.
5. Give ordered troubleshooting steps.
6. Start with the least disruptive diagnostic checks.
7. Only recommend disruptive actions when supported by the evidence.
8. If the evidence does not establish the root cause, say that
   further diagnosis is required.
9. Keep the answer focused on the reported problem.
10. Do not blindly include every piece of retrieved evidence.

Format:

Diagnosis:
- Briefly state what the evidence suggests.

Troubleshooting steps:
1. ...
2. ...
3. ...

Expected result:
- Explain what the user should observe.

If unresolved:
- State the next diagnostic direction.
"""

    user_prompt = f"""
User Wi-Fi problem:

{user_problem}

Retrieved evidence:

{evidence_text}

Generate the troubleshooting procedure according to the rules.
"""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "stream": False
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    try:

        with urllib.request.urlopen(
            request,
            timeout=300
        ) as response:

            result = json.loads(
                response.read().decode("utf-8")
            )

        return result["message"]["content"]

    except urllib.error.URLError as e:

        print("\nERROR: Cannot connect to Ollama.")
        print("Make sure Ollama is running.")
        print(f"Details: {e}")

        return None

    except Exception as e:

        print(f"\nERROR during generation: {e}")

        return None


# ============================================================
# Run all test cases
# ============================================================

all_results = []

for case_number, case in enumerate(
    test_cases,
    start=1
):

    case_id = case["case_id"]
    user_problem = case["user_problem"]

    print("\n" + "=" * 75)
    print(
        f"CASE {case_number}/{len(test_cases)}: "
        f"{case_id}"
    )
    print("=" * 75)

    print(f"\nUser problem:\n{user_problem}")

    # --------------------------------------------------------
    # Retrieve
    # --------------------------------------------------------

    evidence = retrieve_evidence(
        user_problem
    )

    print("\nRetrieved evidence:")

    for item in evidence:

        print(
            f"{item['rank']}. "
            f"{item['section']} "
            f"(similarity={item['similarity']:.4f})"
        )

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    print("\nGenerating with Qwen3:8B...")

    answer = generate_with_ollama(
        user_problem,
        evidence
    )

    if answer is None:
        print("Generation failed.")
        continue

    print("\nLLM answer:")
    print("-" * 75)
    print(answer)
    print("-" * 75)

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    all_results.append({
        "case_id": case_id,
        "user_problem": user_problem,
        "retrieved_evidence": evidence,
        "llm_answer": answer
    })


# ============================================================
# Save all results
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        all_results,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# Final
# ============================================================

print("\n" + "=" * 75)
print("RAG GENERATION COMPLETE")
print("=" * 75)

print(
    f"Successfully generated: "
    f"{len(all_results)}/{len(test_cases)} cases"
)

print(
    f"Saved to: {OUTPUT_FILE}"
)