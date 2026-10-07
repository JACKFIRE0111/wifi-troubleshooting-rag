import json
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "gpt-oss:120b-cloud"
OUTPUT_FILE = "baseline_results.json"


SYSTEM_PROMPT = """
You are a Windows Wi-Fi troubleshooting assistant.

Your task is to generate a practical troubleshooting procedure
for the user's Wi-Fi problem.

IMPORTANT RULES:

1. Use your general technical knowledge to reason about the problem.
2. Do not mention retrieval.
3. Do not mention hidden ground truth.
4. Give ordered troubleshooting steps.
5. Start with the least disruptive diagnostic checks.
6. Only recommend disruptive actions when appropriate.
7. If the root cause cannot be established, say that further
   diagnosis is required.
8. Keep the answer focused on the reported problem.

Format:

Diagnosis:
- Briefly state what the problem may indicate.

Troubleshooting steps:
1. ...
2. ...
3. ...

Expected result:
- Explain what the user should observe.

If unresolved:
- State the next diagnostic direction.
"""


with open("wifi_test_cases.json", "r", encoding="utf-8") as f:
    cases = json.load(f)

results = []

print(f"Loaded {len(cases)} test cases.\n")

for i, case in enumerate(cases, 1):

    print("=" * 60)
    print(f"CASE {i}/{len(cases)}: {case['case_id']}")
    print(f"User problem: {case['user_problem']}")
    print("Generating with Qwen3:8B...")

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": case["user_problem"]
                }
            ],
            "stream": False
        },
        timeout=600
    )

    response.raise_for_status()

    answer = response.json()["message"]["content"]

    print("\nLLM answer:")
    print(answer)

    results.append({
        "case_id": case["case_id"],
        "user_problem": case["user_problem"],
        "model": MODEL_NAME,
        "condition": "No-RAG",
        "answer": answer
    })


with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 60)
print("BASELINE GENERATION COMPLETE")
print(f"Successfully generated: {len(results)}/{len(cases)} cases")
print(f"Saved to: {OUTPUT_FILE}")