import json

with open("wifi_test_cases.json", "r", encoding="utf-8") as f:
    test_cases = json.load(f)

with open("wifi_ground_truth.json", "r", encoding="utf-8") as f:
    ground_truth = json.load(f)

# Match ground truth by case_id
ground_truth_map = {
    item["case_id"]: item
    for item in ground_truth
}

training_data = []

for case in test_cases:
    case_id = case["case_id"]

    if case_id not in ground_truth_map:
        print(f"Warning: no ground truth for {case_id}")
        continue

    truth = ground_truth_map[case_id]

    sample = {
        "instruction": "Diagnose the following Windows Wi-Fi troubleshooting case.",
        "input": (
            f"User problem: {case['user_problem']}\n"
            f"Environment: {case['environment']}"
        ),
        "output": {
            "expected_evidence": truth["expected_evidence"],
            "expected_concepts": truth["expected_concepts"]
        }
    }

    training_data.append(sample)

with open("lora_dataset.json", "w", encoding="utf-8") as f:
    json.dump(training_data, f, indent=2, ensure_ascii=False)

print("Created lora_dataset.json")
print("Training samples:", len(training_data))

print("\nFirst training sample:")
print(json.dumps(training_data[0], indent=2, ensure_ascii=False))