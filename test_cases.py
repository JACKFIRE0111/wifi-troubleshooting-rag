import json

# Load the test cases
with open("wifi_test_cases.json", "r", encoding="utf-8") as file:
    cases = json.load(file)

print(f"Loaded {len(cases)} test cases.\n")

# Display each case
for case in cases:
    print("=" * 60)
    print(f"Case ID: {case['case_id']}")
    print(f"Difficulty: {case['difficulty']}")
    print(f"Environment: {case['environment']}")
    print(f"User Problem: {case['user_problem']}")
    print()
