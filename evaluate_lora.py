import os
import json
import glob
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

from peft import PeftModel


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = r"E:\Murl LLM Research"

# LoRA adapter
LORA_PATH = os.path.join(
    PROJECT_DIR,
    "qwen_wifi_lora"
)

# Ground-truth file
GROUND_TRUTH_FILE = os.path.join(
    PROJECT_DIR,
    "wifi_ground_truth.json"
)

# Actual Wi-Fi test cases
TEST_CASE_FILE = os.path.join(
    PROJECT_DIR,
    "wifi_test_cases.json"
)

# Local Hugging Face cache
HF_CACHE = os.path.expandvars(
    r"%USERPROFILE%\.cache\huggingface\hub"
)

MODEL_CACHE = os.path.join(
    HF_CACHE,
    "models--Qwen--Qwen2.5-7B-Instruct"
)


# ============================================================
# FIND LOCAL MODEL SNAPSHOT
# ============================================================

print("=" * 70)
print("FINDING LOCAL QWEN MODEL")
print("=" * 70)

SNAPSHOT_DIR = os.path.join(
    MODEL_CACHE,
    "snapshots"
)

if not os.path.isdir(SNAPSHOT_DIR):
    raise FileNotFoundError(
        f"Qwen model cache not found:\n{MODEL_CACHE}"
    )

snapshots = [
    os.path.join(SNAPSHOT_DIR, x)
    for x in os.listdir(SNAPSHOT_DIR)
    if os.path.isdir(os.path.join(SNAPSHOT_DIR, x))
]

if not snapshots:
    raise FileNotFoundError(
        f"No model snapshot found in:\n{SNAPSHOT_DIR}"
    )

# Use the first local snapshot
MODEL_PATH = snapshots[0]

print("Model path:")
print(MODEL_PATH)


# ============================================================
# CHECK LORA
# ============================================================

print("\n" + "=" * 70)
print("CHECKING LORA ADAPTER")
print("=" * 70)

if not os.path.isdir(LORA_PATH):
    raise FileNotFoundError(
        f"LoRA adapter not found:\n{LORA_PATH}"
    )

print("LoRA path:")
print(LORA_PATH)


# ============================================================
# CHECK DATA FILES
# ============================================================

print("\n" + "=" * 70)
print("CHECKING DATA FILES")
print("=" * 70)

if not os.path.isfile(GROUND_TRUTH_FILE):
    raise FileNotFoundError(
        f"Ground-truth file not found:\n{GROUND_TRUTH_FILE}"
    )

if not os.path.isfile(TEST_CASE_FILE):
    raise FileNotFoundError(
        f"Test-case file not found:\n{TEST_CASE_FILE}"
    )

print("Ground truth:")
print(GROUND_TRUTH_FILE)

print("\nTest cases:")
print(TEST_CASE_FILE)


# ============================================================
# LOAD JSON FILE
# ============================================================

def load_json_file(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("\n" + "=" * 70)
print("LOADING GROUND TRUTH")
print("=" * 70)

ground_truth_data = load_json_file(
    GROUND_TRUTH_FILE
)

if isinstance(ground_truth_data, dict):

    if "data" in ground_truth_data:
        ground_truth_data = ground_truth_data["data"]

    elif "train" in ground_truth_data:
        ground_truth_data = ground_truth_data["train"]

    else:
        ground_truth_data = [ground_truth_data]


if not isinstance(
    ground_truth_data,
    list
):
    raise ValueError(
        "Ground truth dataset must be a list."
    )

print(
    "Number of ground-truth samples:",
    len(ground_truth_data)
)


# ============================================================
# LOAD TEST CASES
# ============================================================

print("\n" + "=" * 70)
print("LOADING TEST CASES")
print("=" * 70)

test_case_data = load_json_file(
    TEST_CASE_FILE
)

if isinstance(test_case_data, dict):

    if "data" in test_case_data:
        test_case_data = test_case_data["data"]

    elif "train" in test_case_data:
        test_case_data = test_case_data["train"]

    else:
        test_case_data = [test_case_data]


if not isinstance(
    test_case_data,
    list
):
    raise ValueError(
        "Test-case dataset must be a list."
    )

print(
    "Number of test cases:",
    len(test_case_data)
)


# ============================================================
# BUILD CASE LOOKUP
# ============================================================

case_lookup = {}

for case in test_case_data:

    if not isinstance(case, dict):
        continue

    case_id = case.get("case_id")

    if case_id:
        case_lookup[case_id] = case


print(
    "Indexed case descriptions:",
    len(case_lookup)
)


# ============================================================
# VERIFY DATASET STRUCTURE
# ============================================================

print("\n" + "=" * 70)
print("VERIFYING DATASET STRUCTURE")
print("=" * 70)

if len(ground_truth_data) > 0:

    print(
        "Ground-truth fields:"
    )

    print(
        list(
            ground_truth_data[0].keys()
        )
    )


if len(test_case_data) > 0:

    print(
        "\nTest-case fields:"
    )

    print(
        list(
            test_case_data[0].keys()
        )
    )


# ============================================================
# VERIFY CASE IDs
# ============================================================

missing_cases = []

for example in ground_truth_data:

    case_id = example.get(
        "case_id"
    )

    if case_id not in case_lookup:

        missing_cases.append(
            case_id
        )


if missing_cases:

    raise ValueError(
        "The following case IDs exist in ground truth "
        "but not in wifi_test_cases.json:\n"
        + str(missing_cases)
    )


print("\nAll case IDs matched successfully.")


# ============================================================
# GET CASE DESCRIPTION
# ============================================================

def get_case_text(example):

    case_id = example.get(
        "case_id"
    )

    if case_id not in case_lookup:

        raise ValueError(
            f"Could not find case_id "
            f"{case_id} in wifi_test_cases.json"
        )

    case = case_lookup[case_id]

    user_problem = case.get(
        "user_problem",
        ""
    )

    environment = case.get(
        "environment",
        ""
    )

    return f"""
User problem:
{user_problem}

Environment:
{environment}
""".strip()


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\n" + "=" * 70)
print("LOADING TOKENIZER")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer loaded.")


# ============================================================
# LOAD BASE MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING BASE MODEL")
print("=" * 70)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    local_files_only=True
)

base_model.eval()

print(
    "Base Qwen2.5-7B model loaded."
)


# ============================================================
# LOAD LORA MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING LORA ADAPTER")
print("=" * 70)

lora_model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH
)

lora_model.eval()

print("LoRA adapter loaded.")

try:
    print(
        "Active adapter:",
        lora_model.active_adapter
    )
except Exception:
    pass


# ============================================================
# GENERATION FUNCTION
# ============================================================

def generate_response(
    model,
    prompt
):

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    # Move inputs to the same device
    # used by the model's input embedding layer
    try:

        model_device = next(
            model.parameters()
        ).device

    except StopIteration:

        model_device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    inputs = {
        key: value.to(model_device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    input_length = inputs[
        "input_ids"
    ].shape[1]

    generated_tokens = outputs[
        0,
        input_length:
    ]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return response.strip()


# ============================================================
# KEYWORD SCORE
# ============================================================

def keyword_score(
    output,
    expected_items
):

    if not expected_items:
        return 0.0

    output_lower = output.lower()

    matched = 0

    for item in expected_items:

        item_lower = str(
            item
        ).lower()

        if item_lower in output_lower:

            matched += 1

    return (
        matched /
        len(expected_items)
    )


# ============================================================
# BUILD PROMPT
# ============================================================

def build_prompt(example):

    case_id = example.get(
        "case_id",
        "UNKNOWN"
    )

    expected_evidence = example.get(
        "expected_evidence",
        []
    )

    expected_concepts = example.get(
        "expected_concepts",
        []
    )

    case_text = get_case_text(
        example
    )

    prompt = f"""
You are a technical Wi-Fi troubleshooting assistant.

Case ID:
{case_id}

Case description:
{case_text}

Analyze this Wi-Fi troubleshooting case.

Identify:

1. Important evidence that should be checked.
2. Relevant technical concepts.
3. A concise explanation of what the technician should investigate.

Provide a concise technical response.
""".strip()

    return (
        prompt,
        expected_evidence,
        expected_concepts
    )


# ============================================================
# RUN EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("STARTING EVALUATION")
print("=" * 70)

results = []


for index, example in enumerate(
    ground_truth_data
):

    print("\n")
    print("-" * 70)

    print(
        f"SAMPLE {index + 1}/"
        f"{len(ground_truth_data)}"
    )

    print("-" * 70)

    case_id = example.get(
        "case_id",
        f"sample_{index}"
    )

    expected_evidence = example.get(
        "expected_evidence",
        []
    )

    expected_concepts = example.get(
        "expected_concepts",
        []
    )

    # --------------------------------------------------------
    # BUILD PROMPT
    # --------------------------------------------------------

    prompt, _, _ = build_prompt(
        example
    )

    print(
        "\nCase ID:",
        case_id
    )

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    print(
        "\nRunning BASELINE..."
    )

    baseline_output = generate_response(
        base_model,
        prompt
    )

    # --------------------------------------------------------
    # LORA
    # --------------------------------------------------------

    print(
        "Running LORA..."
    )

    lora_output = generate_response(
        lora_model,
        prompt
    )

    # --------------------------------------------------------
    # SCORES
    # --------------------------------------------------------

    baseline_evidence_score = keyword_score(
        baseline_output,
        expected_evidence
    )

    lora_evidence_score = keyword_score(
        lora_output,
        expected_evidence
    )

    baseline_concept_score = keyword_score(
        baseline_output,
        expected_concepts
    )

    lora_concept_score = keyword_score(
        lora_output,
        expected_concepts
    )

    # --------------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------------

    result = {

        "case_id":
            case_id,

        "baseline_output":
            baseline_output,

        "lora_output":
            lora_output,

        "baseline_evidence_score":
            baseline_evidence_score,

        "lora_evidence_score":
            lora_evidence_score,

        "baseline_concept_score":
            baseline_concept_score,

        "lora_concept_score":
            lora_concept_score
    }

    results.append(
        result
    )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print(
        "\nBASELINE OUTPUT:"
    )

    print(
        baseline_output
    )

    print(
        "\nLORA OUTPUT:"
    )

    print(
        lora_output
    )

    print(
        "\nSCORES:"
    )

    print(
        "Baseline evidence:",
        round(
            baseline_evidence_score,
            3
        )
    )

    print(
        "LoRA evidence:",
        round(
            lora_evidence_score,
            3
        )
    )

    print(
        "Baseline concepts:",
        round(
            baseline_concept_score,
            3
        )
    )

    print(
        "LoRA concepts:",
        round(
            lora_concept_score,
            3
        )
    )


# ============================================================
# AVERAGES
# ============================================================

def average(key):

    if not results:
        return 0.0

    return sum(
        r[key]
        for r in results
    ) / len(results)


summary = {

    "number_of_samples":
        len(results),

    "baseline_evidence":
        average(
            "baseline_evidence_score"
        ),

    "lora_evidence":
        average(
            "lora_evidence_score"
        ),

    "baseline_concepts":
        average(
            "baseline_concept_score"
        ),

    "lora_concepts":
        average(
            "lora_concept_score"
        )
}


# ============================================================
# IMPROVEMENT
# ============================================================

evidence_improvement = (
    summary["lora_evidence"]
    -
    summary["baseline_evidence"]
)

concept_improvement = (
    summary["lora_concepts"]
    -
    summary["baseline_concepts"]
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(
    "Number of samples:",
    summary["number_of_samples"]
)

print(
    "\nBaseline evidence:",
    round(
        summary["baseline_evidence"],
        3
    )
)

print(
    "LoRA evidence:",
    round(
        summary["lora_evidence"],
        3
    )
)

print(
    "\nBaseline concepts:",
    round(
        summary["baseline_concepts"],
        3
    )
)

print(
    "LoRA concepts:",
    round(
        summary["lora_concepts"],
        3
    )
)


# ============================================================
# IMPROVEMENT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("IMPROVEMENT")
print("=" * 70)

print(
    "Evidence improvement:",
    round(
        evidence_improvement,
        3
    )
)

print(
    "Concept improvement:",
    round(
        concept_improvement,
        3
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

output_file = os.path.join(
    PROJECT_DIR,
    "evaluation_results.json"
)

with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "summary":
                summary,

            "results":
                results
        },
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print(
    "Results saved to:"
)

print(
    output_file
)