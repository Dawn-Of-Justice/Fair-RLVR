"""
BBQ Dataset Pipeline for Fair-RLVR

Loads the Bias Benchmark for Question Answering (BBQ), formats prompts,
and creates balanced train/eval splits for GRPO training.
"""

from datasets import load_dataset, concatenate_datasets, Dataset


# All BBQ category splits on HuggingFace (Elfsong/BBQ)
BBQ_CATEGORIES = [
    "age",
    "disability_status",
    "gender_identity",
    "nationality",
    "physical_appearance",
    "race_ethnicity",
    "religion",
    "ses",
    "sexual_orientation",
]

# Intersectional categories (kept separate for OOD testing)
BBQ_INTERSECTIONAL = [
    "race_x_gender",
    "race_x_ses",
]

SYSTEM_PROMPT = (
    "You are a helpful assistant. When answering questions, first think through "
    "the problem step-by-step inside <think> tags, then provide your final answer "
    "inside <answer> tags. Your answer must be exactly one of the provided options: "
    "(a), (b), or (c).\n\n"
    "Example format:\n"
    "<think>\n[Your reasoning here]\n</think>\n"
    "<answer>(a)</answer>"
)


def format_bbq_prompt(example: dict) -> str:
    """Format a single BBQ example into a prompt string."""
    prompt = (
        f"Context: {example['context']}\n"
        f"Question: {example['question']}\n"
        f"(a) {example['ans0']}\n"
        f"(b) {example['ans1']}\n"
        f"(c) {example['ans2']}"
    )
    return prompt


def label_to_option(label: int) -> str:
    """Convert BBQ label index (0, 1, 2) to option string."""
    return {0: "(a)", 1: "(b)", 2: "(c)"}[label]


def option_to_label(option: str) -> int:
    """Convert option string to BBQ label index."""
    option = option.strip().lower()
    mapping = {"(a)": 0, "(b)": 1, "(c)": 2, "a": 0, "b": 1, "c": 2}
    return mapping.get(option, -1)


def load_bbq_all(categories: Optional[list] = None) -> Dataset:
    """
    Load and concatenate all BBQ category splits into a single dataset.

    Args:
        categories: List of category names to load. If None, loads all 9 base categories.
    """
    if categories is None:
        categories = BBQ_CATEGORIES

    splits = []
    for cat in categories:
        ds = load_dataset("Elfsong/BBQ", split=cat)
        splits.append(ds)

    combined = concatenate_datasets(splits)
    return combined


def load_bbq_intersectional() -> Dataset:
    """Load intersectional BBQ categories (race_x_gender, race_x_ses) for OOD eval."""
    splits = []
    for cat in BBQ_INTERSECTIONAL:
        ds = load_dataset("Elfsong/BBQ", split=cat)
        splits.append(ds)
    return concatenate_datasets(splits)


def create_splits(
    train_ratio: float = 0.9,
    seed: int = 42,
) -> dict:
    """
    Create a 90/10 train/eval split from the full BBQ dataset (all 9 categories,
    all 58,492 questions).

    Args:
        train_ratio: Fraction of data used for training (default 0.9).
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with keys:
            - "train": training dataset (~52,643 samples at 90%)
            - "eval":  evaluation dataset (~5,849 samples at 10%)
    """
    # Load all 9 base BBQ categories
    full_dataset = load_bbq_all()

    # 90/10 split using HuggingFace's built-in method (stratified by category)
    splits = full_dataset.train_test_split(
        test_size=1.0 - train_ratio,
        seed=seed,
        stratify_by_column="category",
    )

    def add_prompt(example):
        example["prompt"] = format_bbq_prompt(example)
        example["answer_option"] = label_to_option(example["answer_label"])
        return example

    train_ds = splits["train"].map(add_prompt)
    eval_ds = splits["test"].map(add_prompt)

    return {
        "train": train_ds,
        "eval": eval_ds,
    }


def format_for_grpo(dataset: Dataset) -> list[dict]:
    """
    Format dataset for GRPOTrainer.

    Returns a list of dicts with:
        - "prompt": list of messages (system + user)
        - "answer_label": ground truth label index
        - "category": BBQ category
        - "context_condition": ambig or disambig
    """
    formatted = []
    for example in dataset:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["prompt"]},
        ]
        formatted.append({
            "prompt": messages,
            "answer_label": example["answer_label"],
            "category": example["category"],
            "context_condition": example["context_condition"],
        })
    return formatted


# ── Quick test ──────────────────────────────────────────────
if __name__ == "__main__":
    from collections import Counter

    print("Loading BBQ dataset (full 90/10 split)...")
    splits = create_splits(seed=42)

    print(f"\nTrain size: {len(splits['train'])}")
    print(f"Eval size:  {len(splits['eval'])}")
    print(f"Total:      {len(splits['train']) + len(splits['eval'])}")

    # Print 3 samples
    print("\n" + "=" * 60)
    print("SAMPLE TRAINING PROMPTS")
    print("=" * 60)
    for i in range(3):
        ex = splits["train"][i]
        print(f"\n--- Sample {i+1} [{ex['category']}] [{ex['context_condition']}] ---")
        print(ex["prompt"])
        print(f"Correct answer: {ex['answer_option']} (label={ex['answer_label']})")

    # Verify category distribution in train
    cats_train = Counter(splits["train"]["category"])
    cats_eval  = Counter(splits["eval"]["category"])
    print("\n\nCategory distribution (train | eval):")
    for cat in sorted(cats_train.keys()):
        print(f"  {cat}: {cats_train[cat]} | {cats_eval.get(cat, 0)}")

    # Verify ambig/disambig ratio
    conditions = Counter(splits["train"]["context_condition"])
    print(f"\nContext conditions (train): {dict(conditions)}")
    conditions_eval = Counter(splits["eval"]["context_condition"])
    print(f"Context conditions (eval):  {dict(conditions_eval)}")
