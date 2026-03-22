"""
BBQ Dataset Pipeline for Fair-RLVR

Loads the Bias Benchmark for Question Answering (BBQ), formats prompts,
and creates balanced train/eval splits for GRPO training.
"""

import random
from typing import Optional
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
    n_train: int = 1000,
    n_eval: int = 500,
    ambiguous_ratio: float = 0.7,
    seed: int = 42,
    holdout_categories: Optional[list] = None,
) -> dict:
    """
    Create balanced train/eval splits from BBQ.

    Args:
        n_train: Number of training samples.
        n_eval: Number of evaluation samples.
        ambiguous_ratio: Fraction of ambiguous context samples in training set.
        seed: Random seed for reproducibility.
        holdout_categories: Categories to hold out entirely for OOD eval.
            Defaults to ["religion", "sexual_orientation"].

    Returns:
        Dictionary with keys:
            - "train": training dataset (formatted)
            - "eval_ambiguous": eval set, ambiguous context only
            - "eval_disambiguated": eval set, disambiguated context only
            - "eval_holdout": held-out categories for OOD testing
            - "eval_intersectional": intersectional categories for OOD testing
    """
    if holdout_categories is None:
        holdout_categories = ["religion", "sexual_orientation"]

    random.seed(seed)

    # Separate train and holdout categories
    train_categories = [c for c in BBQ_CATEGORIES if c not in holdout_categories]

    # Load datasets
    train_pool = load_bbq_all(train_categories)
    holdout_pool = load_bbq_all(holdout_categories)
    intersectional_pool = load_bbq_intersectional()

    # Group indices by (category, context_condition)
    cat_cond_indices = {}
    for i, ex in enumerate(train_pool):
        key = (ex["category"], ex["context_condition"])
        cat_cond_indices.setdefault(key, []).append(i)

    # Shuffle each group
    for key in cat_cond_indices:
        random.shuffle(cat_cond_indices[key])

    # Balanced sampling: equal samples per category, then split ambig/disambig within each
    n_cats = len(train_categories)
    per_cat = n_train // n_cats
    n_ambig_per_cat = int(per_cat * ambiguous_ratio)
    n_disambig_per_cat = per_cat - n_ambig_per_cat

    train_indices = []
    for cat in train_categories:
        cat_title = cat.replace("_", " ").title().replace(" ", "_")
        # Try both naming conventions
        ambig_key = None
        disambig_key = None
        for key in cat_cond_indices:
            if key[0].lower().replace("_", "") == cat.lower().replace("_", "") and key[1] == "ambig":
                ambig_key = key
            if key[0].lower().replace("_", "") == cat.lower().replace("_", "") and key[1] == "disambig":
                disambig_key = key

        if ambig_key:
            train_indices.extend(cat_cond_indices[ambig_key][:n_ambig_per_cat])
        if disambig_key:
            train_indices.extend(cat_cond_indices[disambig_key][:n_disambig_per_cat])

    random.shuffle(train_indices)

    # Collect remaining indices for eval
    train_set = set(train_indices)
    ambig_indices = [i for i in range(len(train_pool))
                     if i not in train_set and train_pool[i]["context_condition"] == "ambig"]
    disambig_indices = [i for i in range(len(train_pool))
                        if i not in train_set and train_pool[i]["context_condition"] == "disambig"]

    random.shuffle(ambig_indices)
    random.shuffle(disambig_indices)

    # Eval indices
    eval_ambig_idx = ambig_indices[:n_eval]
    eval_disambig_idx = disambig_indices[:n_eval]

    # Build datasets
    train_ds = train_pool.select(train_indices)
    eval_ambig_ds = train_pool.select(eval_ambig_idx)
    eval_disambig_ds = train_pool.select(eval_disambig_idx)

    # Format all datasets with prompt column
    def add_prompt(example):
        example["prompt"] = format_bbq_prompt(example)
        example["answer_option"] = label_to_option(example["answer_label"])
        return example

    train_ds = train_ds.map(add_prompt)
    eval_ambig_ds = eval_ambig_ds.map(add_prompt)
    eval_disambig_ds = eval_disambig_ds.map(add_prompt)
    holdout_ds = holdout_pool.map(add_prompt)
    intersectional_ds = intersectional_pool.map(add_prompt)

    return {
        "train": train_ds,
        "eval_ambiguous": eval_ambig_ds,
        "eval_disambiguated": eval_disambig_ds,
        "eval_holdout": holdout_ds,
        "eval_intersectional": intersectional_ds,
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
    print("Loading BBQ dataset...")
    splits = create_splits(n_train=100, n_eval=50, seed=42)

    print(f"\nTrain size: {len(splits['train'])}")
    print(f"Eval (ambiguous): {len(splits['eval_ambiguous'])}")
    print(f"Eval (disambiguated): {len(splits['eval_disambiguated'])}")
    print(f"Eval (holdout): {len(splits['eval_holdout'])}")
    print(f"Eval (intersectional): {len(splits['eval_intersectional'])}")

    # Print 5 samples
    print("\n" + "=" * 60)
    print("SAMPLE TRAINING PROMPTS")
    print("=" * 60)
    for i in range(5):
        ex = splits["train"][i]
        print(f"\n--- Sample {i+1} [{ex['category']}] [{ex['context_condition']}] ---")
        print(ex["prompt"])
        print(f"Correct answer: {ex['answer_option']} (label={ex['answer_label']})")

    # Verify category distribution
    from collections import Counter
    cats = Counter(splits["train"]["category"])
    print("\n\nCategory distribution in training set:")
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")

    # Verify ambig/disambig ratio
    conditions = Counter(splits["train"]["context_condition"])
    print(f"\nContext conditions: {dict(conditions)}")
