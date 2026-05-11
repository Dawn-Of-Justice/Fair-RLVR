"""
BBQ Dataset Pipeline for Fair-RLVR

Loads the Bias Benchmark for Question Answering (BBQ), formats prompts,
and creates template-family-aware train/eval splits for GRPO training.

Split strategy: groups examples by (category, question_index) so that all
ambiguous/disambiguated variants and demographic fills of the same template
family always land in the same split — preventing near-duplicate leakage.
"""

import random
from collections import defaultdict
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


def get_unknown_label(example: dict) -> int:
    """
    Return the index (0/1/2) of the 'Unknown / Cannot be determined' answer option.

    BBQ stores this in the 'unknown_label' field. Falls back to checking answer
    text if the field is absent (for compatibility with older dataset versions).
    """
    if "unknown_label" in example and example["unknown_label"] is not None:
        return int(example["unknown_label"])
    # Fallback: scan answer texts for "unknown" indicator
    for idx, key in enumerate(["ans0", "ans1", "ans2"]):
        text = str(example.get(key, "")).lower()
        if "unknown" in text or "cannot" in text or "can't" in text or "not enough" in text:
            return idx
    return 2  # last-resort default


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
    n_train: Optional[int] = None,
    n_eval: Optional[int] = None,
    seed: int = 42,
    holdout_categories: Optional[list] = None,
    sort_by_family: bool = False,
) -> dict:
    """
    Create template-family-aware train/eval splits from BBQ.

    Groups examples by (category, question_index) so all ambiguous/disambiguated
    and demographic variants of the same template family land in the same split.
    This prevents near-duplicate leakage that inflates eval scores.

    Args:
        n_train: Max training samples. If None, uses all available (~52,643).
        n_eval: Max eval samples per condition. If None, uses all available (~5,849 total).
        seed: Random seed for reproducibility (must be 42 across all scripts).
        holdout_categories: Categories to hold out entirely for OOD eval.
            All 9 base categories are used for training; holdout only affects OOD split.
        sort_by_family: If True, sort training set by question_index so sibling
            variants land in the same GRPO batch (required for R_consistency).

    Returns:
        Dictionary with keys:
            - "train": training dataset (formatted, all 9 base categories)
            - "eval_ambiguous": eval set, ambiguous context only
            - "eval_disambiguated": eval set, disambiguated context only
            - "eval_holdout": holdout categories for OOD testing
            - "eval_intersectional": intersectional categories for OOD testing
    """
    if holdout_categories is None:
        holdout_categories = []

    rng = random.Random(seed)

    # Load all 9 base categories — ALL are used for training (no holdout from train)
    full_pool = load_bbq_all(BBQ_CATEGORIES)
    intersectional_pool = load_bbq_intersectional()

    # ── Template-family-aware 90/10 split ─────────────────
    # Group indices by (category, question_index) — each group is a template family
    family_to_indices = defaultdict(list)
    for i, ex in enumerate(full_pool):
        family_key = (ex["category"], ex["question_index"])
        family_to_indices[family_key].append(i)

    all_families = list(family_to_indices.keys())
    rng.shuffle(all_families)

    # 90% of families → train, 10% → eval
    split_point = int(len(all_families) * 0.9)
    train_families = set(all_families[:split_point])
    eval_families = set(all_families[split_point:])

    train_indices = []
    eval_ambig_indices = []
    eval_disambig_indices = []

    for family_key, indices in family_to_indices.items():
        if family_key in train_families:
            train_indices.extend(indices)
        else:
            for i in indices:
                cond = full_pool[i]["context_condition"]
                if cond == "ambig":
                    eval_ambig_indices.append(i)
                else:
                    eval_disambig_indices.append(i)

    # ── Optionally subsample (dry-run / ablation) ─────────
    if n_train is not None and n_train < len(train_indices):
        rng.shuffle(train_indices)
        train_indices = train_indices[:n_train]
    elif sort_by_family:
        # Sort by question_index to co-batch sibling variants (for R_consistency)
        train_indices = sorted(
            train_indices,
            key=lambda i: (full_pool[i]["category"], full_pool[i]["question_index"]),
        )
    else:
        rng.shuffle(train_indices)

    if n_eval is not None:
        rng.shuffle(eval_ambig_indices)
        rng.shuffle(eval_disambig_indices)
        eval_ambig_indices = eval_ambig_indices[:n_eval]
        eval_disambig_indices = eval_disambig_indices[:n_eval]

    # ── Build HuggingFace datasets ─────────────────────────
    train_ds = full_pool.select(train_indices)
    eval_ambig_ds = full_pool.select(eval_ambig_indices)
    eval_disambig_ds = full_pool.select(eval_disambig_indices)

    # OOD holdout split (from holdout_categories, if any)
    if holdout_categories:
        holdout_ds = load_bbq_all(holdout_categories).map(_add_fields)
    else:
        holdout_ds = Dataset.from_list([])

    train_ds = train_ds.map(_add_fields)
    eval_ambig_ds = eval_ambig_ds.map(_add_fields)
    eval_disambig_ds = eval_disambig_ds.map(_add_fields)
    intersectional_ds = intersectional_pool.map(_add_fields)

    return {
        "train": train_ds,
        "eval_ambiguous": eval_ambig_ds,
        "eval_disambiguated": eval_disambig_ds,
        "eval_holdout": holdout_ds,
        "eval_intersectional": intersectional_ds,
    }


def _add_fields(example: dict) -> dict:
    """Add prompt, answer_option, and unknown_label fields to a BBQ example."""
    example["prompt"] = format_bbq_prompt(example)
    example["answer_option"] = label_to_option(example["answer_label"])
    example["unknown_label"] = get_unknown_label(example)
    return example


def format_for_grpo(dataset: Dataset) -> list[dict]:
    """
    Format dataset for GRPOTrainer.

    Returns a list of dicts with:
        - "prompt": list of messages (system + user)
        - "answer_label": ground truth label index
        - "unknown_label": index of the "Unknown" answer option for this question
        - "category": BBQ category
        - "context_condition": ambig or disambig
        - "question_index": template family identifier
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
            "unknown_label": example.get("unknown_label", 2),
            "category": example["category"],
            "context_condition": example["context_condition"],
            "question_index": example.get("question_index", -1),
        })
    return formatted


# ── Quick test ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading BBQ dataset with template-family-aware split...")
    splits = create_splits(seed=42)

    print(f"\nTrain size:               {len(splits['train'])}")
    print(f"Eval (ambiguous):         {len(splits['eval_ambiguous'])}")
    print(f"Eval (disambiguated):     {len(splits['eval_disambiguated'])}")
    print(f"Eval (intersectional):    {len(splits['eval_intersectional'])}")

    # Verify no question_index overlap between train and eval
    train_families = set(
        (ex["category"], ex["question_index"]) for ex in splits["train"]
    )
    eval_families = set(
        (ex["category"], ex["question_index"]) for ex in splits["eval_ambiguous"]
    ) | set(
        (ex["category"], ex["question_index"]) for ex in splits["eval_disambiguated"]
    )
    overlap = train_families & eval_families
    print(f"\nTemplate family overlap between train/eval: {len(overlap)} (should be 0)")

    # Verify unknown_label field is present
    sample = splits["train"][0]
    print(f"\nSample unknown_label: {sample['unknown_label']} (should be 0, 1, or 2)")

    # Print 3 samples
    print("\n" + "=" * 60)
    print("SAMPLE TRAINING PROMPTS")
    print("=" * 60)
    for i in range(3):
        ex = splits["train"][i]
        print(f"\n--- Sample {i+1} [{ex['category']}] [{ex['context_condition']}] ---")
        print(ex["prompt"])
        print(f"Correct answer: {ex['answer_option']} (label={ex['answer_label']}, unknown_label={ex['unknown_label']})")

    # Category distribution
    from collections import Counter
    cats = Counter(splits["train"]["category"])
    print("\n\nCategory distribution in training set:")
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")
