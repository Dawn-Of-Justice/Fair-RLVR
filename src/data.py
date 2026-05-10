"""
BBQ Dataset Pipeline for Fair-RLVR

Loads the Bias Benchmark for Question Answering (BBQ), formats prompts,
and creates template-family-aware train/eval splits for GRPO training.

Split strategy
--------------
BBQ generates questions from templates. Each template produces a cluster of
related examples that share the same underlying narrative:
  - ambiguous context  ↔  disambiguated context  (same story, different info)
  - nonneg polarity    ↔  neg polarity            (same question, flipped)
  - demographic fill A ↔  demographic fill B      (same template, different groups)

A naive random 90/10 split puts near-duplicates from the same template in both
train and eval, inflating eval scores. Fix: group examples by
(category, question_index) — the template-family key — and split the *groups*,
never individual examples.
"""

import random
import warnings
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
    "inside <answer> tags. Your final answer must be exactly one of: (a), (b), or (c).\n\n"
    "Output format:\n"
    "<think>\n[step-by-step reasoning]\n</think>\n"
    "<answer>[exactly one of (a), (b), (c)]</answer>"
)


# Phrases that identify the "Unknown / Can't determine" answer option in BBQ
_UNKNOWN_PHRASES = frozenset([
    "unknown", "cannot be determined", "can't be determined",
    "not enough info", "not enough information", "undetermined",
    "cannot tell", "can't tell",
])


def get_unknown_label(example: dict) -> int:
    """
    Return the index (0, 1, or 2) of the 'Unknown / Can't determine' answer option.

    BBQ always includes one such option per question. Scans ans0, ans1, ans2
    for known indicator phrases.

    Returns -1 if no option matches (should not happen for standard BBQ).
    """
    for i, key in enumerate(["ans0", "ans1", "ans2"]):
        text = example.get(key, "").lower()
        if any(phrase in text for phrase in _UNKNOWN_PHRASES):
            return i
    return -1


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


def _get_family_key(example: dict, use_question_index: bool) -> tuple:
    """
    Return the template-family key for a BBQ example.

    Primary: (category, question_index) — BBQ's own template identifier.
    Fallback: (category, question) — question text is identical across
              ambig/disambig variants of the same template, so it groups
              the most critical leakage pairs even without question_index.
    """
    category = example.get("category", "unknown")
    if use_question_index:
        return (category, example.get("question_index", -1))
    return (category, example.get("question", ""))


def create_splits(
    train_ratio: float = 0.9,
    seed: int = 42,
) -> dict:
    """
    Create a 90/10 train/eval split using template-family grouping.

    Groups BBQ examples by (category, question_index) so that all examples
    from the same template family — ambiguous/disambiguated pairs, negated/
    non-negated variants, and demographic swaps — land in the same split.
    This prevents near-duplicate leakage that inflates eval scores.

    Args:
        train_ratio: Fraction of template families used for training (default 0.9).
        seed: Random seed for reproducibility. Must match across train/eval scripts.

    Returns:
        Dictionary with keys:
            - "train": training dataset (~52,643 samples at 90%)
            - "eval":  evaluation dataset (~5,849 samples at 10%)
            - "n_train_families": number of template families in train split
            - "n_eval_families":  number of template families in eval split
    """
    full_dataset = load_bbq_all()

    # ── Detect whether question_index is a usable family key ─────────────────
    # If question_index is a unique row counter rather than a template ID,
    # every "family" has size 1 and we gain nothing over a random split.
    # Check by sampling the first 500 rows.
    sample_size = min(500, len(full_dataset))
    sample = [full_dataset[i] for i in range(sample_size)]
    has_question_index = "question_index" in full_dataset.column_names

    if has_question_index:
        sample_families: dict = defaultdict(int)
        for ex in sample:
            key = (ex.get("category", ""), ex.get("question_index", -1))
            sample_families[key] += 1
        avg_family_size = sample_size / max(len(sample_families), 1)
        use_question_index = avg_family_size >= 2.0
        if not use_question_index:
            warnings.warn(
                f"question_index average family size = {avg_family_size:.2f} < 2.0. "
                "Falling back to (category, question) as family key. "
                "This groups ambig/disambig pairs but may miss neg/nonneg variants."
            )
    else:
        use_question_index = False
        warnings.warn(
            "question_index column not found in BBQ dataset. "
            "Falling back to (category, question) as family key."
        )

    # ── Build family index: family_key → list of row indices ─────────────────
    families: dict = defaultdict(list)
    for i in range(len(full_dataset)):
        key = _get_family_key(full_dataset[i], use_question_index)
        families[key].append(i)

    family_keys = list(families.keys())
    n_families = len(family_keys)
    avg_size = len(full_dataset) / n_families

    print(f"Template families found: {n_families}  (avg {avg_size:.1f} examples each)")
    print(f"Family key: {'(category, question_index)' if use_question_index else '(category, question)'}")

    # ── Split families (not examples) 90/10 ──────────────────────────────────
    rng = random.Random(seed)
    rng.shuffle(family_keys)

    n_train_families = int(n_families * train_ratio)
    train_family_set = set(family_keys[:n_train_families])

    train_indices = []
    eval_indices = []
    for key, indices in families.items():
        if key in train_family_set:
            train_indices.extend(indices)
        else:
            eval_indices.extend(indices)

    # ── Apply prompt formatting ───────────────────────────────────────────────
    def add_prompt(example):
        example["prompt"] = format_bbq_prompt(example)
        example["answer_option"] = label_to_option(example["answer_label"])
        example["unknown_label"] = get_unknown_label(example)
        # Store the resolved family key so train.py doesn't need to re-derive it.
        # Include context_condition: ambig and disambig have different correct answers,
        # so cross-condition consistency pairing would add noise, not signal.
        if use_question_index:
            key_part = str(example.get("question_index", -1))
        else:
            key_part = example.get("question", "")
        example["template_family_key"] = (
            f"{example['category']}:{key_part}:{example['context_condition']}"
        )
        return example

    train_ds = full_dataset.select(train_indices).map(add_prompt)
    eval_ds  = full_dataset.select(eval_indices).map(add_prompt)

    print(f"Train: {len(train_ds)} examples from {n_train_families} families")
    print(f"Eval:  {len(eval_ds)} examples from {n_families - n_train_families} families")

    return {
        "train": train_ds,
        "eval":  eval_ds,
        "n_train_families": n_train_families,
        "n_eval_families":  n_families - n_train_families,
    }


# ── Quick test ──────────────────────────────────────────────
if __name__ == "__main__":
    from collections import Counter

    print("Loading BBQ dataset with template-family split...")
    splits = create_splits(seed=42)

    print(f"\nTrain size: {len(splits['train'])}")
    print(f"Eval size:  {len(splits['eval'])}")
    print(f"Total:      {len(splits['train']) + len(splits['eval'])}")
    print(f"Train families: {splits['n_train_families']}")
    print(f"Eval families:  {splits['n_eval_families']}")

    # Verify no question_index overlap between train and eval
    if "question_index" in splits["train"].column_names:
        train_keys = set(zip(splits["train"]["category"], splits["train"]["question_index"]))
        eval_keys  = set(zip(splits["eval"]["category"],  splits["eval"]["question_index"]))
        overlap = train_keys & eval_keys
        if overlap:
            print(f"\n⚠️  LEAKAGE DETECTED: {len(overlap)} template families appear in BOTH splits!")
        else:
            print(f"\n✅ No template-family overlap between train and eval.")

    # Print 3 samples
    print("\n" + "=" * 60)
    print("SAMPLE TRAINING PROMPTS")
    print("=" * 60)
    for i in range(3):
        ex = splits["train"][i]
        print(f"\n--- Sample {i+1} [{ex['category']}] [{ex['context_condition']}] ---")
        print(ex["prompt"])
        print(f"Correct answer: {ex['answer_option']} (label={ex['answer_label']})")

    # Verify category distribution
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
