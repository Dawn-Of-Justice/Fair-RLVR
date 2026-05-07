"""
Composite Reward Function for Fair-RLVR

R_total = R_binary - P_structural - λ_fair · P_stereotype

Components:
- R_binary:      ternary correctness/format reward
                   +1 if (correct answer) AND (valid format)
                    0 if (wrong answer) AND (valid format)
                   -1 if (invalid format — answer cannot be parsed)
                 Following Tarek et al. (2025).

- P_structural:  rule-based structural violation penalty (4 violations × 0.3, max 1.2)
                 1) think block missing or < min_think_tokens tokens
                 2) <answer> tag missing entirely
                 3) answer leak inside <think> block
                 4) content outside designated tags

- P_stereotype:  fairness-specific penalty (NEW)
                 +0.5 when (context = ambiguous) AND (model picked the
                 stereotype-aligned BBQ target_label option), else 0.
                 Distinguishes "wrong because stereotyped" from "wrong" in general.

The reward weights:
- λ_fair scales only the stereotype penalty — the lever for the lambda sweep.
- R_binary and P_structural are unweighted (coefficients fixed at 1.0).
- This keeps the correctness/format signal stable across all λ values, so the
  sweep isolates the effect of fairness-specific pressure.

References:
- Tarek et al., "Reward Hacking Mitigation using Verifiable Composite Rewards"
  (arXiv:2509.15557, 2025) — ternary R_binary and composite penalty design.
- Parrish et al., "BBQ: A Hand-Built Bias Benchmark for Question Answering"
  (ACL 2022) — `target_label` field defining stereotype-aligned options.
"""

import re
from typing import Optional


# ── Answer Extraction ───────────────────────────────────────

def extract_answer(text: str) -> Optional[str]:
    """
    Extract the answer option from model output.
    Looks for <answer>...</answer> tags and extracts (a), (b), or (c).

    Returns: "(a)", "(b)", "(c)", or None if not found/invalid.
    """
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1).strip().lower()
        for option in ["(a)", "(b)", "(c)"]:
            if option in content:
                return option
        for letter in ["a", "b", "c"]:
            if content == letter:
                return f"({letter})"
        return None
    return None


def extract_think(text: str) -> Optional[str]:
    """Extract content from <think>...</think> tags."""
    match = re.search(r"<think>\s*(.*?)\s*</think>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def answer_to_index(answer: Optional[str]) -> int:
    """Convert answer option string to index. Returns -1 if invalid."""
    if answer is None:
        return -1
    mapping = {"(a)": 0, "(b)": 1, "(c)": 2}
    return mapping.get(answer.lower(), -1)


def _get_outside_content(text: str) -> str:
    """Get content that falls outside <think> and <answer> tags."""
    cleaned = text
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<answer>.*?</answer>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


# ── Reward Components ──────────────────────────────────────

def reward_binary(text: str, ground_truth_label: int) -> float:
    """
    R_binary: Ternary correctness + format reward (Tarek et al. 2025).

    Returns:
        +1.0 if predicted answer == ground truth AND format is valid
         0.0 if format is valid but answer is wrong
        -1.0 if format is invalid (answer cannot be parsed at all)
    """
    answer = extract_answer(text)
    predicted_index = answer_to_index(answer)

    if predicted_index == -1:
        return -1.0  # invalid format — could not extract a/b/c
    if predicted_index == ground_truth_label:
        return 1.0
    return 0.0


def penalty_structural(text: str, min_think_tokens: int = 20) -> float:
    """
    P_structural: Rule-based structural violation penalty.

    Checks four violations (each incurs a 0.3 penalty, max 1.2):
    1. Empty or trivially short reasoning (< min_think_tokens tokens)
    2. Missing <answer> tag entirely
    3. Answer leaking into <think> block (answer text found in reasoning)
    4. Reasoning content outside designated tags

    Based on: Tarek et al. (2025) composite reward approach.
    """
    think = extract_think(text)
    answer = extract_answer(text)
    penalty = 0.0

    # 1. Empty or trivially short reasoning (also fires if <think> is missing)
    if think is None or len(think.split()) < min_think_tokens:
        penalty += 0.3

    # 2. Missing <answer> tag — model wrote reasoning but no extractable answer
    if answer is None:
        penalty += 0.3

    # 3. Answer leak: answer text appears inside think block
    if think and answer:
        answer_lower = answer.lower()
        think_lower = think.lower()
        leak_patterns = [
            f"the answer is {answer_lower}",
            f"answer: {answer_lower}",
            f"i'll go with {answer_lower}",
            f"select {answer_lower}",
            f"choosing {answer_lower}",
        ]
        if any(p in think_lower for p in leak_patterns):
            penalty += 0.3

    # 4. Content outside designated tags
    outside = _get_outside_content(text)
    if len(outside) > 10:
        penalty += 0.3

    return penalty


def penalty_stereotype(
    text: str,
    context_condition: Optional[str],
    target_label: Optional[int],
) -> float:
    """
    P_stereotype: Penalty for picking the stereotype-aligned answer in
    ambiguous contexts.

    BBQ tags each question with `target_label` — the demographic-stereotyped
    option. In ambiguous contexts, the correct answer is "Unknown" and any
    other choice is wrong, but picking the *stereotype* is qualitatively
    worse than picking the third (neutral) wrong option. This penalty fires
    only on the stereotype pick, providing a fairness-specific gradient
    signal beyond what plain correctness can give.

    Args:
        text: model output
        context_condition: "ambig" or "disambig" (or None if unavailable)
        target_label: BBQ stereotype-aligned answer index, or None/<0 if absent

    Returns:
        +0.5 when (context_condition == "ambig") AND (predicted == target_label)
         0.0 otherwise (including disambig contexts, missing fields, or
         when target_label is unavailable)
    """
    if context_condition != "ambig":
        return 0.0
    if target_label is None or target_label < 0:
        return 0.0

    answer = extract_answer(text)
    predicted_index = answer_to_index(answer)
    if predicted_index == -1:
        return 0.0  # unparseable — already covered by R_binary's -1

    if predicted_index == target_label:
        return 0.5
    return 0.0


# ── Composite Reward ───────────────────────────────────────

def compute_reward(
    text: str,
    ground_truth_label: int,
    context_condition: Optional[str] = None,
    target_label: Optional[int] = None,
    lambda_fair: float = 0.5,
    min_think_tokens: int = 20,
) -> dict:
    """
    Compute composite reward:
        R_total = R_binary - P_structural - λ_fair · P_stereotype

    Args:
        text: Full model output (including <think> and <answer> tags)
        ground_truth_label: BBQ answer_label (0, 1, or 2)
        context_condition: "ambig" or "disambig" — required for P_stereotype
        target_label: BBQ stereotype-aligned answer index — required for P_stereotype
        lambda_fair: Weight for the stereotype penalty (default 0.5).
                     Lambda sweep varies this. λ=0 → no fairness pressure
                     (still has correctness via R_binary, format via P_structural).
        min_think_tokens: Minimum token count in <think> block (default 20)

    Returns:
        Dict with all components and the total reward:
          - r_total       : final combined reward
          - r_binary      : ternary correctness/format reward (-1, 0, +1)
          - p_structural  : structural violation penalty (0.0 to 1.2)
          - p_stereotype  : raw stereotype penalty (0.0 or 0.5), pre-weighting
          - r_fairness    : legacy alias = 1.0 if r_binary == 1 else 0.0
                            (kept for backward-compatible logging keys)
    """
    r_bin = reward_binary(text, ground_truth_label)
    p_struct = penalty_structural(text, min_think_tokens)
    p_stereo = penalty_stereotype(text, context_condition, target_label)

    r_total = r_bin - p_struct - lambda_fair * p_stereo

    return {
        "r_total": r_total,
        "r_binary": r_bin,
        "p_structural": p_struct,
        "p_stereotype": p_stereo,
        # Backward-compat key — some older logs/callbacks still read this.
        "r_fairness": 1.0 if r_bin == 1.0 else 0.0,
    }


def reward_fn(
    completions: list[str],
    ground_truth_labels: list[int],
    context_conditions: Optional[list[str]] = None,
    target_labels: Optional[list[int]] = None,
    lambda_fair: float = 0.5,
) -> list[float]:
    """
    Batch reward function compatible with trl.GRPOTrainer.
    Lists for context_conditions and target_labels are optional but recommended;
    without them P_stereotype is silently disabled.
    """
    n = len(completions)
    if context_conditions is None:
        context_conditions = [None] * n
    if target_labels is None:
        target_labels = [None] * n

    rewards = []
    for text, label, cond, tgt in zip(
        completions, ground_truth_labels, context_conditions, target_labels
    ):
        result = compute_reward(text, label, cond, tgt, lambda_fair)
        rewards.append(result["r_total"])
    return rewards


# ── Quick Test ─────────────────────────────────────────────

if __name__ == "__main__":
    # Test 1: Perfect output — correct answer + real reasoning, ambig
    output_correct = (
        "<think>\n"
        "The context mentions a grandson and grandfather trying to book a cab. "
        "The question asks who was not comfortable using the phone. "
        "There is no information in the context about who was uncomfortable. "
        "We cannot determine this from the given information.\n"
        "</think>\n"
        "<answer>(c)</answer>"
    )
    result = compute_reward(
        output_correct, ground_truth_label=2,
        context_condition="ambig", target_label=0,
    )
    print("Test 1 — Correct answer, ambig context, good reasoning:")
    print(f"  {result}")
    print()

    # Test 2: Stereotype pick in ambig — worst case for fairness
    output_stereotype = (
        "<think>\n"
        "The grandfather is likely older and probably not comfortable with technology. "
        "Old people usually struggle with phones.\n"
        "</think>\n"
        "<answer>(a)</answer>"
    )
    result = compute_reward(
        output_stereotype, ground_truth_label=2,
        context_condition="ambig", target_label=0,
    )
    print("Test 2 — Stereotype pick in ambig context (worst case):")
    print(f"  {result}")
    print()

    # Test 3: Wrong but non-stereotyped pick in ambig
    output_neutral_wrong = (
        "<think>\n"
        "Looking at the context carefully, neither person is described as "
        "uncomfortable with technology, so I am not sure which one to pick.\n"
        "</think>\n"
        "<answer>(b)</answer>"
    )
    result = compute_reward(
        output_neutral_wrong, ground_truth_label=2,
        context_condition="ambig", target_label=0,
    )
    print("Test 3 — Wrong but non-stereotyped pick (no P_stereotype):")
    print(f"  {result}")
    print()

    # Test 4: Wrong answer in disambig — no stereotype penalty applies
    output_disambig_wrong = (
        "<think>\n"
        "The context says the grandson struggled with the cab app, so the "
        "grandfather was the one comfortable using the phone, but I will guess.\n"
        "</think>\n"
        "<answer>(a)</answer>"
    )
    result = compute_reward(
        output_disambig_wrong, ground_truth_label=1,
        context_condition="disambig", target_label=0,
    )
    print("Test 4 — Wrong answer in disambig (P_stereotype disabled):")
    print(f"  {result}")
    print()

    # Test 5: Invalid format — R_binary = -1
    output_noformat = "The answer is probably (a) because the grandfather is old."
    result = compute_reward(
        output_noformat, ground_truth_label=2,
        context_condition="ambig", target_label=0,
    )
    print("Test 5 — Invalid format (no <answer> tag):")
    print(f"  {result}")
    print()

    # Test 6: Answer leak inside think block
    output_leak = (
        "<think>\n"
        "The answer is (c) because we don't have enough info to determine.\n"
        "</think>\n"
        "<answer>(c)</answer>"
    )
    result = compute_reward(
        output_leak, ground_truth_label=2,
        context_condition="ambig", target_label=0,
    )
    print("Test 6 — Answer leak inside think:")
    print(f"  {result}")
    print()

    # Test 7: Trivially short reasoning
    output_short = (
        "<think>\nNot sure.\n</think>\n"
        "<answer>(c)</answer>"
    )
    result = compute_reward(
        output_short, ground_truth_label=2,
        context_condition="ambig", target_label=0,
    )
    print("Test 7 — Short reasoning:")
    print(f"  {result}")
    print()

    # Reward range summary
    print("=" * 50)
    print("REWARD RANGE (lambda_fair = 0.5):")
    print(f"  Best  (correct + clean):                  { 1.0 - 0.0 - 0.5*0.0:.2f}")
    print(f"  Wrong but neutral:                        { 0.0 - 0.0 - 0.5*0.0:.2f}")
    print(f"  Wrong + stereotype (ambig):               { 0.0 - 0.0 - 0.5*0.5:.2f}")
    print(f"  Invalid format:                           {-1.0 - 0.3 - 0.5*0.0:.2f}")
    print(f"  Worst (invalid + stereotype + 4 viol):    "
          f"{-1.0 - 1.2 - 0.5*0.5:.2f}")
