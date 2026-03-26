"""
Extract Chain-of-Thought Examples for the Paper

Finds the best qualitative examples of emergent de-biasing reasoning
from Fair-RLVR predictions. These go in the Analysis section of the paper.

Usage:
    python -m src.extract_examples --predictions results/fair_rlvr/predictions.json
"""

import argparse
import json
from pathlib import Path
from src.reward import extract_think, extract_answer, answer_to_index


def categorize_prediction(pred):
    """Classify a prediction into one of several categories for the paper."""
    think = extract_think(pred["model_output"]) or ""
    answer = extract_answer(pred["model_output"])
    predicted_idx = answer_to_index(answer)
    is_correct = predicted_idx == pred["answer_label"]
    is_ambig = pred["context_condition"] == "ambig"
    think_lower = think.lower()

    # Emergent de-biasing: model explicitly identifies insufficient evidence
    evidence_phrases = [
        "not enough information", "no information", "cannot determine",
        "does not specify", "doesn't specify", "no evidence",
        "cannot be determined", "not provided", "not mentioned",
        "insufficient", "we don't know", "we cannot tell",
        "nothing in the context", "doesn't tell us", "does not tell",
        "not stated", "no way to know", "unclear", "ambiguous",
    ]
    has_evidence_reasoning = any(p in think_lower for p in evidence_phrases)

    # Stereotype catching: model explicitly names or rejects a stereotype
    stereotype_phrases = [
        "stereotype", "assumption", "bias", "prejudice",
        "cannot assume", "should not assume", "wrong to assume",
        "generalization", "discriminat",
    ]
    has_stereotype_catch = any(p in think_lower for p in stereotype_phrases)

    # Logical reasoning: model uses logical structure
    logic_phrases = [
        "therefore", "however", "because", "since", "thus",
        "on the other hand", "in contrast", "given that",
        "based on", "according to",
    ]
    has_logic = sum(1 for p in logic_phrases if p in think_lower) >= 2

    category = "other"
    if is_correct and is_ambig and has_evidence_reasoning:
        category = "emergent_debiasing"
    elif is_correct and is_ambig and has_stereotype_catch:
        category = "stereotype_catching"
    elif is_correct and not is_ambig and has_logic:
        category = "evidence_following"
    elif not is_correct and is_ambig:
        category = "failure_ambig"
    elif not is_correct and not is_ambig:
        category = "failure_disambig"

    return {
        "category": category,
        "is_correct": is_correct,
        "is_ambig": is_ambig,
        "has_evidence_reasoning": has_evidence_reasoning,
        "has_stereotype_catch": has_stereotype_catch,
        "has_logic": has_logic,
        "think_length": len(think.split()),
    }


def extract_best_examples(predictions, n_per_category=3):
    """Extract the best examples from each category for the paper."""
    categorized = {}

    for pred in predictions:
        info = categorize_prediction(pred)
        cat = info["category"]
        if cat not in categorized:
            categorized[cat] = []
        categorized[cat].append({
            **pred,
            **info,
        })

    # Sort each category: prefer longer think blocks (more detailed reasoning)
    for cat in categorized:
        categorized[cat].sort(key=lambda x: x["think_length"], reverse=True)

    # Pick top N from each category
    selected = {}
    for cat, preds in categorized.items():
        selected[cat] = preds[:n_per_category]

    return selected, {cat: len(preds) for cat, preds in categorized.items()}


def format_for_paper(examples):
    """Format examples as LaTeX-ready text."""
    lines = []

    for cat, preds in examples.items():
        lines.append(f"\n{'='*60}")
        lines.append(f"CATEGORY: {cat.upper().replace('_', ' ')} ({len(preds)} examples)")
        lines.append(f"{'='*60}")

        for i, pred in enumerate(preds):
            think = extract_think(pred["model_output"]) or "(no think block)"
            answer = extract_answer(pred["model_output"]) or "(no answer)"

            lines.append(f"\n--- Example {i+1} [{pred.get('category', '?')}] [{pred['context_condition']}] ---")
            lines.append(f"Prompt: {pred['prompt']}")
            lines.append(f"\nThink block:\n{think}")
            lines.append(f"\nAnswer: {answer}")
            lines.append(f"Correct: {pred['answer_label']} | Predicted correct: {pred['is_correct']}")
            lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract CoT examples for the paper")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to predictions.json")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file (default: same dir as predictions)")
    parser.add_argument("--n-per-category", type=int, default=3)
    args = parser.parse_args()

    with open(args.predictions) as f:
        predictions = json.load(f)

    print(f"Loaded {len(predictions)} predictions")

    examples, counts = extract_best_examples(predictions, args.n_per_category)

    print("\nCategory distribution:")
    for cat, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    output_text = format_for_paper(examples)
    print(output_text)

    # Save
    if args.output is None:
        args.output = str(Path(args.predictions).parent / "cot_examples.txt")

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(output_text)
    print(f"\nExamples saved to {args.output}")

    # Also save as JSON for programmatic use
    json_output = str(Path(args.output).with_suffix(".json"))
    # Clean up non-serializable fields
    clean_examples = {}
    for cat, preds in examples.items():
        clean_examples[cat] = [{
            "prompt": p["prompt"],
            "think": extract_think(p["model_output"]) or "",
            "answer": extract_answer(p["model_output"]) or "",
            "correct_label": p["answer_label"],
            "is_correct": p["is_correct"],
            "bbq_category": p.get("category", ""),
            "context_condition": p["context_condition"],
        } for p in preds]

    with open(json_output, "w") as f:
        json.dump(clean_examples, f, indent=2)
    print(f"JSON saved to {json_output}")
