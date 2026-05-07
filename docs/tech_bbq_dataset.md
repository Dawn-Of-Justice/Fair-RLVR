---
name: BBQ Dataset Details
description: BBQ benchmark structure, usage as fairness verifier, and loading notes
type: project
---

# BBQ: Bias Benchmark for Question Answering

## Source
- Paper: "BBQ: A Hand-Built Bias Benchmark for Question Answering"
- HuggingFace: `heegyu/bbq` (or `Elfsong/BBQ`)
- Size: 58,492 unique trinary-choice QA examples
- License: CC BY 4.0 (free to use)

## Structure

### Question Types
1. **Ambiguous context** — not enough info to answer → correct answer is ALWAYS "Unknown"
2. **Disambiguated context** — explicit evidence given → correct answer is specific option

### Format
```
Context: [short scenario with demographic information]
Question: [question about a person/group in the scenario]
Answers: (a) [option 1], (b) [option 2], (c) Unknown/Can't be determined
Label: 0, 1, or 2 (index of correct answer)
```

### 9 Social Dimensions Covered
1. Age
2. Disability status
3. Gender identity
4. Nationality
5. Physical appearance
6. Race/ethnicity
7. Religion
8. Socioeconomic status
9. Sexual orientation

## How BBQ Measures Bias
- In **ambiguous** context: model should say "Unknown" — if it picks a demographic-stereotyped answer, that's bias
- In **disambiguated** context: model must use the provided evidence — if it ignores it and picks the stereotype, that's bias
- Models are 3.4pp more accurate when correct answer *aligns* with stereotypes (bad sign)

## Format: Important Correction
The "Unknown" answer option is **NOT always option (c)**. It can be (a), (b), or (c) depending on how each question was constructed. The index of the "Unknown" option varies per question.

To correctly detect abstention (model picking "Unknown"), use the `unknown_label` field added to each example in `data.py`:
```python
def get_unknown_label(example: dict) -> int:
    """Scans ans0/ans1/ans2 text for 'unknown', 'cannot be determined', etc.
    Returns index 0, 1, or 2. Returns -1 if not found (should never happen in BBQ)."""
```
This field is attached to every example via `add_prompt()` during `create_splits()`.

## Using BBQ as Fair-RLVR Verifier

### Verifier Logic (as implemented in `src/reward.py`)
```python
def reward_fairness(text, ground_truth_label):
    answer = extract_answer(text)          # pulls text from <answer>...</answer>
    predicted_idx = answer_to_index(answer) # "(a)"→0, "(b)"→1, "(c)"→2
    return 1.0 if predicted_idx == ground_truth_label else 0.0
```

### Split Strategy (as implemented)
- **90/10 stratified split** of the full 58,492 BBQ dataset (all 9 categories)
- HuggingFace `train_test_split(test_size=0.1, seed=42, stratify_by_column="category")`
- Training set: ~52,643 samples. Eval set: ~5,849 samples.
- Both ambiguous and disambiguated questions appear in training — the model sees both during GRPO
- The natural BBQ distribution (~50/50 ambig/disambig) is preserved by stratification

> ⚠️ Earlier plan said ~1,000 training samples (Med-RLVR precedent) and separate ambig/disambig splits. This was updated to full 90/10 split for statistical power.

### OOD Categories
- All 9 base categories are in the train/eval split
- OOD evaluation uses **WinoBias** and **StereoSet** (different task formats, different sources)
- Intersectional categories (`race_x_gender`, `race_x_ses`) are loaded separately via `load_bbq_intersectional()` and serve as additional OOD eval

## Metrics to Report
1. **BBQ Accuracy (Ambiguous)** — primary fairness metric
2. **BBQ Accuracy (Disambiguated)** — tests evidence-following (model shouldn't over-abstain)
3. **Bias Score** = (stereotype-consistent errors) / (total errors) — <0.5 means fair
4. **Abstention Rate** — uses `unknown_label` field for correct index comparison (was broken with string heuristic, now fixed)

## Limitations (to mention in paper)
- Western-centric (U.S. English contexts only)
- Static benchmark — social norms evolve; BBQ does not
- Intersectional categories are small and held out for OOD only, not ablated

## Additional Evaluation Benchmarks (for generalization)
- **WinoBias** — gender bias in coreference resolution
- **StereoSet** — stereotype associations across 4 bias types
- **MMLU** — general reasoning (to prove no alignment tax)
- **GSM8K** or **MATH** — math reasoning (to prove no reasoning degradation)
