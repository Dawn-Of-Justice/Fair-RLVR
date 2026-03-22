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

## Using BBQ as Fair-RLVR Verifier

### Verifier Logic
```python
def fairness_reward(model_answer, bbq_label):
    # model_answer: extracted from <answer> block (0, 1, or 2)
    # bbq_label: ground truth index from BBQ
    return 1.0 if model_answer == bbq_label else 0.0
```

### Critical: Ambiguous vs Disambiguated Split
- For training: use AMBIGUOUS context (tests spontaneous stereotype avoidance)
- For evaluation: use BOTH (measures full fairness profile)
- Ambiguous is harder for models → better learning signal

### Sampling Strategy
- ~1,000 training samples (Med-RLVR precedent)
- Balance across 9 categories (111 per category)
- Balance ambiguous vs disambiguated (50/50 or 70/30 ambiguous-heavy)
- Reserve at least 500 for evaluation (held-out, unseen categories)

## Metrics to Report
1. **BBQ Accuracy (Ambiguous)** — primary fairness metric
2. **BBQ Accuracy (Disambiguated)** — tests evidence-following
3. **Bias Score** = (% stereotype-consistent errors) / (total errors)
4. **Abstention Rate** — how often model says "Unknown" (watch for over-abstention)

## Limitations (to mention in paper)
- Western-centric (U.S. English contexts only)
- No intersectional questions (race × gender combined)
- Static benchmark — model may overfit if used both as train and eval
- Fix: train on subset, eval on held-out + external benchmark (WinoBias, StereoSet)

## Additional Evaluation Benchmarks (for generalization)
- **WinoBias** — gender bias in coreference resolution
- **StereoSet** — stereotype associations across 4 bias types
- **MMLU** — general reasoning (to prove no alignment tax)
- **GSM8K** or **MATH** — math reasoning (to prove no reasoning degradation)
