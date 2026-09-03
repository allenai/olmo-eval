# FrontierScience Olympiad Base-Model Multiple-Choice Plan

## Status

Implemented for an initial verified 83-question release. The packaged task is
`frontierscience_olympiad:mc`; `frontierscience_olympiad:mc:olmo3base` is the
base-model alias. The 17 questions with active source concerns remain excluded.

## Decision summary

Create a separate multiple-choice adaptation of FrontierScience Olympiad for tracking base-model checkpoints. Keep the existing free-response, judge-scored task unchanged.

The proposed base task will:

- Provide four answer choices per problem: the reference answer and three fixed distractors.
- Use a completion-style prompt rather than a chat prompt.
- Score the log-likelihood of answer labels such as ` A`, ` B`, ` C`, and ` D`.
- Report accuracy and continuous logprob-based metrics.
- Require no model judge during routine evaluation.
- Use the distinct task `frontierscience_olympiad:mc`, with the alias
  `frontierscience_olympiad:mc:olmo3base`.

Scores from this adaptation will not be directly comparable with the original FrontierScience scores. Its purpose is a stable internal learning-curve metric during base-model training.

## Motivation

The current `frontierscience_olympiad` task sends the original problem as a chat request, extracts the final answer, and asks an external model judge to determine semantic equivalence. That protocol is suitable for post-trained models but is inconvenient and potentially unreliable for frequent base-model checkpoint evaluation.

Existing OLMo base evaluations use three relevant patterns:

- Logprob-based multiple choice.
- Gold-continuation bits per byte (BPB).
- Completion-style reasoning generation followed by deterministic math scoring.

The multiple-choice approach is the best initial fit for FrontierScience because the released Olympiad data does not contain distractors or reference solution derivations. It contains only the problem, reference answer, subject, and task-group ID.

## Dataset constraints

The pinned Olympiad set contains 100 problems:

| Subject | Count |
|---|---:|
| Physics | 50 |
| Chemistry | 40 |
| Biology | 10 |

The answer formats are heterogeneous:

- Physics answers include numerical results, scalar and vector expressions, differential equations, and piecewise results.
- Chemistry answers include quantities, formulas, reaction equations, compound names, and structural identifiers.
- Biology answers are short semantic phrases or entity/process names.
- Thirteen answers contain InChI/SMILES-style structural identifiers.
- Fourteen answers exceed 150 characters; the longest answer is 990 characters.

These properties make universal exact-match or symbolic free-response scoring brittle. They also make raw gold-answer BPB unusually dependent on serialization and answer length.

## Proposed task format

Strip the benchmark's standard free-response instruction from each problem and replace it with a fixed multiple-choice completion format:

```text
Problem:
{problem}

Choices:
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Answer:
```

The request should be `LOGLIKELIHOOD` with label continuations:

```text
 A
 B
 C
 D
```

Scoring labels instead of full answer texts avoids direct length penalties for long formulas and chemical serializations. Correct positions should be balanced exactly across A, B, C, and D over the 100-item set.

## Metrics

Report at least three metrics:

1. **Accuracy**: whether the highest-logprob label is correct.
2. **Gold-choice probability**: softmax the four label logprobs and average the probability assigned to the correct answer.
3. **Logprob margin**: the correct label logprob minus the highest incorrect label logprob.

Accuracy is the easiest metric to interpret across model families. Gold probability and margin provide smoother checkpoint curves on a dataset with only 100 items.

Report all metrics overall and by subject. Treat the ten-item biology slice as diagnostic because its estimate will be noisy.

## Distractor construction

### Candidate generation

Generate six to ten candidate distractors per problem using a combination of:

- Incorrect solution rollouts from multiple models.
- Frontier-model proposals for plausible expert mistakes.
- Domain-specific programmatic perturbations, such as changing a sign, coefficient, exponent, unit, species, oxidation state, structural isomer, or biological mechanism.

The frontier model may assist with offline dataset construction, but routine scoring must not call a judge.

### Selection requirements

Select three distractors per item that:

- Are scientifically plausible responses to the given problem.
- Correspond to recognizable mistakes rather than arbitrary alternatives.
- Are unambiguously inequivalent to the reference answer.
- Match the reference answer's representation and approximate length.
- Use the same variables, units, chemical representation, and level of specificity where applicable.
- Do not reveal the correct choice through typography, verbosity, metadata tags, or malformed syntax.
- Remain incorrect under reasonable rounding, unit conversion, algebraic equivalence, chemical aliases, and synonymous terminology.

### Domain-specific checks

For physics:

- Check dimensions and variable conventions.
- Detect algebraically equivalent distractors where feasible.
- Include common derivation mistakes without making every distractor dimensionally invalid.

For chemistry:

- Use one consistent representation within an item.
- Canonicalize structures where possible to reject equivalent SMILES or aliases.
- Check stoichiometry, charge, stereochemistry, and oxidation state as appropriate.
- Audit the longest multi-compound answers for context-length problems.

For biology:

- Prefer competing mechanisms or entities at the same level of specificity.
- Reject synonyms and answers that could both be accepted under a reasonable interpretation.

Human or domain-expert review remains the final check, especially for symbolic physics and structural chemistry.

## Pilot protocol

Build a pilot of 15–20 items before curating the full dataset. It should deliberately include:

- Numerical physics.
- Symbolic physics.
- Numerical chemistry.
- Compound identification.
- Structural chemistry.
- Biology mechanisms or entities.
- At least one unusually long answer.

Evaluate both zero-shot formatting and three fixed, non-benchmark demonstrations. The demonstrations should teach the response format without overlapping the scored questions or supplying relevant domain facts.

Run the pilot on several base checkpoints spanning model size or training progress, plus at least one strong post-trained model as an upper anchor.

## Validation gates

Do not scale the pilot until it passes these checks:

1. **Distractor correctness**: reviewers agree that every distractor is incorrect and the gold answer is uniquely best.
2. **Label balance**: correct answers and label-token frequencies are balanced.
3. **Permutation stability**: model ordering is reasonably stable after deterministically permuting choice positions.
4. **Answer-only control**: removing or mismatching the problem should reduce results toward chance, showing that answer style alone does not identify the gold choice.
5. **Checkpoint sensitivity**: gold probability or margin changes smoothly enough to be useful during training.
6. **Upper anchor**: a strong model performs meaningfully above chance and above weak base checkpoints.
7. **Artifact review**: inspect items that all models solve, all models miss, or solve without the problem.
8. **Context audit**: every prompt fits the supported context lengths without silently truncating the problem or choices.

During development, judge-scored free-response results or human scoring may be used as an external validation anchor. The MC metric does not need perfect item-level agreement, but its aggregate checkpoint ordering should be sensible.

## Secondary variants

### Answer-text RC/PMI

Score each candidate answer as a continuation after `Answer:` without displaying choices, then subtract an unconditional answer logprob and normalize for length. This can expose label-following or choice-presentation artifacts, but it should remain a diagnostic because normalization and answer serialization may alter model rankings.

### Generated MC reasoning

Prompt the base model with the problem, choices, and `Solution:`, sample reasoning rollouts, and deterministically extract a final choice label. This permits test-time reasoning and resembles the existing OLMo base math evaluations. It is more sensitive to response formatting and should be attempted after the direct logprob task is stable.

### Two-stage rationale-conditioned scoring

Generate a rationale first, then score the four answer labels conditioned on that rationale. This avoids parsing the final answer while allowing test-time reasoning, but the current request abstraction does not natively chain completion and loglikelihood requests. Treat this as a later runner-level enhancement.

### Gold-answer BPB

A raw or conditional BPB metric is easy to add but should not be a primary capability measure. It rewards likelihood of one exact answer serialization and is particularly problematic for the long chemical identifiers in this dataset.

## Risks and interpretation

- Multiple choice measures recognition and discrimination rather than the original free-response generation task.
- Direct label logprobs provide no extra reasoning tokens, so they may understate capabilities that emerge only through long rollouts.
- Distractor quality defines task difficulty and can change checkpoint rankings.
- Fixed 100-item accuracy will be coarse; continuous metrics are important for training curves.
- Repeated internal evaluation could eventually lead to benchmark overfitting if results influence training decisions heavily.
- Scores must be reported under the new task name and must not be compared numerically with original FrontierScience accuracy.

## Expected effort

- Task implementation and tests: low effort once the data schema is fixed.
- Metric implementation for gold probability and margin: low to moderate effort.
- Pilot distractor generation and review: moderate effort.
- Full 100-item curation and validation: the dominant cost, especially for physics equivalence and structural chemistry.

## Open decisions

- Zero-shot or three fixed external demonstrations.
- Exact prompt wording and separators.
- Whether answer labels are always single-token for the intended tokenizers, and how to handle models where they are not.
- Where the curated choices should live: a checked-in artifact or a separately pinned dataset revision.
- Whether the first release includes only label-logprob scoring or also the RC/PMI diagnostic.
- Required level of domain-expert review before using the metric for training decisions.
- Which checkpoints and post-trained model will form the pilot validation set.

## Implementation acceptance criteria

When implementation is approved, completion should require:

- A separately registered task that does not modify `frontierscience_olympiad`.
- A frozen, versioned four-choice dataset with provenance for every distractor.
- Deterministic, balanced choice ordering.
- Completion-style `LOGLIKELIHOOD` requests with four label continuations.
- Accuracy, gold-choice probability, and logprob-margin metrics overall and by subject.
- Unit tests for prompt construction, instruction removal, shuffling, metric computation, serialization, and task hashing.
- Pilot validation results covering the gates above.
- Documentation stating that the metric is intended for internal base-model tracking and is not comparable with the published FrontierScience score.
