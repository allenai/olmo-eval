# Pairwise analysis — how to read the heatmap

Head-to-head win rates between N models on a shared task (or suite of tasks),
with per-cell standard errors (SE) and matrix-wide validity stats in the
footer. Every acronym here is defined in the [Glossary](#glossary) below.

All examples below come from:

```
olmo-eval results pairwise \
  -G olmo-eval-sandboxfusion \
  -t humaneval:3shot:pass_at_1 \
  --metric pass_at_1:code_exec
```

7 models, 164 shared HumanEval instances, binary pass/fail per question.

![Pairwise heatmap for HumanEval 3-shot pass@1](pairwise_heatmap_example.png)

## Reading a cell — `win_rate (SE)`

Each non-diagonal cell shows model A vs model B on **contested** instances
(ties — questions where both models got the same score — are excluded from the
win rate):

```
  Qwen3-8B vs Olmo-3-7B:   57.6% (6.5%)
  Qwen3-8B vs Apertus-8B:  82.5% (4.8%)
```

- `win_rate = wins_a / (wins_a + wins_b)` — fraction of contested questions A won.
- `SE = sqrt(p(1-p) / (n-1))` — standard error of that rate via the Central
  Limit Theorem (CLT), where `n` is the contested count.

**Rough 95% confidence interval (CI): `win_rate ± 2·SE`.**

- `57.6% (6.5%)` → roughly 44.6% – 70.6%. Crosses 50%, so we **cannot** rule
  out a tie between Qwen3 and Olmo-3 from this eval alone.
- `82.5% (4.8%)` → roughly 72.9% – 92.1%. Clearly above 50% → Qwen3 beats
  Apertus decisively.

**Comparing two cells:** a difference between two win rates is meaningful if
it exceeds roughly `2·sqrt(SE1² + SE2²)`.

Lots of ties shrinks the contested `n` and inflates SE — so a cell with
`62.3% (8.1%)` on a high-tie pair is less precise than `58.0% (4.5%)` on a
low-tie pair, even though the headline number looks more decisive.

## Reading the footer

The footer shows matrix-wide validity stats. From the HumanEval example:

```
n for 3pp MDE    median n_eff    shared n / pair    MDE @ 80% power
    2,182            218              164                10.9%
```

### `shared n / pair`
Number of instances every model answered. Your raw sample size for every pair.
HumanEval has 164 questions.

### `MDE @ 80% power`
Minimum Detectable Effect (MDE) — the smallest true win-rate gap the matrix
can reliably resolve at significance level α=0.05 and 80% statistical power,
given its `shared n` and the observed between-model variance.

- **10.9%** means: only pairs where the true gap is ≥ 10.9 percentage points
  (pp) will come out statistically significant. The 57.6% Qwen-vs-Olmo cell
  (7.6pp above tie) is **below** this threshold — don't claim a winner.
- Tiers: good ≤ 3pp, ok ≤ 10pp, bad > 10pp.

### `n for 3pp MDE`
How many shared instances you'd need to pull the MDE down to 3pp.

- **2,182** means: if you want to reliably resolve 3pp gaps on this task, you
  need ~2,182 shared instances — about 13× HumanEval's 164. Practically,
  that's unachievable on HumanEval alone, which is why code evals are often
  pooled into suites (e.g. HumanEval + HumanEval+ + MBPP).
- Tiers relative to current `shared n`: good ≤ 1×, ok ≤ 3×, bad > 3×.

### `median n_eff`
Effective sample size (n_eff) under the paired design — how many independent
samples the paired comparison is worth after accounting for question-level
correlation between models.

- `n_eff = n × (σ_A² + σ_B²) / Var(d)` per pair; the footer shows the median
  across pairs.
- `n_eff > shared n` means models disagree on which questions are hard, so
  pairing is buying precision. `n_eff ≈ shared n` means pairing adds little.
- **218 vs 164** on HumanEval: pairing is helping a bit (218/164 ≈ 1.3×), but
  not dramatically — HumanEval questions are somewhat model-specific in
  difficulty.
- Tiers: good ≥ 2× shared n, ok ≥ 1.2×, bad < 1.2×.

## How cell SE and footer stats differ

- **Per-cell SE** is pair-specific and uses that pair's contested-only `n`.
  It tells you "how precise is *this* comparison?"
- **Footer MDE** is matrix-wide. It uses the paired-difference variance
  pooled across pairs on the full `shared n` (ties included). It tells you
  "how precise are most comparisons in this matrix on average?"
- A pair with many ties (like Qwen vs Olmo: 126 ties out of 164) has a wide
  cell SE even if the footer MDE looks decent — because that pair's contested
  `n` is only 38. Always check the cell SE before quoting the gap.

## Rules of thumb

1. **MDE red** (> 10pp): only trust cells where `win_rate ± 2·SE` is far from 50%.
2. **MDE yellow** (3–10pp): most small- to medium-sized gaps are suggestive
   but not conclusive; lean on the per-cell SE.
3. **MDE green** (≤ 3pp): you can read the matrix head-on; most cells are
   trustworthy.
4. **Scaling up:** halving the MDE requires ~4× more instances. If the number
   in `n for 3pp MDE` is too big to run per-task, try `--suite` to pool
   instances across a related task group.
5. **Ignore cell differences smaller than `2·sqrt(SE_a² + SE_b²)`** when
   comparing two non-diagonal cells in the matrix.

## Glossary

- **Win rate** — fraction of contested instances (ties excluded) on which
  model A outscored model B. `wins_a / (wins_a + wins_b)`.
- **Contested instance** — a shared question where the two models got
  *different* scores. Ties are excluded from the numerator and denominator
  of the win rate.
- **Tie** — a shared question where both models got the same score (within
  `--margin`, which defaults to 0). For binary 0/1 scoring ties are common
  since both models either solve it or neither does.
- **Shared instance** — an instance every model in the matrix has a score
  for. The pairwise comparison runs only on this intersection.
- **SE (standard error)** — the standard deviation of a statistic's sampling
  distribution. Shrinks as `1/√n`. Roughly, the true value is within `±2·SE`
  of the observed value ~95% of the time (CLT-based CI).
- **CLT (Central Limit Theorem)** — lets us treat the sample mean of enough
  independent observations as approximately normal, so we can use `SE` to
  form CIs and run z-tests without bootstrapping.
- **CI (confidence interval)** — `estimate ± 2·SE` is the approximate 95% CI.
  If the CI crosses 50%, we can't distinguish the pair from a tie.
- **α (significance level)** — false-positive rate. The pairwise tool uses
  α=0.05: a 5% chance of declaring a difference when none exists.
- **Power (1−β)** — probability of detecting a real gap of a given size. The
  tool uses 80%: when a true gap equals the MDE, we expect to find it ~80%
  of the time.
- **MDE (Minimum Detectable Effect)** — the smallest true win-rate gap that
  will come out significant at the given α and power. Matrix-wide MDE uses
  the *paired-difference* variance pooled across pairs.
- **Paired comparison** — both models answer the same shared questions, so
  per-question luck cancels in `d_i = score_a_i − score_b_i`. Shrinks the
  variance whenever models agree on what's easy vs hard.
- **Var(d) (paired-difference variance)** — sample variance of `d_i` across
  shared instances. Drives the paired MDE and `n_eff`.
- **σ² (marginal variance)** — per-model sample variance of scores across
  the same instances. `σ_A² + σ_B²` is what the unpaired MDE would use.
- **ω² (omega squared)** — between-question variance of the paired
  conditional means (Miller 2024 Eq. 8). In practice `Var(d) ≈ ω²` when
  each model produces one score per question.
- **n_eff (effective sample size)** — `n × (σ_A² + σ_B²) / Var(d)`. The
  number of independent samples the paired comparison is worth. Higher than
  `shared n` means pairing is buying precision; close to `shared n` means
  models disagree on which questions are hard.
- **Required n (n for target MDE)** — from `MDE ∝ 1/√n`, the instance count
  needed to hit a desired MDE at fixed α and power. The footer reports the
  number for a 3pp target.
- **pp (percentage point)** — absolute difference between two percentages.
  A move from 52% to 55% is a 3pp gap, not a 3% gap.
- **z-test** — two-sample hypothesis test that uses the normal approximation
  from the CLT to decide whether an observed difference is larger than
  chance. The pairwise MDE is derived from the z-test formula.

## See also

- `eval_power.py` — Central Limit Theorem (CLT) helpers
  (`required_sample_size`, `minimum_detectable_effect`,
  `estimate_variance_components`).
- Miller, "Adding Error Bars to Evals" (arXiv:2411.00640) — methodology and
  worked examples.
