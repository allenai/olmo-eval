# Qwen3 MATH-500 response coherence across active-expert counts

Date: 2026-07-20

## Question

Does the strong Qwen3 MATH-500 score over much of the active-expert sweep hide degradation in
response coherence, completion behavior, formatting, or reasoning efficiency?

This analysis uses every existing normalized MATH-500 replicate at
K={1,2,3,...,16,32}. There are three replicates and 1,500 responses per K except K=13, K=14, and
K=15, which have two replicates and 1,000 responses each. Examples are aligned to the corresponding
K=8 replicate and MATH-500 document ID for paired comparisons.

Saved olmo-eval predictions contain the parser-returned assistant content but not Qwen's hidden
`reasoning_content`. Here, "final response" does not mean only the terse extracted scalar answer:
it commonly contains a complete user-facing worked solution. The saved generated-token count
covers hidden reasoning plus that returned response. Consequently, this analysis directly evaluates
the coherence of the returned solution and generation stability, while hidden-reasoning coherence
can only be inferred indirectly from length, termination, and the returned response.

A spot check of one complete K=8 replicate supports that the Qwen reasoning parser worked: none of
the 500 returned responses contains `<think>` tags, the median returned text is 1,589 characters,
and the server reports a much larger median of 3,506 completion tokens. The latter includes hidden
reasoning. The returned response can nevertheless be several thousand characters because the model
often writes a long polished solution after thinking. Parser behavior should still be treated as a
caveat rather than assuming that `final_output` is synonymous with `extracted_answer`.

## Aggregate diagnostics

| K | Accuracy | Median generated tokens | Mean generated tokens | 32k cap hits | Empty visible final | Formatting proxy |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.00% | 32,768 | 22,584 | 51.87% | 100.00% | 0.00% |
| 2 | 0.07% | 32,768 | 31,994 | 96.87% | 99.13% | 0.13% |
| 3 | 62.13% | 7,538 | 10,979 | 10.80% | 29.07% | 6.27% |
| 4 | 94.27% | 4,604 | 6,284 | 0.60% | 0.87% | 1.60% |
| 5 | 95.07% | 3,936 | 5,598 | 0.20% | 0.13% | 0.87% |
| 6 | 95.07% | 3,828 | 5,340 | 0.40% | 0.27% | 1.00% |
| 7 | 95.13% | 3,554 | 5,136 | 0.47% | 0.40% | 1.33% |
| 8 | 95.33% | 3,470 | 4,999 | 0.20% | 0.20% | 0.87% |
| 9 | 94.40% | 3,377 | 4,953 | 0.47% | 0.27% | 1.27% |
| 10 | 94.87% | 3,322 | 4,963 | 0.27% | 0.27% | 0.87% |
| 11 | 95.00% | 3,201 | 4,763 | 0.47% | 0.47% | 1.07% |
| 12 | 94.47% | 3,166 | 4,871 | 0.80% | 0.73% | 1.20% |
| 13 | 93.80% | 3,154 | 4,820 | 0.60% | 0.50% | 1.20% |
| 14 | 94.00% | 3,082 | 4,774 | 0.70% | 0.60% | 1.20% |
| 15 | 94.40% | 3,027 | 4,683 | 0.40% | 0.40% | 1.10% |
| 16 | 93.73% | 3,040 | 4,754 | 1.00% | 1.00% | 1.47% |
| 32 | 76.00% | 2,923 | 8,798 | 19.07% | 18.80% | 4.93% |

`Formatting proxy` is deliberately simple: an unmatched dollar delimiter or unequal counts of
opening and closing braces. It catches visible LaTeX damage but is not a full parser. Its zero rate
at K=1 is not evidence of good formatting: every parsed final response is empty.

The aggregate curve has three regimes:

- K=1--2 collapse before a usable final answer is exposed. K=1 has no visible final in all 1,500
  responses. At K=2, 96.87% hit the token cap and 99.13% have an empty visible final.
- K=3 is a transition regime: accuracy recovers to 62.13%, but 29.07% of final responses are empty
  and successful generations remain unusually long. K=4 sharply recovers to near-native accuracy
  and stable completion behavior.
- K=5--16 form a broad stable region. Accuracy stays between 93.73% and 95.33%, while median
  generation length gradually declines as K increases. K=32 is a second, distinct failure regime:
  its median is short because successful responses are concise, but a roughly 19% nonterminating
  tail drives its mean token count sharply upward and accuracy down to 76%.

## Fair prompt-matched comparison

The primary comparison now uses the same 500 prompt IDs at every K, without conditioning on whether
any model response is correct. Repetitions are first aggregated within each prompt, and the median
full-generation length for that prompt is divided by the corresponding K=8 prompt median. Thus
every point contains exactly the same problems and each problem receives equal weight. K=13--15
having two rather than three runs changes their precision but not their prompt composition or
weighting.

Two fixed secondary cohorts check sensitivity to prompt selection:

- **K=8-solvable:** 477 prompts that K=8 answers correctly in at least two of three repetitions.
  This cohort is defined once using only K=8 and then reused unchanged at every K. Failures at other
  K values remain failures.
- **Common stable successes:** 416 prompts answered correctly in every available repetition at
  every K from 4 through 16. This deliberately outcome-conditioned cohort is useful only for asking
  whether successful reasoning on a universally solved core changes length.

| K | All 500 prompts: token multiplier | Fixed K=8-solvable: token multiplier | Accuracy on fixed K=8-solvable cohort | Common K=4--16 successes: token multiplier |
|---:|---:|---:|---:|---:|
| 1 | 6.57x | 6.76x | 0.00% | 7.51x |
| 2 | 9.57x | 9.71x | 0.07% | 10.31x |
| 3 | 2.08x | 2.08x | 64.50% | 2.10x |
| 4 | 1.29x | 1.29x | 97.83% | 1.29x |
| 5 | 1.14x | 1.14x | 98.67% | 1.15x |
| 6 | 1.09x | 1.09x | 98.67% | 1.08x |
| 7 | 1.03x | 1.03x | 98.81% | 1.03x |
| 8 | 1.00x | 1.00x | 99.58% | 1.00x |
| 9 | 0.99x | 0.98x | 98.32% | 0.98x |
| 10 | 0.97x | 0.96x | 98.67% | 0.95x |
| 11 | 0.93x | 0.93x | 98.95% | 0.93x |
| 12 | 0.92x | 0.92x | 98.18% | 0.92x |
| 13 | 0.93x | 0.92x | 98.11% | 0.92x |
| 14 | 0.92x | 0.91x | 97.80% | 0.90x |
| 15 | 0.90x | 0.90x | 98.32% | 0.89x |
| 16 | 0.88x | 0.88x | 97.69% | 0.87x |
| 32 | 0.85x | 0.84x | 78.83% | 0.83x |

The important result is the agreement among cohorts. From K=4 through K=16, the all-prompt,
K=8-solvable, and common-success token curves are nearly identical. The earlier length trend is
therefore not an artifact of one K contributing a different collection of easy successful prompts.
At every K in that range, the largest separation among the three median multipliers is at most
0.015x.

Conditioning separately at every K did introduce some optimism at the unstable endpoints. The old
both-correct median was 2.01x at K=3 versus 2.08x on the fixed all-prompt cohort, and 0.77x at K=32
versus 0.85x on all prompts. At K=32 this happens because the old subset excludes most capped or
empty generations. Even the new 0.85x median should be read alongside the 19% cap rate: capped
lengths are right-censored at 32,768 tokens, and the median does not represent the expensive tail.
K=1/2 multipliers are similarly descriptive only; their responses overwhelmingly fail to expose a
usable final answer.

## Outcome-conditioned successful responses versus K=8

For historical continuity, this table retains the earlier calculation in which each K uses the
aligned responses correct both at that K and at K=8. It compares behavior on matched successful
examples, but its prompt subset varies with K, so it is now a secondary diagnostic rather than the
headline length curve. The token multiplier is `generated_tokens(K) / generated_tokens(K=8)` per
matched response.

| K | Both correct | Correctness agreement | Median token multiplier | At least 1.5x K=8 | Formatting proxy at K | Formatting proxy at K=8 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 4.67% | -- | -- | -- | -- |
| 2 | 1 | 4.73% | 2.19x | 100.00% | 0.00% | 0.00% |
| 3 | 924 | 65.73% | 2.01x | 75.87% | 8.23% | 0.87% |
| 4 | 1,400 | 97.07% | 1.29x | 31.36% | 1.64% | 0.93% |
| 5 | 1,414 | 98.13% | 1.13x | 16.05% | 0.92% | 0.92% |
| 6 | 1,410 | 97.60% | 1.08x | 10.35% | 1.06% | 0.92% |
| 7 | 1,414 | 98.07% | 1.02x | 7.00% | 1.34% | 0.92% |
| 8 | 1,430 | 100.00% | 1.00x | 0.00% | 0.91% | 0.91% |
| 9 | 1,405 | 97.60% | 0.97x | 4.98% | 1.21% | 0.93% |
| 10 | 1,411 | 97.93% | 0.97x | 3.76% | 0.92% | 0.92% |
| 11 | 1,414 | 98.20% | 0.92x | 3.11% | 1.13% | 0.92% |
| 12 | 1,405 | 97.53% | 0.91x | 3.84% | 1.21% | 0.93% |
| 13 | 935 | 97.90% | 0.91x | 3.32% | 1.28% | 1.07% |
| 14 | 934 | 97.50% | 0.90x | 2.68% | 1.18% | 1.07% |
| 15 | 934 | 97.10% | 0.89x | 2.57% | 1.18% | 1.07% |
| 16 | 1,395 | 96.93% | 0.88x | 2.87% | 1.58% | 0.93% |
| 32 | 1,127 | 78.93% | 0.77x | 3.46% | 5.50% | 0.71% |

The visible answer lengths are nearly unchanged from K=4 through K=16. The systematic change is
the number of total tokens used before and during the visible answer:

- K=3 uses a median 2.01 times as many generated tokens as K=8 on matched correct responses.
- K=4 uses 1.29 times as many. K=5, K=6, and K=7 then converge smoothly toward K=8 at 1.13x,
  1.08x, and 1.02x.
- K=9--16 are modestly shorter than K=8, declining from 0.97x at K=9 to 0.88x at K=16. Their
  accuracy, completion, and formatting rates provide no evidence of a broad coherence loss.
- Correct K=32 responses are even shorter at 0.77x, but this is only the successful mode of a
  bimodal distribution. The nonterminating tail is excluded by the both-correct filter.

## K=3/K=4/K=8 correctness zoom-in

The per-example correctness patterns below use the order K=3/K=4/K=8, with `1` meaning correct:

| Pattern | Responses |
|:--|--:|
| `111` | 914 |
| `011` | 486 |
| `001` | 20 |
| `101` | 10 |
| `110` | 8 |
| `010` | 6 |
| `000` | 56 |

K=4 and K=8 agree on correctness for 1,456/1,500 responses (97.1%). There are 30 responses correct
only at K=8 relative to K=4, and 14 correct only at K=4 relative to K=8. This is consistent with a
small K=4 degradation mixed with ordinary sampling variation, not a broad collapse in mathematical
reasoning.

Subject-level accuracy does not show a clean difficulty-specific K=4 failure. Geometry has the
largest observed K=4 versus K=8 gap (86.18% versus 91.06%), but the other subject gaps are small and
some favor K=4. Level-5 accuracy is 92.04% at K=4 and 91.29% at K=8.

## Qualitative review

The matched examples show four recurring patterns:

1. **Most completed K=4--16 answers are normally coherent.** On straightforward and moderately
   hard problems, these K values provide essentially the same explanation, level of detail, and
   final formatting as K=8. K=4 is usually more expensive in hidden/generated tokens; the overhead
   steadily disappears by K=7.
2. **K=3 can reach the right answer through visibly rougher reasoning.** One correct complex-number
   solution repeatedly states and retracts incorrect values for `2 cos(pi/5)`, emits malformed
   LaTeX such as `\bb`, and then recovers to the correct answer. The matched K=8 response gives the
   same solution directly. Similar but less frequent self-correction appears at K=4.
3. **Very-low-K failures are often generation-state failures rather than concise wrong proofs.** At
   K=3, the dominant failure is an empty visible answer after a long hidden generation. K=4 has rare
   prompt-specific looping: one simple binary-to-octal problem hit the 32k cap with no visible final
   in all three K=4 replicates, while K=3 and K=8 answered it correctly.
4. **K=32 is bimodal, not uniformly incoherent.** Many successful K=32 solutions are coherent and
   shorter than K=8. However, 19.07% of all generations hit the cap and 18.80% expose no visible
   final. Among completed-but-wrong responses inspected manually, the model usually makes ordinary
   coherent mathematical errors rather than producing word salad. Correct K=32 responses also show
   somewhat more malformed LaTeX, particularly missing braces, consistent with the 5.50% matched
   formatting-proxy rate versus 0.71% for the paired K=8 responses.

## Full-token decomposition on common correct prompts

The generated-token measurements above are full completion counts: hidden reasoning plus the
parser-returned worked solution. Although the prediction files do not retain vLLM's separate
`reasoning_content`, the two components can be closely reconstructed. The returned solution was
tokenized with Qwen3's own tokenizer, without added special tokens, and subtracted from the server's
full completion-token count. The residual is hidden reasoning plus a small number of generated
reasoning delimiters, finalization markers, and EOS/boundary tokens.

The decomposition uses the same 416 prompts correct in every available run at every K from 4
through 16. Values below are prompt-balanced means, so the components add exactly to the total.

| K | Full completion | Inferred hidden reasoning + boundaries | Returned worked solution | Returned share |
|---:|---:|---:|---:|---:|
| 4 | 5,225 | 4,625 | 601 | 11.5% |
| 5 | 4,625 | 4,017 | 609 | 13.2% |
| 6 | 4,356 | 3,747 | 609 | 14.0% |
| 7 | 4,154 | 3,543 | 611 | 14.7% |
| 8 | 4,096 | 3,486 | 610 | 14.9% |
| 9 | 3,994 | 3,379 | 616 | 15.4% |
| 10 | 3,958 | 3,347 | 611 | 15.4% |
| 11 | 3,788 | 3,182 | 606 | 16.0% |
| 12 | 3,801 | 3,192 | 610 | 16.0% |
| 13 | 3,767 | 3,160 | 607 | 16.1% |
| 14 | 3,688 | 3,081 | 607 | 16.5% |
| 15 | 3,663 | 3,059 | 604 | 16.5% |
| 16 | 3,638 | 3,032 | 605 | 16.6% |

The returned worked solution is essentially invariant at roughly 600--615 tokens. The full length
change comes almost entirely from hidden reasoning:

- **K=4 to K=8:** the full completion falls by 1,129 mean tokens. Inferred hidden reasoning falls
  by 1,139 tokens, while the returned solution grows by about nine tokens. Thus slightly more than
  100% of the net reduction comes from shorter hidden reasoning.
- **Largest adjacent step, K=4 to K=5:** the full completion falls by 600 tokens. Inferred hidden
  reasoning falls by 608 tokens, while the returned solution grows by eight.
- **K=8 to K=16:** the full completion falls by another 458 tokens. Inferred hidden reasoning
  accounts for 454 tokens, or 99.0% of that change; the returned solution falls by only five.
- **K=4 to K=16 overall:** inferred hidden reasoning falls by 1,592 tokens while the returned
  solution is four to five tokens longer. The latter is effectively unchanged.

The changing returned share, from 11.5% at K=4 to 16.6% at K=16, therefore does not mean answers
become longer. The denominator shrinks because the model uses fewer hidden reasoning tokens as more
experts are active.

## Interpretation

- **K=5--7:** This is the cleanest reduced-K region. Accuracy and visible response quality match
  K=8, completion pathologies are rare, and the generation-length penalty shrinks smoothly as K
  approaches 8. On the fixed all-prompt comparison, a rough routed expert-token proxy is
  `K * token_multiplier / 8`: 0.71 for K=5, 0.81 for K=6, and 0.90 for K=7.
- **K=4:** The high MATH score is representative of user-visible answer quality. There is little
  evidence of a broad coherence loss, although total generation is about 29% longer at the median,
  visible formatting damage is slightly more common, and rare nontermination remains. Its routed
  expert-token proxy is about 0.65 of K=8 on the all-prompt comparison.
- **K=3:** The aggregate score reflects a real degradation. Among responses that finish correctly,
  most final solutions remain readable, but they use about twice as many generated tokens and have
  materially more formatting damage and false starts. Across all responses, failure to transition
  to a visible final answer is the main pathology.
- **K=9--16:** These values are stable rather than noisier or less coherent. Successful generations
  become slightly shorter, although using more experts still raises routed expert-token work above
  K=8; for example K=16's all-prompt proxy is approximately `16 * 0.88 / 8 = 1.76`.
- **K=32:** The lower aggregate score hides a mixture of concise, coherent successes and severe
  nontermination/final-transition failures. Its short median alone is misleading: the mean rises to
  8,798 tokens due to the 32k tail. This is a qualitatively different high-K pathology and not
  evidence that simply adding experts produces uniformly worse prose.

## K=8-reference-scaled replication

The same K=4--16 decomposition was repeated with the K=8-reference-scaled runs. These use three
replicates at every K: the dedicated reference-scaled K=4 runs, the native K=8 runs (which are
identical to K=8 reference scaling), and the expanded reference-scaled sweep for the remaining K
values. This routing condition has 434 prompts correct in every replicate at every K=4--16,
compared with 416 under normalized routing.

| K | Accuracy | Full completion | Inferred hidden reasoning + boundaries | Returned worked solution |
|---:|---:|---:|---:|---:|
| 4 | 94.87% | 4,022 | 3,411 | 611 |
| 5 | 95.33% | 3,983 | 3,360 | 623 |
| 6 | 94.80% | 4,072 | 3,453 | 619 |
| 7 | 94.67% | 4,150 | 3,523 | 627 |
| 8 | 95.33% | 4,219 | 3,598 | 621 |
| 9 | 95.20% | 4,222 | 3,594 | 628 |
| 10 | 95.00% | 4,187 | 3,574 | 613 |
| 11 | 95.33% | 4,211 | 3,594 | 617 |
| 12 | 95.27% | 4,161 | 3,551 | 610 |
| 13 | 95.00% | 4,161 | 3,549 | 612 |
| 14 | 94.73% | 4,137 | 3,524 | 613 |
| 15 | 95.47% | 4,170 | 3,550 | 620 |
| 16 | 94.33% | 4,164 | 3,549 | 615 |

Reference scaling eliminates the normalized sweep's strong reasoning-length gradient. Accuracy
stays in a narrow 94.33--95.47% band, full completions stay within 3,983--4,222 mean tokens, and
the parser-returned solution remains within 610--628 tokens. Prompt-matched median full-generation
ratios relative to K=8 stay between 0.934 and 1.005 across the entire range. In particular,
reference-scaled K=4 is slightly shorter than K=8 rather than 29% longer as in the normalized
analysis.

That last comparison also holds on a stricter shared cohort rather than being an artifact of the
two routing conditions having different common-correct sets. Across the 406 prompts common to both
cohorts, normalized K=4 averages 5,131 full tokens, reference-scaled K=4 averages 3,816, and native
K=8 averages 4,024. Returned solutions are nearly unchanged at 596, 597, and 605 tokens,
respectively; the difference again comes almost entirely from inferred hidden reasoning.

The reference-scaled result therefore strengthens the view that the low-K length penalty is not an
unavoidable consequence of using fewer experts. It appears largely induced by renormalizing the
surviving expert weights. Preserving the native K=8 scale lets K=4 retain both MATH-500 accuracy
and the native model's reasoning-length regime.

## Artifacts

- Sweep plot: `notes/plots/qwen3_math_response_coherence_sweep.png` and `.svg`
- Common-correct token-decomposition plot:
  `notes/plots/qwen3_math_common_correct_token_decomposition.png` and `.svg`
- K=8-reference-scaled common-correct token-decomposition plot:
  `notes/plots/qwen3_math_common_correct_token_decomposition_reference_scaled.png` and `.svg`
- Analysis script: `scripts/adaptive_experts/analyze_qwen_math_coherence.py`
- Plot script: `scripts/adaptive_experts/plot_qwen_math_coherence.py`
- Token-decomposition analysis and plot scripts:
  `scripts/adaptive_experts/analyze_qwen_math_token_decomposition.py` and
  `scripts/adaptive_experts/plot_qwen_math_token_decomposition.py`
- Machine-readable report, including matched comparisons and manual-review candidates:
  `notes/qwen_math_response_coherence.json`
- Machine-readable token decomposition: `notes/qwen_math_token_decomposition.json`
- K=8-reference-scaled run manifest and machine-readable token decomposition:
  `notes/qwen_math_reference_scaled_runs.json` and
  `notes/qwen_math_token_decomposition_reference_scaled.json`
