# MATH-500 delimiter normalization

`math500:latex_delimiters` is an opt-in correction to answer extraction. It removes
the whole LaTeX delimiter-sizing commands `\left` and `\right` before applying the
existing math extractor, on both reference solutions and model responses.

The reference normalizer removes the unit string `ft` without checking word
boundaries. That turns `\left` into `\le`, so later delimiter cleanup cannot remove
it. For example, a reference answer `\left(3,\frac{\pi}{2}\right)` fails to match
the correct response `(3,\frac{\pi}{2})`. The corrected spec accepts that pair and
continues to reject incorrect coordinates.

The original `math500` spec, its variants, and the shared math extractor retain
their reference behavior. The new spec has a distinct serialized name and pins
MATH-500 to revision `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`. Prompt formatting,
sampling, and the equivalence scorer otherwise inherit the base configuration.
Other legacy normalization behavior is unchanged; this is not a general rewrite
of unit removal or symbolic math grading.

Report corrected-spec results separately from reference-spec results. To compare
engines on an existing evaluation, keep its original scoring policy fixed and
show corrected scoring as a separate diagnostic over the same saved responses.
