# SGLang generation provider

Select `-o provider.kind=sglang` to use a local SGLang engine with the existing
olmo-eval tasks, generation harness, scoring, prediction saving and metrics.
Install a compatible SGLang runtime separately; this provider does not install
or upgrade CUDA dependencies. Use an isolated environment from inline vLLM.

`provider.max_model_len` maps to `context_length`; `provider.max_concurrency`
maps to `max_running_requests`. Use `provider.kwargs.tensor_parallel_size` for
tensor parallelism so olmo-eval's GPU planner sees the allocation; the provider
maps it to SGLang's `tp_size`. The worker also forwards its allocated GPU count.
Additional engine arguments go under `provider.kwargs`, including
`mem_fraction_static` and `random_seed`. `provider.force_download` uses the
shared Hugging Face cache-refresh helper.
For model plugins, `provider.kwargs.model_registration=olmo_sglang:register`
calls that registration function before constructing the engine. Engine workers
must also be able to import the model package.

Chat requests use the model tokenizer's chat template with a generation prompt.
The provider forwards temperature, top-p, top-k, generation length and string
stops; EOS handling remains with SGLang. It retains sampled-token logprobs,
engine metadata, token IDs and finish reasons. The provider defaults
`SGLANG_RETURN_ORIGINAL_LOGPROB=1` before model registration or engine imports,
so sampled-token logprobs and `sum_logits` use raw model probabilities rather
than temperature-scaled probabilities. An explicitly conflicting flag or an
already-imported sampler with the other convention fails with a restart message.
Set the flag before importing SGLang in applications that initialize it elsewhere.
The enforced convention is recorded in request traces and output metadata;
it does not imply cross-engine numerical parity. Generation samples and their token IDs
are preserved separately from the top-ranked alternatives. Finish reasons use
the harness's `stop`/`length` strings; `sglang_finish_reason` in returned metadata
retains the engine's original matched-token or length detail.

Only the batched generation strategy is supported. Continuation likelihood
scoring, prompt truncation, automatic fitting of generation length to the context
window and tool requests fail explicitly. Image requests require explicit
multimodal enablement. Aborted generation or missing sampled-token probabilities
raise errors rather than becoming scored answers. Install the optimized
olmo-sglang preset separately when desired; selecting this provider
does not automatically enable model-specific optimization switches.

For top-p evaluations, select `provider.kwargs.sampling_backend=flashinfer`
in a compatible pinned runtime. This uses SGLang's existing nucleus sampler.
The olmo-sglang simple fused sampler applies only to unrestricted sampling;
its narrower coverage does not prevent efficient top-p generation. Preserve
the task's sampling settings when comparing engines.

For a small generation trial in an environment where compatible SGLang and
olmo-eval are already installed:

```bash
SGLANG_RETURN_ORIGINAL_LOGPROB=1 uv run --no-sync olmo-eval run \
    --harness default -o provider.kind=sglang \
    -o provider.kwargs.tensor_parallel_size=1 \
    -m /path/to/model -t gsm8k -o limit=5 --no-upload
```

For remote jobs, provide a compatible runtime through the image or the existing
`provider.package` / `provider.dependencies` settings. There is no default
SGLang dependency extra: selecting this backend does not replace CUDA, PyTorch,
or Transformers in an existing runtime. Keep SGLang and inline vLLM in separate
environments when their dependency pins differ.

The provider delegates sampling to SGLang. It does not implement a sampler or
promise identical random draws to another backend for the same seed. Keep task
definitions and scorer revisions fixed when comparing providers, and record the
runtime and model-plugin revisions alongside the olmo-eval commit.

Equal weights and seeds also do not fix every numerical choice within an engine.
Autotuned kernels can select different launch configurations across processes
and produce different floating-point results. When investigating repeatability,
record the chosen kernel configurations and cache policy where available, along
with the runtime settings. Compare token IDs and logprobs on identical input
histories: after generated tokens differ, later logprobs describe different
conditional distributions. Repeated accuracy measurements and controlled
numerical checks answer different questions; neither substitutes for the other.


## Image requests

Set `provider.kwargs.enable_multimodal=true` to accept `LMRequest.images`.
The provider resolves olmo-eval's lazy image references inside the inference
worker, preserves their order, and forwards one image list per engine request.
Repeated samples reuse the resolved images; mixed text/image batches use empty
image slots for text requests. A declared image request that resolves to no
images fails. Tasks and scoring code are unchanged.

With the default `provider.kwargs.prompt_format=chat`, the engine's model
processor renders structured image/text messages. Images attach to the first
user turn, after any system turn. Tokenization does not add a second set of
special tokens to that rendered template. Put images in `LMRequest.images`,
not embedded media objects in message content.

For checkpoints trained on a plain document stream, explicitly set
`provider.kwargs.prompt_format=document`. The model processor must implement
`document_prompt_ids(text, num_images=...)`; the Olmo vision processor provides
this interface. It owns the document boundary and image placeholders, while
SGLang owns image expansion and preprocessing. Later nonempty turns are encoded
separately with a leading space, matching Core's document layout. Roles become
plain text in this mode. Do not set `add_bos_token`: the processor owns BOS/EOS.
The provider does not infer the prompt format from the task name or inherited
chat template.

For example, with a complete Olmo vision checkpoint and the qualified model
plugin installed:

```bash
SGLANG_RETURN_ORIGINAL_LOGPROB=1 uv run --no-sync olmo-eval run \
  --harness default -o provider.kind=sglang \
  -o provider.kwargs.model_registration=olmo_sglang:register \
  -o provider.kwargs.enable_multimodal=true \
  -o provider.kwargs.prompt_format=document \
  -o provider.kwargs.tensor_parallel_size=1 \
  -m /path/to/restored-vision-checkpoint \
  -t olmocr_bench -o prompt_templates=none \
  -o system_prompt_style=style_and_length_v2 -o limit=4 -o max_tokens=512 \
  --no-upload
```

This is a capped integration trial, not a benchmark score. The task still owns
its OCR style tag, dataset revision, renderer and official scoring. The provider
owns transport and model prompt layout. Use the checkpoint/runtime's qualified
context, memory and attention settings when launching.

Returned `LMOutput` objects retain sampled-token logprobs and token IDs, plus
the unexpanded input token IDs, resolved image count and prompt format for
multimodal runs. The standard predictions JSONL saves generation text, finish
reasons and logprob summaries; it does not serialize the full per-token arrays
or arbitrary provider metadata. Numerical replay tools must save those returned
fields explicitly. Request traces identify the selected prompt format without
loading image pixels.
Generic multimodal chat transport has CPU contract tests; native GPU qualification
must be recorded per model/processor. Text-only settings retain the original
provider behavior unless multimodal mode is explicitly enabled.
