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
engine metadata, token IDs and finish reasons. The logprob convention follows
the runtime; set `SGLANG_RETURN_ORIGINAL_LOGPROB=1` before process startup when
unscaled model probabilities are required on the runtime. Set this before any
SGLang imports: some runtime versions cache the flag at import time. The provider
records the environment flag in request traces and output metadata; the flag
alone is not a numerical parity guarantee. Generation samples and their token IDs
are preserved separately from the top-ranked alternatives.

Only the batched generation strategy is supported. Continuation likelihood
scoring, prompt truncation, automatic fitting of generation length to the context
window, image requests and tool requests fail explicitly. Aborted generation or
missing sampled-token probabilities raise errors rather than becoming scored
answers. Install the
optimized olmo-sglang preset separately when desired; selecting this provider
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
