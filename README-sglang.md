# SGLang generation provider

Select `-o provider.kind=sglang` to use a local SGLang engine with the existing
olmo-eval tasks, generation harness, scoring, prediction saving and metrics.
Install a compatible SGLang runtime separately; this provider does not install
or upgrade CUDA dependencies. Use an isolated environment from inline vLLM.

`provider.max_model_len` maps to `context_length`; `provider.max_concurrency`
maps to `max_running_requests`. Additional engine arguments go under
`provider.kwargs`, including `tp_size`, `mem_fraction_static`, and `random_seed`.
For model plugins, `provider.kwargs.model_registration=olmo_sglang:register`
calls that registration function before constructing the engine. Engine workers
must also be able to import the model package.

Chat requests use the model tokenizer's chat template with a generation prompt.
The provider forwards temperature, top-p, top-k, generation length and string
stops; EOS handling remains with SGLang. It retains sampled-token logprobs,
engine metadata, token IDs and finish reasons. The logprob convention follows
the runtime; set `SGLANG_RETURN_ORIGINAL_LOGPROB=1` before process startup when
unscaled model probabilities are required on the pinned runtime.

Only the batched generation strategy is supported. Continuation likelihood
scoring, prompt truncation and tool requests fail explicitly. Install the
optimized olmo-sglang preset separately when desired; selecting this provider
does not automatically enable model-specific optimization switches.

For top-p evaluations, select `provider.kwargs.sampling_backend=flashinfer`
in a compatible pinned runtime. This uses SGLang's existing nucleus sampler.
The olmo-sglang simple fused sampler applies only to unrestricted sampling;
its narrower coverage does not prevent efficient top-p generation. Preserve
the task's sampling settings when comparing engines.
