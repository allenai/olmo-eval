# Qwen3.6 above-native-K diagnostic results

Gold bands in the plot mark full-attention layers; other layers use DeltaNet.

| Tokens | Layers | raw top-8 mass | raw top-32 mass | ref K32 weight sum | normalized K32 top-8 share | tail/native routed norm | ref local delta | normalized local delta | keep-8 control delta | tail/top-8 cosine |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all_tokens | all_layers | 22.68% | 42.82% | 2.092 | 50.04% | 40.76% | 29.49% | 39.94% | 2.418% | 0.068 |
| all_tokens | linear_attention | 22.74% | 42.85% | 2.082 | 50.21% | 40.15% | 29.33% | 40.02% | 2.355% | 0.066 |
| all_tokens | full_attention | 22.51% | 42.75% | 2.123 | 49.56% | 42.60% | 29.97% | 39.71% | 2.609% | 0.072 |
| prompt | all_layers | 21.94% | 42.02% | 2.115 | 49.42% | 41.51% | 29.88% | 40.22% | 2.493% | 0.068 |
| prompt | linear_attention | 21.97% | 42.02% | 2.105 | 49.57% | 40.92% | 29.72% | 40.29% | 2.419% | 0.067 |
| prompt | full_attention | 21.85% | 42.02% | 2.145 | 48.99% | 43.28% | 30.39% | 40.00% | 2.717% | 0.072 |
| generated | all_layers | 36.18% | 57.53% | 1.670 | 61.36% | 27.04% | 22.25% | 34.81% | 1.045% | 0.057 |
| generated | linear_attention | 36.69% | 58.01% | 1.652 | 61.85% | 26.01% | 22.23% | 34.96% | 1.178% | 0.051 |
| generated | full_attention | 34.68% | 56.09% | 1.722 | 59.91% | 30.14% | 22.31% | 34.34% | 0.644% | 0.074 |

## End-to-end short probes

Worker 0:

| Policy | top-1 match | top-10 overlap | cosine | KL(native || policy) | mean abs logit delta |
|---|---:|---:|---:|---:|---:|
| native_repeat | True | 10/10 | 1.000000 | 0.000000 | 0.00000 |
| keep8 | True | 8/10 | 0.995108 | 0.000332 | 0.20713 |
| reference32 | True | 6/10 | 0.901790 | 0.042716 | 0.91059 |
| normalized32 | True | 5/10 | 0.904563 | 0.031065 | 0.99484 |
