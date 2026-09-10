# Qwen3.6 above-native-K diagnostic results

Gold bands in the plot mark full-attention layers; other layers use DeltaNet.

| Tokens | Layers | raw top-8 mass | raw top-32 mass | ref K32 weight sum | normalized K32 top-8 share | tail/native routed norm | ref local delta | normalized local delta | K32-top8-only delta | native replay control | tail/top-8 cosine |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all_tokens | all_layers | 19.38% | 39.30% | 2.205 | 47.09% | 45.65% | 32.61% | 41.43% | 2.809% | 0.0000% | 0.073 |
| all_tokens | linear_attention | 19.54% | 39.47% | 2.194 | 47.32% | 45.03% | 32.56% | 41.50% | 2.811% | 0.0000% | 0.070 |
| all_tokens | full_attention | 18.89% | 38.80% | 2.241 | 46.39% | 47.49% | 32.75% | 41.22% | 2.800% | 0.0000% | 0.080 |
| prompt | all_layers | 19.49% | 39.50% | 2.203 | 47.15% | 46.10% | 32.83% | 41.37% | 2.819% | 0.0000% | 0.070 |
| prompt | linear_attention | 19.65% | 39.67% | 2.190 | 47.38% | 45.46% | 32.82% | 41.47% | 2.829% | 0.0000% | 0.068 |
| prompt | full_attention | 19.00% | 38.98% | 2.240 | 46.45% | 48.03% | 32.85% | 41.04% | 2.789% | 0.0000% | 0.076 |
| generated | all_layers | 19.11% | 38.84% | 2.212 | 46.93% | 44.56% | 32.10% | 41.57% | 2.784% | 0.0000% | 0.081 |
| generated | linear_attention | 19.27% | 39.01% | 2.201 | 47.16% | 44.01% | 31.96% | 41.55% | 2.770% | 0.0000% | 0.077 |
| generated | full_attention | 18.62% | 38.35% | 2.246 | 46.26% | 46.20% | 32.52% | 41.63% | 2.827% | 0.0000% | 0.091 |

## End-to-end short probes

Worker 0:

| Policy | top-1 match | top-10 overlap | cosine | KL(native || policy) | mean abs logit delta |
|---|---:|---:|---:|---:|---:|
| native_repeat | True | 10/10 | 1.000000 | 0.000000 | 0.00000 |
| keep8 | True | 10/10 | 0.990265 | 0.000981 | 0.24397 |
| reference32 | True | 8/10 | 0.834816 | 0.093199 | 1.02786 |
| normalized32 | True | 7/10 | 0.843186 | 0.044755 | 1.03922 |

Worker 1:

| Policy | top-1 match | top-10 overlap | cosine | KL(native || policy) | mean abs logit delta |
|---|---:|---:|---:|---:|---:|
| native_repeat | True | 10/10 | 1.000000 | 0.000000 | 0.00000 |
| keep8 | True | 10/10 | 0.990265 | 0.000981 | 0.24397 |
| reference32 | True | 8/10 | 0.834816 | 0.093199 | 1.02786 |
| normalized32 | True | 7/10 | 0.843186 | 0.044755 | 1.03922 |

Worker 2:

| Policy | top-1 match | top-10 overlap | cosine | KL(native || policy) | mean abs logit delta |
|---|---:|---:|---:|---:|---:|
| native_repeat | True | 10/10 | 1.000000 | 0.000000 | 0.00000 |
| keep8 | True | 10/10 | 0.990265 | 0.000981 | 0.24397 |
| reference32 | True | 8/10 | 0.834816 | 0.093199 | 1.02786 |
| normalized32 | True | 7/10 | 0.843186 | 0.044755 | 1.03922 |

Worker 3:

| Policy | top-1 match | top-10 overlap | cosine | KL(native || policy) | mean abs logit delta |
|---|---:|---:|---:|---:|---:|
| native_repeat | True | 10/10 | 1.000000 | 0.000000 | 0.00000 |
| keep8 | True | 10/10 | 0.990265 | 0.000981 | 0.24397 |
| reference32 | True | 8/10 | 0.834816 | 0.093199 | 1.02786 |
| normalized32 | True | 7/10 | 0.843186 | 0.044755 | 1.03922 |
