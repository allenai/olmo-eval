"""Named checkpoint geometries, independent of any inference engine's settings."""

from typing import Any

# Geometry from the measured BF16 Olmo 3.5 TP1 profile (open-instruct PR #2076).
OLMO35_12_5B_SHAPE: dict[str, Any] = {
    "model_type": "olmo3moe",
    "hidden_size": 1024,
    "num_hidden_layers": 16,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "attention_hidden_size": 1024,
    "vocab_size": 100278,
    "tie_word_embeddings": False,
    "n_routed_experts": 512,
    "num_experts_per_tok": 16,
    "moe_intermediate_size": 1024,
    "latent_moe_dim": 512,
    "dense_layers_indices": [0],
    "dense_mlp_intermediate_size": 8192,
    "dense_layers_use_shared_expert": True,
    "shared_expert_intermediate_size": 1024,
    "linear_num_key_heads": 8,
    "linear_num_value_heads": 8,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 256,
    "linear_conv_kernel_dim": 4,
    "number_of_conv_states": 3,
    "layer_types": ["full_attention" if i in (7, 15) else "linear_attention" for i in range(16)],
}


# A model name alone never qualifies an engine profile. Providers register their
# own measured defaults for these shapes and validate their runtime constraints.
MODEL_SHAPES: dict[str, dict[str, Any]] = {
    "olmo3.5-12.5b": {
        **OLMO35_12_5B_SHAPE,
        "architectures": ["Olmo3MoeForCausalLM"],
    },
}
