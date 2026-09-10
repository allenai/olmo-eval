import torch

from olmo_eval.compat.vllm_gptoss_topk import (
    make_non_power_of_two_dispatch,
    reference_scaled_topk,
)


def test_non_power_of_two_k_uses_torch_fallback() -> None:
    calls: list[tuple[str, tuple[object, ...]]] = []

    def triton_topk(*args: object) -> str:
        calls.append(("triton", args))
        return "triton"

    def torch_topk(*args: object) -> str:
        calls.append(("torch", args))
        return "torch"

    dispatch = make_non_power_of_two_dispatch(triton_topk, torch_topk)

    assert dispatch("x", 3) == "torch"
    assert calls == [("torch", ("x", 3, True, 1, None, None))]


def test_power_of_two_k_keeps_triton_path() -> None:
    calls: list[tuple[object, ...]] = []

    def triton_topk(*args: object) -> str:
        calls.append(args)
        return "triton"

    dispatch = make_non_power_of_two_dispatch(triton_topk, lambda *args: "torch")

    assert dispatch("x", 4) == "triton"
    assert calls == [("x", 4, True, 1, None, None, False)]


def test_all_gather_keeps_triton_path() -> None:
    calls: list[tuple[object, ...]] = []

    def triton_topk(*args: object) -> str:
        calls.append(args)
        return "triton"

    dispatch = make_non_power_of_two_dispatch(triton_topk, lambda *args: "torch")

    assert dispatch("x", 3, all_gather=True) == "triton"
    assert calls == [("x", 3, True, 1, None, None, True)]


def test_reference_scaled_topk_preserves_native_top_four_weights() -> None:
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0, 0.0, -1.0]])

    native_weights, native_ids = reference_scaled_topk(
        logits,
        top_k=4,
        reference_k=4,
    )
    expanded_weights, expanded_ids = reference_scaled_topk(
        logits,
        top_k=6,
        reference_k=4,
    )

    torch.testing.assert_close(native_weights, torch.softmax(logits[:, :4], dim=-1))
    torch.testing.assert_close(expanded_weights[:, :4], native_weights)
    torch.testing.assert_close(native_ids, torch.tensor([[0, 1, 2, 3]]))
    torch.testing.assert_close(expanded_ids, torch.tensor([[0, 1, 2, 3, 4, 5]]))
    assert expanded_weights.sum().item() > 1


def test_reference_scaled_topk_drops_mass_below_reference_k() -> None:
    logits = torch.tensor([[0.0, 3.0, 1.0, 2.0, -1.0]])

    weights, ids = reference_scaled_topk(
        logits,
        top_k=2,
        reference_k=4,
    )
    expected_reference = torch.softmax(torch.tensor([[3.0, 2.0, 1.0, 0.0]]), dim=-1)

    torch.testing.assert_close(ids, torch.tensor([[1, 3]]))
    torch.testing.assert_close(weights, expected_reference[:, :2])
    assert weights.sum().item() < 1
