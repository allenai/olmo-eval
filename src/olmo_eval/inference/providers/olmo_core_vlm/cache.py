"""Cached decoding support for the multimodal provider.

Without a cache, generating a token re-runs the language model over the whole
sequence. Two kinds of block keep state between decode steps instead:

* softmax attention keeps its keys and values. The stock ``torch``/``flex``
  attention backends that support the bidirectional image-token mask reject KV
  caching, so a cache-capable dense-SDPA subclass is installed at load time,
  driven by explicit per-row RoPE positions;
* Kimi Delta Attention keeps its recurrent state and the last inputs of its
  short convolutions, through a subclass installed the same way.

:func:`enable_kv_cache` installs both and reports whether every block of the
model is covered; callers decode without a cache when it is not.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


def build_decode_attention_mask(
    *,
    seq_len: int,
    pos: int,
    total: int,
    device: Any,
    or_mask: Any = None,
    and_mask: Any = None,
    cache_leftpad: Any = None,
) -> Any:
    """The boolean SDPA mask for one cached decode forward, or ``None`` for plain causal.

    ``seq_len`` query rows sit at absolute positions ``pos .. pos+seq_len-1`` over
    ``total`` cached keys. Causality is expressed against absolute positions;
    ``or_mask`` re-opens key columns (bidirectional image tokens), ``and_mask``
    closes them, and both are left-padded to the key axis when they only cover the
    current step. With left padding, real queries never see pad-slot keys while
    pad-slot queries keep their causal rows so no softmax row is fully masked.
    """
    import torch

    has_leftpad = cache_leftpad is not None
    if seq_len <= 1 and or_mask is None and and_mask is None and not has_leftpad:
        return None

    base = torch.ones(seq_len, total, device=device, dtype=torch.bool).tril(diagonal=pos)

    def _align_keys(mask: Any, pad_value: bool) -> Any:
        mask = mask.to(device=device, dtype=torch.bool)
        missing = total - mask.shape[-1]
        if missing > 0:
            pad = torch.full((*mask.shape[:-1], missing), pad_value, device=device)
            mask = torch.cat([pad, mask], dim=-1)
        return mask

    if or_mask is not None:
        base = base | _align_keys(or_mask, False)
    if and_mask is not None:
        base = base & _align_keys(and_mask, True)
    if has_leftpad:
        leftpad = cache_leftpad.to(device=device, dtype=torch.long)
        key_ok = torch.arange(total, device=device) >= leftpad[:, None]
        q_abs = pos + torch.arange(seq_len, device=device)
        pad_query = q_abs[None, :] < leftpad[:, None]
        base = base & (key_ok[:, None, None, :] | pad_query[:, None, :, None])
    return base


_CACHED_BACKEND_CLS: type | None = None


def _cached_torch_backend_class() -> type:
    """Build (once) a ``TorchAttentionBackend`` subclass that supports KV caching.

    OLMo-core's dense ``torch`` backend is the only one that supports the
    multimodal ``or_mask`` / ``and_mask``, but it rejects KV caching — so
    multimodal decoding would otherwise re-run the LM over the whole sequence
    for every generated token. The ``Attention`` module already carries all the
    cache plumbing (``KVCacheManager``, RoPE ``start_pos`` from
    ``current_position()``); this subclass adds the missing piece: write the
    new K/V into the cache and run SDPA over the cached prefix.

    Scope (asserted): no sliding window, no context parallelism, no
    intra-document masking, no attention sinks. Queries at absolute positions ``pos..pos+T-1``
    attend keys ``0..pos+T-1`` causally; ``or_mask`` / ``and_mask`` (sized
    ``(B, 1, T, T)`` over the current forward's tokens) are aligned to the key
    axis by left-padding, which is exact for the prefill call (``pos == 0``,
    the only call that passes them).

    Variable-length batches are supported through cache left-padding: rows are
    right-aligned in the cache (``kv_cache_manager.cache_leftpad`` holds each
    row's pad length, set by :func:`prepare_kv_caches`) so every row's last
    prompt token lands in the final prefill slot and all rows share one write
    position per decode step. Real queries then must not attend the pad slots
    (their K/V is garbage from pad tokens), while pad-slot queries keep their
    causal rows so no softmax row is fully masked (a fully-masked row turns
    into NaN attention that would poison later layers through the cached K/V).
    """
    global _CACHED_BACKEND_CLS
    if _CACHED_BACKEND_CLS is not None:
        return _CACHED_BACKEND_CLS

    import torch
    import torch.nn.functional as F
    from olmo_core.nn.attention.backend import TorchAttentionBackend, _repeat_kv

    class CachedTorchAttentionBackend(TorchAttentionBackend):
        @classmethod
        def assert_supports_kv_cache(cls) -> None:
            pass

        def forward(  # noqa: C901
            self,
            qkv,
            cu_doc_lens=None,
            cu_doc_lens_q=None,
            cu_doc_lens_k=None,
            max_doc_len=None,
            max_doc_len_q=None,
            max_doc_len_k=None,
            local_k_slice=None,
            kv_cache_manager=None,
            or_mask=None,
            and_mask=None,
            sinks=None,
        ):
            if kv_cache_manager is None:
                # `sinks` exists only in OLMo-core builds whose attention backends take
                # attention-sink logits; pass it through only when set, so older builds work.
                sink_kwargs = {} if sinks is None else {"sinks": sinks}
                return super().forward(
                    qkv,
                    cu_doc_lens=cu_doc_lens,
                    cu_doc_lens_q=cu_doc_lens_q,
                    cu_doc_lens_k=cu_doc_lens_k,
                    max_doc_len=max_doc_len,
                    max_doc_len_q=max_doc_len_q,
                    max_doc_len_k=max_doc_len_k,
                    local_k_slice=local_k_slice,
                    or_mask=or_mask,
                    and_mask=and_mask,
                    **sink_kwargs,
                )

            if isinstance(qkv, torch.Tensor):
                raise RuntimeError(f"'{type(self).__name__}' doesn't support packed QKV")
            if self.window_size != (-1, -1):
                raise RuntimeError(
                    f"'{type(self).__name__}' doesn't support KV caching with sliding windows"
                )
            if any(
                opt is not None
                for opt in (
                    cu_doc_lens,
                    cu_doc_lens_q,
                    cu_doc_lens_k,
                    max_doc_len,
                    max_doc_len_q,
                    max_doc_len_k,
                )
            ):
                raise RuntimeError(
                    f"'{type(self).__name__}' doesn't support intra-document masking"
                )
            if self.cp_enabled:
                raise RuntimeError(
                    f"'{type(self).__name__}' doesn't support KV caching with context parallelism"
                )
            if sinks is not None:
                raise RuntimeError(
                    f"'{type(self).__name__}' doesn't support KV caching with attention sinks"
                )

            q, k, v = qkv
            seq_len = q.shape[1]
            # CPU-side mirror of ``cache_seqlens`` to avoid a GPU sync per layer
            # per step. ``Attention.sdpa`` calls ``update_seqlen(seq_len)`` right
            # after this forward, so advance the mirror by the same amount.
            pos = getattr(kv_cache_manager, "_position_mirror", None)
            if pos is None:
                pos = int(kv_cache_manager.cache_seqlens.item())
            kv_cache_manager._position_mirror = pos + seq_len
            total = pos + seq_len

            k_cache, v_cache = kv_cache_manager.k_cache, kv_cache_manager.v_cache
            if total > k_cache.shape[1]:
                raise RuntimeError(f"KV cache overflow: {total} > allocated {k_cache.shape[1]}")
            k_cache[:, pos:total] = k.to(k_cache.dtype)
            v_cache[:, pos:total] = v.to(v_cache.dtype)
            k_full = k_cache[:, :total].to(q.dtype)
            v_full = v_cache[:, :total].to(q.dtype)

            has_leftpad = getattr(kv_cache_manager, "_has_leftpad", False)
            attn_mask = build_decode_attention_mask(
                seq_len=seq_len,
                pos=pos,
                total=total,
                device=q.device,
                or_mask=or_mask,
                and_mask=and_mask,
                cache_leftpad=kv_cache_manager.cache_leftpad if has_leftpad else None,
            )

            n_rep = self.n_heads // self.n_kv_heads
            k_full = _repeat_kv(k_full, n_rep)
            v_full = _repeat_kv(v_full, n_rep)
            q, k_full, v_full = q.transpose(1, 2), k_full.transpose(1, 2), v_full.transpose(1, 2)
            att = F.scaled_dot_product_attention(
                q,
                k_full,
                v_full,
                attn_mask=attn_mask,
                dropout_p=self.dropout_p,
                is_causal=False,
                scale=self.scale,
            )
            return att.transpose(1, 2).contiguous()

    _CACHED_BACKEND_CLS = CachedTorchAttentionBackend
    return CachedTorchAttentionBackend


def _lm_mixers(model: Any) -> list[Any]:
    return [getattr(block, "attention", None) for block in model.lm.blocks.values()]


def _is_kda(mixer: Any) -> bool:
    return any(cls.__name__ == "KimiDeltaAttention" for cls in type(mixer).__mro__)


def _lm_attention_modules(model: Any) -> list[Any]:
    """The LM's softmax-attention modules (the ones that take a KV cache)."""
    return [m for m in _lm_mixers(model) if m is not None and not _is_kda(m)]


def _lm_kda_modules(model: Any) -> list[Any]:
    return [m for m in _lm_mixers(model) if m is not None and _is_kda(m)]


def causal_conv_tail(conv: Any, window: Any, steps: int) -> Any:
    """The outputs of a depthwise causal convolution for the last ``steps`` positions of ``window``.

    ``window`` holds the ``kernel_size - 1`` inputs before those positions followed by the
    positions' own inputs, ``(batch, kernel_size - 1 + steps, channels)``, which is all a
    causal convolution reads for them.
    """
    import torch.nn.functional as F

    weight = conv.weight.to(window.dtype)
    bias = None if conv.bias is None else conv.bias.to(window.dtype)
    out = F.conv1d(window.transpose(1, 2), weight, bias, groups=conv.hidden_size)
    out = out.transpose(1, 2)[:, -steps:]
    if conv.activation is not None:
        out = F.silu(out)
    return out


_CACHED_KDA_CLS: type | None = None


def _cached_kda_class() -> type:
    """Build (once) a ``KimiDeltaAttention`` subclass that decodes from its recurrent state.

    Kimi Delta Attention is a recurrence: each head keeps a ``(K, V)`` state matrix that
    every token decays, erases from and writes to, after a short causal convolution over
    the token's own q/k/v projections. Without a cache the eval re-runs the LM over the
    whole sequence for every generated token. With ``_decode`` set (by
    :func:`prepare_kv_caches`) this subclass instead keeps, per batch row, the state
    after the last token seen and each convolution's last ``kernel_size - 1`` inputs:

    * the first forward (the prefill) runs the usual chunked kernel over the prompt and
      keeps the final state. Rows are left-padded, so every row's last prompt token is
      the last position; pad positions are made inert (zero convolution inputs, zero
      write strength, zero decay input), which leaves a row's state exactly zero until
      its first real token, as at the start of an unpadded sequence;
    * later forwards (decode steps) continue from the kept state.

    Rounding differs from a full re-forward (the chunked kernel groups tokens in blocks
    of 64 from the padded start, and a step's convolution runs through ``F.conv1d``),
    so logits agree to float tolerance, not bitwise.
    """
    global _CACHED_KDA_CLS
    if _CACHED_KDA_CLS is not None:
        return _CACHED_KDA_CLS

    import torch
    import torch.nn.functional as F
    from olmo_core.nn.attention.flash_linear_attn_api import dispatch_chunk_kda
    from olmo_core.nn.attention.kda import KimiDeltaAttention

    class CachedKimiDeltaAttention(KimiDeltaAttention):
        _decode: Any = None

        def forward(self, x, cu_doc_lens=None, **kwargs):  # type: ignore[override]
            decode = self._decode
            if decode is None:
                return super().forward(x, cu_doc_lens=cu_doc_lens, **kwargs)
            if cu_doc_lens is not None or self.cp_enabled:
                raise RuntimeError(
                    "Cached KDA decoding supports neither intra-document masking nor "
                    "context parallelism"
                )
            batch_size, seq_len, _ = x.shape
            q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)
            raw_decay = self.f_proj_2(self.f_proj_1(x))
            beta = self.w_b(x).float().sigmoid()
            if self.allow_neg_eigval:
                beta = beta * 2.0

            tail = self.q_conv1d.kernel_size[0] - 1
            prefill = decode.state is None
            if prefill:
                leftpad = decode.leftpad
                if leftpad is not None and bool((leftpad > 0).any()):
                    pad = (
                        torch.arange(seq_len, device=x.device)[None, :]
                        < leftpad.to(device=x.device, dtype=torch.long)[:, None]
                    )[:, :, None]
                    q = q.masked_fill(pad, 0.0)
                    k = k.masked_fill(pad, 0.0)
                    v = v.masked_fill(pad, 0.0)
                    raw_decay = raw_decay.masked_fill(pad, 0.0)
                    beta = beta.masked_fill(pad, 0.0)
                history = [
                    F.pad(t, (0, 0, max(tail - seq_len, 0), 0))[:, -tail:] if tail else None
                    for t in (q, k, v)
                ]
                q, k, v = self.q_conv1d(x=q), self.k_conv1d(x=k), self.v_conv1d(x=v)
            else:
                windows = [
                    t if h is None else torch.cat([h.to(t.dtype), t], dim=1)
                    for h, t in zip(decode.conv, (q, k, v), strict=True)
                ]
                history = [w[:, -tail:] if tail else None for w in windows]
                q = causal_conv_tail(self.q_conv1d, windows[0], seq_len)
                k = causal_conv_tail(self.k_conv1d, windows[1], seq_len)
                v = causal_conv_tail(self.v_conv1d, windows[2], seq_len)
            decode.conv = history

            q = q.view(batch_size, seq_len, -1, self.head_k_dim)
            k = k.view(batch_size, seq_len, -1, self.head_k_dim)
            v = v.view(batch_size, seq_len, -1, self.head_v_dim)
            raw_decay = raw_decay.view(batch_size, seq_len, -1, self.head_k_dim)
            o, state = dispatch_chunk_kda(
                q=q,
                k=k,
                v=v,
                g=raw_decay,
                beta=beta,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                initial_state=decode.state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_experimental_kernels=self.use_experimental_kernels,
            )
            decode.state = state
            output_gate = self.g_proj_2(self.g_proj_1(x)).view(
                batch_size, seq_len, self.n_v_heads, self.head_v_dim
            )
            return self.w_out(self.o_norm(o, output_gate).view(batch_size, seq_len, -1)).view(
                batch_size, seq_len, self.d_model
            )

    _CACHED_KDA_CLS = CachedKimiDeltaAttention
    return CachedKimiDeltaAttention


# Largest relative deviation from the full forward at which state decoding is trusted. Rounding
# alone stays well below it; a state the module's forward no longer matches is far above.
_KDA_DECODE_TOLERANCE = 0.05


def kda_state_decoding_error(kda: Any, *, length: int = 80, pad: int = 7, steps: int = 5) -> float:
    """How far ``kda`` decoded from kept state is from its full forward, on random input.

    Two rows are decoded the way the provider decodes a batch: a prefill, the second row
    left-padded by ``pad``, then ``steps`` single-token forwards. Each row is compared with
    the module's own full forward over its unpadded tokens.

    :returns: The largest relative L2 deviation over the two rows.
    """
    import torch

    weight = kda.w_q.weight
    x = torch.randn(2, length, kda.d_model, generator=torch.Generator().manual_seed(0))
    x = x.to(device=weight.device, dtype=weight.dtype)
    kept = kda._decode
    with torch.no_grad():
        try:
            kda._decode = None
            references = [kda(x[:1])[0].float(), kda(x[1:, pad:])[0].float()]
            kda._decode = SimpleNamespace(
                leftpad=torch.tensor([0, pad], device=weight.device), state=None, conv=None
            )
            prefill = length - steps
            parts = [kda(x[:, :prefill])]
            parts += [kda(x[:, position : position + 1]) for position in range(prefill, length)]
        finally:
            kda._decode = kept
    decoded = torch.cat(parts, dim=1).float()
    errors = [
        (decoded[row, start:] - reference).norm() / reference.norm().clamp_min(1e-12)
        for row, (reference, start) in enumerate(zip(references, (0, pad), strict=True))
    ]
    return float(max(errors))


_CACHED_ATTENTION_CLS: dict[type, type] = {}


def _cached_attention_class(base: type) -> type:
    """A subclass of the attention module ``base`` whose Scalable-Softmax works with a KV cache.

    Scalable-Softmax multiplies each query by ``log(n)`` times a learned per-head scale,
    where ``n`` is the number of tokens the query can see. OLMo-core derives ``n`` from
    the position inside the current forward and so rejects KV caching, where a decode
    step's forward holds one token. Here ``n`` comes from the cache instead: the
    query's cache slot minus the row's left-pad, plus one.
    """
    cached = _CACHED_ATTENTION_CLS.get(base)
    if cached is not None:
        return cached

    import torch

    class CachedAttention(base):  # type: ignore[misc, valid-type]
        def _apply_scalable_softmax(self, q, cu_doc_lens):
            manager = self.kv_cache_manager
            if not self.scalable_softmax or manager is None:
                return super()._apply_scalable_softmax(q, cu_doc_lens)
            if cu_doc_lens is not None:
                raise NotImplementedError(
                    "Cached Scalable-Softmax doesn't support intra-document masking"
                )
            pos = getattr(manager, "_position_mirror", None)
            if pos is None:
                pos = int(manager.cache_seqlens.item())
            slots = pos + torch.arange(q.shape[1], device=q.device)
            visible = slots[None, :] + 1
            if getattr(manager, "_has_leftpad", False):
                visible = (
                    visible - manager.cache_leftpad.to(device=q.device, dtype=torch.long)[:, None]
                )
            visible = visible.clamp_min(1).expand(q.shape[0], -1)  # pad-slot queries: any value
            scale = visible.log().to(q.dtype).unsqueeze(-1)
            scale = scale * self.ssmax_scale.to(q.dtype).view(1, 1, -1)
            return q * scale.unsqueeze(-1)

    CachedAttention.__name__ = f"Cached{base.__name__}"
    _CACHED_ATTENTION_CLS[base] = CachedAttention
    return CachedAttention


def enable_kv_cache(model: Any) -> bool:
    """Make every LM block decodable from cached state.

    Softmax-attention blocks get the cached dense ``torch`` backend; Kimi Delta
    Attention blocks get the state-keeping subclass. Returns ``False`` (leaving the
    model untouched) when any block has no sequence mixer, an attention block uses a
    different backend or a sliding window, or the KDA blocks run under context
    parallelism or do not reproduce their full forward when decoded from state —
    callers should then fall back to no-cache decoding.
    """
    from olmo_core.nn.attention.backend import TorchAttentionBackend

    cached_cls = _cached_torch_backend_class()
    mixers = _lm_mixers(model)
    if any(mixer is None for mixer in mixers):
        return False
    kda_modules = _lm_kda_modules(model)
    attentions = _lm_attention_modules(model)
    for attention in attentions:
        backend = getattr(attention, "backend", None)
        if backend is None or type(backend) not in (TorchAttentionBackend, cached_cls):
            return False
        if backend.window_size != (-1, -1):
            return False
    cached_kda_cls = None
    if kda_modules:
        try:
            cached_kda_cls = _cached_kda_class()
        except ImportError:
            return False
        for kda in kda_modules:
            if (
                getattr(kda, "cp_enabled", False)
                or len({conv.kernel_size[0] for conv in (kda.q_conv1d, kda.k_conv1d, kda.v_conv1d)})
                != 1
            ):
                return False
        if not _kda_decodes_from_state(kda_modules[0], cached_kda_cls):
            return False
    for attention in attentions:
        attention.backend.__class__ = cached_cls
        if getattr(attention, "scalable_softmax", False):
            base = type(attention)
            if base not in _CACHED_ATTENTION_CLS.values():
                attention.__class__ = _cached_attention_class(base)
    for kda in kda_modules:
        kda.__class__ = cached_kda_cls
    return True


def _kda_decodes_from_state(kda: Any, cached_kda_cls: type) -> bool:
    """Whether ``cached_kda_cls`` reproduces this OLMo-core build's ``kda`` forward.

    The cached class restates the module's forward, so it is checked against the module
    itself before it is trusted: a build whose forward has moved on fails here and the
    model decodes without a cache instead of from a wrong state.
    """
    original = kda.__class__
    kda.__class__ = cached_kda_cls
    try:
        error = kda_state_decoding_error(kda)
    except Exception:
        logger.warning("Kimi Delta Attention could not be decoded from state", exc_info=True)
        return False
    finally:
        kda.__class__ = original
    if not error <= _KDA_DECODE_TOLERANCE:
        logger.warning(
            "Kimi Delta Attention decoded from state deviates from its full forward by %.3g "
            "(tolerance %.3g): this OLMo-core build's forward differs from the one the cached "
            "decode was written for",
            error,
            _KDA_DECODE_TOLERANCE,
        )
        return False
    logger.info("Kimi Delta Attention state decoding matches the full forward to %.2g", error)
    return True


_EXPLICIT_POSITION_KV_MANAGER_CLS: type | None = None


def _explicit_position_kv_manager_class() -> type:
    """Build (once) a ``KVCacheManager`` subclass for explicit-position decoding.

    ``Attention.forward`` derives the RoPE ``start_pos`` from
    ``kv_cache_manager.current_position()``, but RoPE forbids combining
    ``start_pos`` with explicit ``position_ids``. The decode loop drives RoPE
    entirely through per-row ``position_ids`` (required for variable-length
    batches, where each row sits at a different absolute position), so this
    manager reports no position of its own.
    """
    global _EXPLICIT_POSITION_KV_MANAGER_CLS
    if _EXPLICIT_POSITION_KV_MANAGER_CLS is not None:
        return _EXPLICIT_POSITION_KV_MANAGER_CLS

    from olmo_core.nn.attention.kv_cache import KVCacheManager

    class ExplicitPositionKVCacheManager(KVCacheManager):
        def current_position(self) -> Any:
            return None

    _EXPLICIT_POSITION_KV_MANAGER_CLS = ExplicitPositionKVCacheManager
    return ExplicitPositionKVCacheManager


def prepare_kv_caches(
    model: Any,
    batch_size: int,
    max_seq_len: int,
    leftpad: torch.Tensor | None = None,
) -> None:
    """Initialize (or reset) the decode state of every LM block.

    Attention blocks get an empty KV cache and Kimi Delta Attention blocks an empty
    recurrent state; the next forward is the prefill.

    :param leftpad: Per-row left-pad lengths ``(batch_size,)`` for
        variable-length batches; rows are right-aligned, the cached backend masks
        the pad slots out of every real query's row, and recurrent blocks ignore
        the pad positions.
    """
    manager_cls = _explicit_position_kv_manager_class()
    has_leftpad = leftpad is not None and bool((leftpad > 0).any())
    for attention in _lm_attention_modules(model):
        manager = attention.kv_cache_manager
        if isinstance(manager, manager_cls):
            manager.reset(batch_size, max_seq_len)
        else:
            manager = manager_cls(
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                num_kv_heads=attention.n_kv_heads,
                head_dim=attention.head_dim,
                device=attention.w_k.weight.device,
            )
            attention.kv_cache_manager = manager
        if leftpad is not None:
            manager.cache_leftpad.copy_(leftpad)
        manager._has_leftpad = has_leftpad
        manager._position_mirror = 0
    for kda in _lm_kda_modules(model):
        kda._decode = SimpleNamespace(leftpad=leftpad, state=None, conv=None)


def free_kv_caches(model: Any) -> None:
    """Drop every LM block's decode state so subsequent forwards run uncached."""
    for attention in _lm_attention_modules(model):
        attention.kv_cache_manager = None
    for kda in _lm_kda_modules(model):
        kda._decode = None


def rope_buffers(model: Any, max_seq_len: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Full-length RoPE sin/cos tables covering positions ``0..max_seq_len - 1``.

    Explicit-``position_ids`` RoPE gathers rows from the sin/cos tables, but
    ``RotaryEmbedding.forward`` sizes its internal tables from the current
    forward's ``seq_len`` (1 during decode) when ``start_pos`` is absent —
    far short of the decode positions. The tables must therefore be passed in
    explicitly; this builds them once per batch. The Transformer broadcasts one
    buffer pair to every block, so all blocks must share one RoPE config
    (asserted here; per-position table rows do not depend on table length, so
    these buffers match what shorter internal tables would hold).
    """
    ropes = [
        rope
        for attention in _lm_attention_modules(model)
        if (rope := getattr(attention, "rope", None)) is not None
    ]
    if not ropes:
        # No block reads positions (recurrent blocks, attention without RoPE).
        return None, None
    configs = {
        (
            type(rope).__name__,
            getattr(rope, "theta", None),
            getattr(rope, "rotary_dim", None),
            repr(getattr(rope, "scaling", None)),
        )
        for rope in ropes
    }
    if len(configs) != 1:
        raise ValueError(
            f"LM blocks use differing RoPE configs ({sorted(configs)}); batched "
            "explicit-position decoding requires one shared config"
        )
    buffers = ropes[0].get_buffers(max_seq_len, next(iter(model.lm.parameters())).device)
    if buffers.pos_sin is None or buffers.pos_cos is None:
        raise ValueError(
            f"{type(ropes[0]).__name__} does not expose sin/cos RoPE buffers; batched "
            "explicit-position decoding requires RotaryEmbedding"
        )
    return buffers.pos_sin, buffers.pos_cos
