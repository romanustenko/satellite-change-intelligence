"""Selective-scan (Mamba S6) operator for MambaSCD: compiled kernel on CUDA, PyTorch reference.

The CUDA path calls ``selective_scan_cuda_oflex`` built from ChangeMamba ``kernels/selective_scan``
(commit ``9ce9cec``, Apache-2.0; build recipe in ``docs/research2/PROGRESS.md``, task R2-V1-T03).
The thin autograd wrapper follows ``SelectiveScanOflex`` in ChangeMamba
``changedetection/models/vmamba.py`` (Copyright the ChangeMamba/VMamba authors, Apache-2.0).
The reference recurrence is an independent PyTorch implementation used for CPU tests and as the
numerical oracle for the kernel. Every call records its analytic FLOPs (``flop_registry``)
for compute accounting, because ``FlopCounterMode`` does not see the kernel. Importing this
module has no side effects: the kernel is loaded lazily on first CUDA use, and a missing kernel
on CUDA raises instead of falling back to the (very slow) reference.

Shapes: ``u``/``delta`` ``[batch, channels, length]``; ``A`` ``[channels, state]``; ``B``/``C``
``[batch, groups, state, length]`` with ``channels % groups == 0``; ``D``/``delta_bias``
``[channels]``. The recurrence per channel ``d`` of group ``g = d // (channels // groups)`` is
``h_t = exp(dt_t A_d) h_{t-1} + dt_t B_{g,t} u_t``, ``y_t = <C_{g,t}, h_t> + D_d u_t`` with
``dt = softplus(delta + delta_bias)`` when ``delta_softplus`` is set.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any, Literal, cast

import torch
import torch.nn.functional as F

from satchange.flop_registry import record_analytic_flops

KERNEL_MODULE = "selective_scan_cuda_oflex"
Backend = Literal["auto", "kernel", "reference"]

_kernel_cache: dict[str, ModuleType] = {}


def load_selective_scan_kernel() -> ModuleType:
    """Import the compiled kernel once, or raise with build instructions."""
    cached = _kernel_cache.get(KERNEL_MODULE)
    if cached is not None:
        return cached
    try:
        module = importlib.import_module(KERNEL_MODULE)
    except ImportError as error:
        raise RuntimeError(
            f"CUDA selective scan requires the compiled '{KERNEL_MODULE}' extension "
            "(package selective_scan 0.0.2 from ChangeMamba kernels/selective_scan, built for "
            "torch 2.12.1+cu130 as recorded for R2-V1-T03); it could not be imported"
        ) from error
    _kernel_cache[KERNEL_MODULE] = module
    return module


def selective_scan_kernel_available() -> bool:
    """Return whether the compiled kernel can be imported."""
    try:
        load_selective_scan_kernel()
    except RuntimeError:
        return False
    return True


def _check_shapes(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None,
    delta_bias: torch.Tensor | None,
) -> None:
    if u.dim() != 3 or delta.shape != u.shape:
        raise ValueError(
            f"u and delta must share shape [batch, channels, length], got {tuple(u.shape)} "
            f"and {tuple(delta.shape)}"
        )
    batch, channels, length = u.shape
    if A.dim() != 2 or A.shape[0] != channels:
        raise ValueError(f"A must be [channels={channels}, state], got {tuple(A.shape)}")
    state = A.shape[1]
    for name, tensor in (("B", B), ("C", C)):
        if (
            tensor.dim() != 4
            or tensor.shape[0] != batch
            or tensor.shape[2] != state
            or tensor.shape[3] != length
        ):
            raise ValueError(
                f"{name} must be [batch={batch}, groups, state={state}, length={length}], "
                f"got {tuple(tensor.shape)}"
            )
    if B.shape[1] != C.shape[1] or channels % B.shape[1] != 0:
        raise ValueError(
            f"B and C must share a group count dividing channels={channels}, got "
            f"{B.shape[1]} and {C.shape[1]}"
        )
    for name, vector in (("D", D), ("delta_bias", delta_bias)):
        if vector is not None and tuple(vector.shape) != (channels,):
            raise ValueError(f"{name} must be [channels={channels}], got {tuple(vector.shape)}")


def selective_scan_reference(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    delta_bias: torch.Tensor | None = None,
    *,
    delta_softplus: bool = False,
) -> torch.Tensor:
    """Sequential recurrence in at least float32 (differentiable); output in ``u.dtype``."""
    _check_shapes(u, delta, A, B, C, D, delta_bias)
    channels = u.shape[1]
    groups = B.shape[1]
    work = torch.promote_types(u.dtype, torch.float32)
    dt = delta.to(work)
    if delta_bias is not None:
        dt = dt + delta_bias.to(work)[:, None]
    if delta_softplus:
        dt = F.softplus(dt)
    u32 = u.to(work)
    B32 = B.to(work).repeat_interleave(channels // groups, dim=1)  # [b, d, n, l]
    C32 = C.to(work).repeat_interleave(channels // groups, dim=1)
    decay = torch.exp(dt[:, :, None, :] * A.to(work)[None, :, :, None])  # [b, d, n, l]
    drive = (dt * u32)[:, :, None, :] * B32
    state = u32.new_zeros(decay.shape[:3])
    outputs = []
    for step in range(u.shape[2]):
        state = decay[..., step] * state + drive[..., step]
        outputs.append((state * C32[..., step]).sum(dim=2))
    y = torch.stack(outputs, dim=2)
    if D is not None:
        y = y + D.to(work)[:, None] * u32
    return y.to(u.dtype)


class _SelectiveScanKernel(torch.autograd.Function):
    """Autograd wrapper around the compiled forward/backward kernels."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor | None,
        delta_bias: torch.Tensor | None,
        delta_softplus: bool,
    ) -> torch.Tensor:
        kernel = load_selective_scan_kernel()
        ctx.delta_softplus = delta_softplus
        out, state, *_ = kernel.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, True)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, state)
        return cast(torch.Tensor, out)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        u, delta, A, B, C, D, delta_bias, state = ctx.saved_tensors
        dout = grad_outputs[0]
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *_ = load_selective_scan_kernel().bwd(
            u, delta, A, B, C, D, delta_bias, dout, state, ctx.delta_softplus, 1
        )
        return (
            du,
            ddelta,
            dA,
            dB,
            dC,
            dD if D is not None else None,
            ddelta_bias if delta_bias is not None else None,
            None,
        )


def selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    delta_bias: torch.Tensor | None = None,
    *,
    delta_softplus: bool = False,
    backend: Backend = "auto",
) -> torch.Tensor:
    """Selective scan; ``auto``: kernel on CUDA (required there), reference on CPU."""
    if backend not in ("auto", "kernel", "reference"):
        raise ValueError(f"backend must be 'auto', 'kernel' or 'reference', got {backend!r}")
    _check_shapes(u, delta, A, B, C, D, delta_bias)
    record_analytic_flops(
        "selective_scan",
        needs_backward=torch.is_grad_enabled()
        and any(t is not None and t.requires_grad for t in (u, delta, A, B, C, D, delta_bias)),
        batch=u.shape[0],
        length=u.shape[2],
        channels=u.shape[1],
        state=A.shape[1],
        with_d=D is not None,
    )
    use_kernel = backend == "kernel" or (backend == "auto" and u.is_cuda)
    if not use_kernel:
        return selective_scan_reference(
            u, delta, A, B, C, D, delta_bias, delta_softplus=delta_softplus
        )
    if not u.is_cuda:
        raise ValueError("the selective-scan kernel needs CUDA tensors")
    output = _SelectiveScanKernel.apply(
        u.contiguous(),
        delta.contiguous(),
        A.contiguous(),
        B.contiguous(),
        C.contiguous(),
        None if D is None else D.contiguous(),
        None if delta_bias is None else delta_bias.contiguous(),
        delta_softplus,
    )
    return cast(torch.Tensor, output)


__all__ = [
    "KERNEL_MODULE",
    "load_selective_scan_kernel",
    "selective_scan",
    "selective_scan_kernel_available",
    "selective_scan_reference",
]
