"""Tests for the selective-scan operator (reference on CPU, compiled kernel opt-in on CUDA)."""

import math
import os

import pytest
import torch
import torch.nn.functional as F

from satchange.models.selective_scan import (
    selective_scan,
    selective_scan_kernel_available,
    selective_scan_reference,
)


def _inputs(
    *,
    batch: int = 2,
    channels: int = 4,
    groups: int = 2,
    state: int = 3,
    length: int = 5,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)

    def draw(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, dtype=torch.float64).to(device, dtype)

    return {
        "u": draw(batch, channels, length),
        "delta": draw(batch, channels, length) * 0.5,
        "A": -torch.exp(draw(channels, state)),
        "B": draw(batch, groups, state, length),
        "C": draw(batch, groups, state, length),
        "D": draw(channels),
        "delta_bias": draw(channels) * 0.1,
    }


def _naive(values: dict[str, torch.Tensor], *, delta_softplus: bool) -> torch.Tensor:
    u, A, B, C, D = values["u"], values["A"], values["B"], values["C"], values["D"]
    dt = values["delta"] + values["delta_bias"][:, None]
    if delta_softplus:
        dt = F.softplus(dt)
    batch, channels, length = u.shape
    per_group = channels // B.shape[1]
    out = torch.zeros_like(u, dtype=torch.float64)
    for b in range(batch):
        for d in range(channels):
            g = d // per_group
            for n in range(A.shape[1]):
                h = 0.0
                for t in range(length):
                    step = float(dt[b, d, t])
                    h = math.exp(step * float(A[d, n])) * h + step * float(B[b, g, n, t]) * float(
                        u[b, d, t]
                    )
                    out[b, d, t] += float(C[b, g, n, t]) * h
            out[b, d] += float(D[d]) * u[b, d].double()
    return out


@pytest.mark.parametrize("delta_softplus", [False, True])
def test_reference_matches_scalar_recurrence(delta_softplus: bool) -> None:
    values = _inputs()
    expected = _naive(values, delta_softplus=delta_softplus)
    actual = selective_scan_reference(**values, delta_softplus=delta_softplus)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.double(), expected, rtol=1e-5, atol=1e-5)


def test_reference_without_optional_terms_and_auto_backend_on_cpu() -> None:
    values = _inputs(groups=1)
    values["D"] = torch.zeros_like(values["D"])
    values["delta_bias"] = torch.zeros_like(values["delta_bias"])
    expected = selective_scan_reference(**values)
    del values["D"], values["delta_bias"]
    torch.testing.assert_close(selective_scan_reference(**values), expected)
    torch.testing.assert_close(selective_scan(**values), expected)


def test_reference_gradients_are_correct() -> None:
    values = {
        key: tensor.requires_grad_(True)
        for key, tensor in _inputs(batch=1, channels=2, groups=1, state=2, length=3,
                                   dtype=torch.float64).items()
    }

    def function(*tensors: torch.Tensor) -> torch.Tensor:
        named = dict(zip(values, tensors, strict=True))
        return selective_scan_reference(**named, delta_softplus=True)

    assert torch.autograd.gradcheck(function, tuple(values.values()))


def test_shape_and_backend_validation() -> None:
    values = _inputs()
    with pytest.raises(ValueError, match="A must be"):
        selective_scan_reference(**{**values, "A": values["A"][:-1]})
    with pytest.raises(ValueError, match="B must be"):
        selective_scan_reference(**{**values, "B": values["B"][..., :-1]})
    with pytest.raises(ValueError, match="group count"):
        selective_scan_reference(**{**values, "B": values["B"][:, :1]})
    with pytest.raises(ValueError, match="D must be"):
        selective_scan_reference(**{**values, "D": values["D"][:-1]})
    with pytest.raises(ValueError, match="backend"):
        selective_scan(**values, backend="triton")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="CUDA tensors"):
        selective_scan(**values, backend="kernel")


def _cuda_device() -> torch.device:
    device_name = os.environ.get("SATCHANGE_TEST_CUDA_DEVICE")
    if not device_name or not torch.cuda.is_available():
        pytest.skip("Requires CUDA and explicit SATCHANGE_TEST_CUDA_DEVICE allocation.")
    if not selective_scan_kernel_available():
        pytest.fail("SATCHANGE_TEST_CUDA_DEVICE is set but the selective-scan kernel is missing")
    return torch.device(device_name)


def test_cuda_kernel_matches_reference_forward_and_backward() -> None:
    device = _cuda_device()
    cpu_values = _inputs(batch=2, channels=8, groups=4, state=16, length=257, seed=3)
    gradients = {}
    outputs = {}
    for backend, target in (("reference", "cpu"), ("kernel", device)):
        values = {
            key: tensor.detach().to(target).requires_grad_(True)
            for key, tensor in cpu_values.items()
        }
        output = selective_scan(**values, delta_softplus=True, backend=backend)  # type: ignore[arg-type]
        weight = torch.linspace(-1, 1, output.numel(), dtype=output.dtype, device=output.device)
        (output.flatten() * weight).sum().backward()
        outputs[backend] = output.detach().cpu()
        gradients[backend] = {key: tensor.grad.detach().cpu() for key, tensor in values.items()
                              if tensor.grad is not None}
    torch.testing.assert_close(outputs["kernel"], outputs["reference"], rtol=1e-4, atol=1e-4)
    assert gradients["kernel"].keys() == gradients["reference"].keys()
    for key, expected in gradients["reference"].items():
        torch.testing.assert_close(gradients["kernel"][key], expected, rtol=1e-3, atol=1e-3,
                                   msg=f"gradient of {key}")
