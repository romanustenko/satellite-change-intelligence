"""Frozen binary-detector probability maps aligned to the native scene grid."""

import torch
from torch import nn
from torch.nn import functional as F

from satchange.execution import precision_context
from satchange.semantic.roi import align_whole_image_probabilities


@torch.inference_mode()
def detector_probability_map(
    detector: nn.Module,
    before: torch.Tensor,
    after: torch.Tensor,
    *,
    input_size: tuple[int, int],
    output_kind: str,
    geometry: str = "whole_image_resize",
    precision: str = "fp32",
    normalization: str = "rgb_0_1",
) -> torch.Tensor:
    """Frozen detector with declared preprocessing; only RGB/255 whole-image resize is supported.

    Load the detector with run.reconstruct_checkpoint; native RGB stays untouched.
    Cropped/tiled/padded/custom-normalized inference must provide a known aligned map instead.
    """
    if geometry != "whole_image_resize" or normalization != "rgb_0_1":
        raise ValueError(
            "Unsupported detector preprocessing; supply an explicitly aligned saved map."
        )
    if output_kind not in ("logits", "probabilities"):
        raise ValueError("Declare detector output_kind as logits or probabilities.")
    if before.ndim != 3 or before.shape[0] != 3 or before.shape != after.shape:
        raise ValueError("Detector requires aligned native RGB [3,H,W].")
    for value in (before, after):
        if (
            value.dtype != torch.float32
            or not torch.isfinite(value).all()
            or value.min() < 0
            or value.max() > 1
        ):
            raise ValueError("Detector inputs must be float32 RGB in [0,1].")
    if len(input_size) != 2 or any(type(v) is not int or v < 1 for v in input_size):
        raise ValueError("Declare positive whole-image detector input_size (H,W).")
    detector.eval()
    device = next(detector.parameters()).device
    inputs = [
        F.interpolate(v.unsqueeze(0), size=input_size, mode="bilinear", align_corners=False).to(
            device
        )
        for v in (before, after)
    ]
    with precision_context(device, precision):
        output = detector(*inputs)
    if output.shape != (1, 1, *input_size) or not torch.isfinite(output).all():
        raise ValueError("Detector output must be finite [1,1,input H,input W].")
    probabilities = output.float().sigmoid() if output_kind == "logits" else output.float()
    return align_whole_image_probabilities(
        probabilities[0].cpu(), (before.shape[-2], before.shape[-1]), input_size
    )
