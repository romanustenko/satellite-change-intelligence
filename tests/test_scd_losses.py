"""Per-model SCD losses (R2-V4-T07) against independent references, on synthetic data."""

import os

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from satchange.semantic import scd_losses as L


def _labels(batch=2, size=12, seed=0, change_fraction=0.4):
    generator = torch.Generator().manual_seed(seed)
    changed = torch.rand(batch, size, size, generator=generator) < change_fraction
    first = torch.randint(1, 7, (batch, size, size), generator=generator)
    second = torch.randint(1, 7, (batch, size, size), generator=generator)
    return first * changed, second * changed


def _outputs(change_channels, batch=2, size=12, seed=1):
    generator = torch.Generator().manual_seed(seed)

    def make(channels):
        return torch.randn(batch, channels, size, size, generator=generator).requires_grad_()

    return make(change_channels), make(7), make(7)


def _lovasz_reference(probabilities, labels, ignore=255):
    """Berman et al. Alg. 1 as the original per-class loop (batch level, classes present)."""
    classes = probabilities.shape[1]
    flat = probabilities.permute(0, 2, 3, 1).reshape(-1, classes)
    target = labels.reshape(-1)
    keep = target != ignore
    flat, target = flat[keep], target[keep]
    losses = []
    for cls in range(classes):
        foreground = (target == cls).float()
        if foreground.sum() == 0:
            continue
        errors = (foreground - flat[:, cls]).abs()
        errors_sorted, permutation = torch.sort(errors, 0, descending=True)
        foreground_sorted = foreground[permutation]
        total = foreground_sorted.sum()
        intersection = total - foreground_sorted.cumsum(0)
        union = total + (1 - foreground_sorted).cumsum(0)
        jaccard = 1.0 - intersection / union
        if len(jaccard) > 1:
            jaccard[1:] = jaccard[1:] - jaccard[:-1]
        losses.append(torch.dot(errors_sorted, jaccard))
    return torch.stack(losses).mean()


def test_masked_cross_entropy_matches_torch_and_handles_empty():
    logits = torch.randn(2, 7, 6, 6, requires_grad=True)
    labels = torch.randint(0, 7, (2, 6, 6))
    valid = labels > 0
    expected = F.cross_entropy(
        logits, torch.where(valid, labels, torch.full_like(labels, 255)), ignore_index=255
    )
    assert torch.allclose(L.masked_cross_entropy(logits, labels, valid), expected, atol=1e-6)
    assert torch.allclose(
        L.masked_cross_entropy(logits, labels), F.cross_entropy(logits, labels), atol=1e-6
    )
    empty = L.masked_cross_entropy(logits, labels, torch.zeros_like(valid))
    assert empty.item() == 0.0
    empty.backward()
    assert logits.grad is not None


def test_lovasz_matches_reference_and_extremes():
    logits = torch.randn(2, 7, 8, 8)
    labels = torch.randint(0, 7, (2, 8, 8))
    valid = labels > 0
    reference_labels = torch.where(valid, labels, torch.full_like(labels, 255))
    reference = _lovasz_reference(F.softmax(logits, dim=1), reference_labels)
    assert torch.allclose(L.lovasz_softmax(logits, labels, valid), reference, atol=1e-5)
    assert torch.allclose(
        L.lovasz_softmax(logits, labels),
        _lovasz_reference(F.softmax(logits, dim=1), labels),
        atol=1e-5,
    )
    perfect = F.one_hot(labels, 7).permute(0, 3, 1, 2).float() * 60.0
    assert L.lovasz_softmax(perfect, labels).item() == pytest.approx(0.0, abs=1e-4)
    wrong = F.one_hot((labels + 1) % 7, 7).permute(0, 3, 1, 2).float() * 60.0
    assert L.lovasz_softmax(wrong, labels).item() > 0.9
    nothing = L.lovasz_softmax(logits.clone().requires_grad_(), labels, torch.zeros_like(valid))
    assert nothing.item() == 0.0


def test_lovasz_gradient_flows():
    logits = torch.randn(1, 3, 5, 5, requires_grad=True)
    L.lovasz_softmax(logits, torch.randint(0, 3, (1, 5, 5))).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_mamba_scd_loss_formula_and_masks():
    first, second = _labels()
    outputs = _outputs(2)
    parts = L.MambaSCDLoss().terms(outputs, first, second)
    change, out_1, out_2 = outputs
    target = (first > 0).long()
    ce_change = F.cross_entropy(change, target)
    ignored_1 = torch.where(first > 0, first, torch.full_like(first, 255))
    ignored_2 = torch.where(second > 0, second, torch.full_like(second, 255))
    ce_1 = F.cross_entropy(out_1, ignored_1, ignore_index=255)
    ce_2 = F.cross_entropy(out_2, ignored_2, ignore_index=255)
    lovasz_change = _lovasz_reference(F.softmax(change, 1), target)
    lovasz_1 = _lovasz_reference(F.softmax(out_1, 1), ignored_1)
    lovasz_2 = _lovasz_reference(F.softmax(out_2, 1), ignored_2)
    mask = (ignored_1 == 255).float().unsqueeze(1).expand_as(out_1)
    similarity = F.mse_loss(F.softmax(out_1, 1) * mask, F.softmax(out_2, 1) * mask)
    expected = (
        ce_change
        + 0.5 * (ce_1 + ce_2 + 0.5 * similarity)
        + 0.75 * (lovasz_change + 0.5 * (lovasz_1 + lovasz_2))
    )
    assert torch.allclose(parts["loss"], expected, atol=1e-5)
    assert torch.allclose(L.MambaSCDLoss()(outputs, first, second), expected, atol=1e-5)
    parts["loss"].backward()
    assert all(o.grad is not None and torch.isfinite(o.grad).all() for o in outputs)


def test_mamba_scd_loss_survives_a_batch_without_change():
    outputs = _outputs(2)
    zeros = torch.zeros(2, 12, 12, dtype=torch.int64)
    loss = L.MambaSCDLoss()(outputs, zeros, zeros)
    assert torch.isfinite(loss)
    loss.backward()


def test_weighted_bce_matches_scannet_formula():
    logits = torch.randn(2, 9, 9)
    target = (torch.rand(2, 9, 9) < 0.3).float()
    flat, truth = logits.reshape(-1), target.reshape(-1)
    loss = F.binary_cross_entropy_with_logits(flat, truth, reduction="none")
    positive, negative = (truth > 0.5).float(), (truth < 0.5).float()
    expected = (
        0.25 * positive * loss / (positive.sum() + 1e-12)
        + 0.75 * negative * loss / (negative.sum() + 1e-12)
    ).sum()
    assert torch.allclose(L.weighted_bce_logits(logits, target), expected, atol=1e-6)
    uniform = L.weighted_bce_logits(torch.zeros(4, 4), torch.ones(4, 4))
    assert uniform.item() == pytest.approx(0.25 * np.log(2), rel=1e-5)


def test_change_similarity_matches_cosine_embedding_loss():
    first, second = torch.randn(2, 6, 5, 5), torch.randn(2, 6, 5, 5)
    changed = torch.rand(2, 5, 5) < 0.5
    batch, classes, height, width = first.shape
    x1 = F.softmax(first, 1).permute(0, 2, 3, 1).reshape(-1, classes)
    x2 = F.softmax(second, 1).permute(0, 2, 3, 1).reshape(-1, classes)
    target = (~changed).float() - changed.float()
    expected = nn.CosineEmbeddingLoss(margin=0.1)(x1, x2, target.reshape(-1))
    assert torch.allclose(L.change_similarity(first, second, changed), expected, atol=1e-6)


def test_scannet_loss_formula_and_pseudo_labels_fill_unchanged_only():
    first, second = _labels(seed=3)
    outputs = _outputs(1, seed=4)
    parts = L.SCanNetLoss().terms(outputs, first, second)
    change, out_1, out_2 = outputs
    expected = (
        0.5
        * (
            F.cross_entropy(out_1, first, ignore_index=0)
            + F.cross_entropy(out_2, second, ignore_index=0)
        )
        + L.weighted_bce_logits(change[:, 0], (first > 0).float())
        + L.change_similarity(out_1[:, 1:], out_2[:, 1:], first > 0)
    )
    assert torch.allclose(parts["loss"], expected, atol=1e-5)
    pseudo = torch.full_like(first, 2)
    with_pseudo = L.SCanNetLoss().terms(
        outputs, first, second, pseudo_before=pseudo, pseudo_after=pseudo
    )
    filled_1 = torch.where(first == 0, pseudo, first)
    filled_2 = torch.where(second == 0, pseudo, second)
    expected_ce = 0.5 * (
        F.cross_entropy(out_1, filled_1, ignore_index=0)
        + F.cross_entropy(out_2, filled_2, ignore_index=0)
    )
    assert torch.allclose(with_pseudo["ce_t1"] + with_pseudo["ce_t2"], 2 * expected_ce, atol=1e-5)
    assert torch.equal(with_pseudo["bce_change"], parts["bce_change"])  # change target unchanged
    with pytest.raises(ValueError, match="Pseudo labels"):
        L.SCanNetLoss().terms(outputs, first, second, pseudo_before=pseudo.float())


class _FixedTeacher(nn.Module):
    """Returns fixed per-pixel logits (flip-aware: it is applied to the flipped input as is)."""

    def __init__(self, logits_1, logits_2):
        super().__init__()
        self.register_buffer("logits_1", logits_1)
        self.register_buffer("logits_2", logits_2)
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, before, after):
        batch, _, height, width = before.shape
        return (torch.zeros(batch, 1, height, width), self.logits_1, self.logits_2)


def test_pseudo_labeler_rules_and_percentile():
    size = 8
    classes = torch.randint(1, 7, (1, size, size))
    logits = F.one_hot(classes, 7).permute(0, 3, 1, 2).float() * 8.0
    teacher = _FixedTeacher(logits, logits.clone())
    labeler = L.SCanNetPseudoLabeler(start_step=1, tta=False)
    images = torch.zeros(1, 3, size, size)
    with pytest.raises(RuntimeError, match="refresh_teacher"):
        labeler(images, images, torch.zeros(1, size, size, dtype=torch.int64))
    labeler.refresh_teacher(teacher)
    assert labeler.active
    unchanged = torch.zeros(1, size, size, dtype=torch.int64)
    unchanged[0, :4] = 3  # the upper half is changed in the ground truth
    pseudo = labeler(images, images, unchanged)
    assert pseudo.dtype == torch.int64
    assert (pseudo[0, :4] == 0).all()  # never inside the changed region
    lower = pseudo[0, 4:]
    assert torch.equal(lower[lower > 0], classes[0, 4:][lower > 0])
    assert (lower > 0).any()
    # dates disagree -> no pseudo label
    other = torch.roll(logits, 1, dims=1)
    labeler.refresh_teacher(_FixedTeacher(logits, other))
    assert (labeler(images, images, torch.zeros_like(unchanged)) == 0).all()


def test_pseudo_labeler_tta_uses_flips_and_percentile_matches_numpy():
    values = torch.rand(101)
    for percent in (0.0, 37.5, 60.0, 100.0):
        assert L._percentile(values, percent) == pytest.approx(
            float(np.percentile(values.numpy().astype(np.float64), percent)), abs=1e-7
        )
    thresholds = L._RunningThresholds(2, 0.6)
    thresholds.update([0.95, 0.1], [3.0, 1.0])
    assert thresholds.value() == [pytest.approx(min((0.6 + 0.95 * 3) / 4, 0.9)), 0.5]

    calls = []

    class Recorder(_FixedTeacher):
        def forward(self, before, after):
            calls.append(before.clone())
            return super().forward(before, after)

    size = 4
    logits = torch.randn(1, 7, size, size)
    labeler = L.SCanNetPseudoLabeler(start_step=1, tta=True)
    labeler.refresh_teacher(Recorder(logits, logits.clone()))
    image = (
        torch.arange(size * size, dtype=torch.float32)
        .reshape(1, 1, size, size)
        .expand(1, 3, -1, -1)
    )
    labeler(image, image, torch.zeros(1, size, size, dtype=torch.int64))
    assert len(calls) == 4
    assert torch.equal(calls[1], torch.flip(image, [2]))
    assert torch.equal(calls[3], torch.flip(image, [2, 3]))


def test_siam_unet_loss_and_registry():
    first, second = _labels(seed=5)
    outputs = _outputs(1, seed=6)
    parts = L.SiamUNetLoss().terms(outputs, first, second)
    change, out_1, out_2 = outputs
    expected = (
        F.binary_cross_entropy_with_logits(change[:, 0], (first > 0).float())
        + F.cross_entropy(out_1, first, ignore_index=0)
        + F.cross_entropy(out_2, second, ignore_index=0)
    )
    assert torch.allclose(parts["loss"], expected, atol=1e-5)
    assert set(L.SCD_LOSS_REGISTRY) == {"mamba_scd", "scannet", "siam_unet_resnet18"}
    assert isinstance(L.create_scd_loss("scannet"), L.SCanNetLoss)
    with pytest.raises(ValueError, match="Unknown SCD model"):
        L.create_scd_loss("fc_siam_semantic")


def test_input_validation():
    first, second = _labels()
    outputs = _outputs(1)
    with pytest.raises(ValueError, match="2-channel"):
        L.MambaSCDLoss()(outputs, first, second)
    with pytest.raises(ValueError, match="1-channel"):
        L.SiamUNetLoss()(_outputs(2), first, second)
    with pytest.raises(ValueError, match="int64"):
        L.SiamUNetLoss()(outputs, first.int(), second)
    with pytest.raises(ValueError, match="tuple"):
        L.SiamUNetLoss()(outputs[0], first, second)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="semantic_t1"):
        L.SiamUNetLoss()((outputs[0], outputs[1][:, :6], outputs[2]), first, second)


def test_real_model_outputs_work_with_their_losses():
    from satchange.models.siam_unet_resnet import SiamUNetResNet18

    model = SiamUNetResNet18(head_layout="multi_head")
    before = torch.rand(2, 3, 64, 64)
    first, second = _labels(size=64, seed=7)
    loss = L.create_scd_loss("siam_unet_resnet18")(model(before, before.flip(3)), first, second)
    loss.backward()
    assert torch.isfinite(loss)


@pytest.mark.skipif(not os.environ.get("SATCHANGE_TEST_CUDA_DEVICE"), reason="opt-in GPU test")
def test_gpu_losses_match_cpu_and_are_deterministic():
    device = torch.device(os.environ["SATCHANGE_TEST_CUDA_DEVICE"])
    torch.use_deterministic_algorithms(True)
    try:
        first, second = _labels(batch=2, size=64, seed=8)
        results = []
        for _ in range(2):
            outputs = _outputs(2, size=64, seed=9)
            gpu_outputs = tuple(o.detach().to(device).requires_grad_() for o in outputs)
            loss = L.MambaSCDLoss()(gpu_outputs, first.to(device), second.to(device))
            loss.backward()
            results.append((loss.detach().cpu(), [o.grad.cpu() for o in gpu_outputs]))
        cpu_loss = L.MambaSCDLoss()(_outputs(2, size=64, seed=9), first, second)
        assert torch.allclose(results[0][0], cpu_loss.detach(), atol=1e-4)
        assert torch.equal(results[0][0], results[1][0])
        for a, b in zip(results[0][1], results[1][1], strict=True):
            assert torch.equal(a, b)
    finally:
        torch.use_deterministic_algorithms(False)
