"""Batched HopSkipJump-style refinement using only hard-label oracle queries.

Starting from an already-adversarial image, the refinement repeatedly
(1) bisects to the decision boundary on the line towards the clean image,
(2) estimates the boundary normal from the hard labels of random probes, and
(3) steps along that normal and bisects back, shrinking the L2 distance.
The best uint8 image that is still adversarial after quantization is kept,
and the result is never worse (by SSIM) than the starting image.

The victim is touched only through ``HardLabelOracle.__call__``, which returns
predicted class indices and counts every queried image per sample.
"""

import math

import torch
from pytorch_msssim import ssim

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


class HardLabelOracle:
    """Hard-label access to the victim with per-sample query accounting."""

    def __init__(self, model, n, mean=MEAN, std=STD, chunk=4096, device="cpu"):
        self.model = model.eval()
        self.device = torch.device(device)
        self.mean = torch.tensor(mean, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=self.device).view(1, 3, 1, 1)
        self.queries = torch.zeros(n, dtype=torch.long)
        self.chunk = chunk

    def __call__(self, images, owners):
        """Return labels for ``images``; ``owners`` maps each row to a sample."""
        self.queries.index_add_(0, owners, torch.ones_like(owners))
        out = []
        with torch.no_grad():
            for start in range(0, len(images), self.chunk):
                batch = images[start:start + self.chunk].to(self.device)
                out.append(self.model((batch - self.mean) / self.std).reshape(-1).long().cpu())
        return torch.cat(out)


def _bview(v):
    return v.view(-1, 1, 1, 1)


def _boundary(oracle, idx, clean, start, labels, steps):
    """Bisect between clean (not adversarial) and start (adversarial)."""
    lo = torch.zeros(len(idx))
    hi = torch.ones(len(idx))
    for _ in range(steps):
        mid = (lo + hi) / 2
        cand = clean + _bview(mid) * (start - clean)
        adv = oracle(cand, idx) != labels
        hi = torch.where(adv, mid, hi)
        lo = torch.where(adv, lo, mid)
    return clean + _bview(hi) * (start - clean)


def _quantize(oracle, idx, clean, x, labels, steps=8):
    """Smallest uint8 point on clean->x (scaled up to 2x) that stays adversarial."""
    n = len(idx)
    best = torch.full((n,), float("nan"))
    lo = torch.zeros(n)
    hi = torch.full((n,), float("nan"))
    for alpha in (1.0, 1.02, 1.05, 1.1, 1.2, 1.4, 1.7, 2.0):
        todo = torch.isnan(hi)
        if not todo.any():
            break
        cand = ((clean + alpha * (x - clean)).clamp(0, 1) * 255).round() / 255
        adv = oracle(cand[todo], idx[todo]) != labels[todo]
        sel = todo.nonzero().flatten()
        hi[sel[adv]] = alpha
        lo[sel[~adv]] = alpha
    found = ~torch.isnan(hi)
    best[found] = hi[found]
    for _ in range(steps):
        if not found.any():
            break
        mid = (lo + torch.nan_to_num(hi)) / 2
        cand = ((clean + _bview(mid) * (x - clean)).clamp(0, 1) * 255).round() / 255
        sel = found.nonzero().flatten()
        adv = oracle(cand[sel], idx[sel]) != labels[sel]
        hi[sel[adv]] = mid[sel[adv]]
        lo[sel[~adv]] = mid[sel[~adv]]
        best[sel[adv]] = mid[sel[adv]]
    out = ((clean + _bview(torch.nan_to_num(best, 0.0)) * (x - clean)).clamp(0, 1) * 255).round()
    return out.to(torch.uint8), found


def refine(model, clean, start_uint8, labels, iterations=30, max_probes=200,
           seed=0, oracle=None, device="cpu"):
    """Refine a batch of adversarial uint8 images; return (uint8, queries).

    ``clean`` is float in [0, 1], shape (N, 3, H, W) on the CPU; ``labels``
    are the victim's hard labels for the clean images. Only the oracle's
    queries run on ``device``.
    """
    gen = torch.Generator().manual_seed(seed)
    n = len(clean)
    idx = torch.arange(n)
    oracle = oracle or HardLabelOracle(model, n, device=device)
    start = start_uint8.float() / 255
    if not torch.all(oracle(start, idx) != labels):
        raise ValueError("Every starting image must already be adversarial.")

    dim = clean[0].numel()
    xb = _boundary(oracle, idx, clean, start, labels, 12)
    for t in range(1, iterations + 1):
        dist = (xb - clean).flatten(1).norm(dim=1)
        delta = (dist / math.sqrt(dim) * 0.1 / math.sqrt(t)).clamp(min=1e-3)
        b = min(int(50 * math.sqrt(t)), max_probes)
        u = torch.randn(n, b, *clean.shape[1:], generator=gen)
        u = u / u.flatten(2).norm(dim=2)[..., None, None, None]
        probes = (xb[:, None] + delta.view(-1, 1, 1, 1, 1) * u).clamp(0, 1)
        owners = idx.repeat_interleave(b)
        phi = (oracle(probes.flatten(0, 1), owners) != labels.repeat_interleave(b))
        phi = phi.view(n, b).float() * 2 - 1
        centred = phi - phi.mean(dim=1, keepdim=True)
        uniform = phi.abs().mean(dim=1, keepdim=True) == 1  # all probes agree
        weights = torch.where(uniform, phi, centred)
        grad = (weights[..., None, None, None] * u).mean(dim=1)
        grad = grad / _bview(grad.flatten(1).norm(dim=1) + 1e-12)

        # Geometric step-size search: halve until the step stays adversarial.
        eps = dist / math.sqrt(t)
        stepped = xb.clone()
        pending = torch.ones(n, dtype=torch.bool)
        for _ in range(12):
            sel = pending.nonzero().flatten()
            if len(sel) == 0:
                break
            cand = (xb[sel] + _bview(eps[sel]) * grad[sel]).clamp(0, 1)
            ok = oracle(cand, sel) != labels[sel]
            stepped[sel[ok]] = cand[ok]
            pending[sel[ok]] = False
            eps[sel[~ok]] /= 2
        moved = ~pending
        if moved.any():
            sel = moved.nonzero().flatten()
            xb[sel] = _boundary(oracle, sel, clean[sel], stepped[sel], labels[sel], 10)

    refined, found = _quantize(oracle, idx, clean, xb, labels)
    # Keep the better of start and refined image per sample (SSIM, the
    # imperceptibility metric); a refined image is only eligible if verified.
    final = start_uint8.clone()
    for i in range(n):
        if not found[i]:
            continue
        s_new = ssim(clean[i:i + 1], refined[i:i + 1].float() / 255, data_range=1.0)
        s_old = ssim(clean[i:i + 1], start[i:i + 1], data_range=1.0)
        if s_new > s_old:
            final[i] = refined[i]
    final_adv = oracle(final.float() / 255, idx) != labels
    if not torch.all(final_adv):
        raise RuntimeError("Refinement produced a non-adversarial image.")
    return final, oracle.queries.clone()
