"""Decision-based hard-label attack and reproducible image generation.

The victim is used only through ``model(normalized_images)`` and is expected to
return a class index for each image. The attack does not inspect logits,
weights, gradients, or probabilities. Every image is re-quantized to uint8 and
queried again before it is written to disk.
"""

from pathlib import Path

import torch
from PIL import Image
from pytorch_msssim import ssim
from torchvision import transforms


CLASS_NAMES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
IMAGES_PER_CLASS = 100
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


class HardLabelAdversarialAttack:
    """A query-only directional search with boundary bisection."""

    def __init__(
        self,
        model,
        mean=MEAN,
        std=STD,
        device="cpu",
        directions_per_scale=8,
        boundary_candidates=4,
    ):
        self.model = model.eval()
        self.device = torch.device(device)
        self.mean_t = torch.tensor(mean, device=self.device).view(3, 1, 1)
        self.std_t = torch.tensor(std, device=self.device).view(3, 1, 1)
        self.directions_per_scale = directions_per_scale
        self.boundary_candidates = boundary_candidates

    def _predict(self, images):
        """Return hard labels; this is the only victim-model access point."""
        with torch.no_grad():
            normalized = (images - self.mean_t.unsqueeze(0)) / self.std_t.unsqueeze(0)
            output = self.model(normalized)
        return output.reshape(-1).to(dtype=torch.long)

    def _is_adversarial(self, image, label):
        return int(self._predict(image.unsqueeze(0))[0].item()) != int(label)

    def _boundary_search(self, clean, direction, label, high):
        """Find a boundary point, assuming clean is original and high is adversarial."""
        low = 0.0
        for _ in range(9):
            mid = (low + high) / 2.0
            candidate = (clean + mid * direction).clamp(0.0, 1.0)
            if self._is_adversarial(candidate, label):
                high = mid
            else:
                low = mid
        return (clean + high * direction).clamp(0.0, 1.0)

    def _quantize_and_verify(self, clean, x_adv, label):
        """Return the best verified uint8 candidate and query count."""
        direction = x_adv - clean
        previous_failed = 0.0
        queries = 0
        # The first successful rounded point is then tightened by bisection.
        for alpha in (1.0, 1.01, 1.02, 1.04, 1.08, 1.15, 1.25, 1.40, 1.60, 2.0):
            candidate = (clean + alpha * direction).clamp(0.0, 1.0)
            uint8 = (candidate * 255.0).round().clamp(0, 255).to(torch.uint8)
            reloaded = uint8.to(dtype=torch.float32) / 255.0
            queries += 1
            if self._is_adversarial(reloaded, label):
                best_uint8 = uint8
                low, high = previous_failed, alpha
                for _ in range(4):
                    mid = (low + high) / 2.0
                    mid_candidate = (clean + mid * direction).clamp(0.0, 1.0)
                    mid_uint8 = (mid_candidate * 255.0).round().clamp(0, 255).to(torch.uint8)
                    mid_reloaded = mid_uint8.to(dtype=torch.float32) / 255.0
                    queries += 1
                    if self._is_adversarial(mid_reloaded, label):
                        high = mid
                        best_uint8 = mid_uint8
                    else:
                        low = mid
                return best_uint8, queries
            previous_failed = alpha
        return None, queries

    @staticmethod
    def _ssim(clean, uint8):
        reloaded = uint8.to(dtype=torch.float32).unsqueeze(0) / 255.0
        return float(ssim(clean.unsqueeze(0), reloaded, data_range=1.0).item())

    def attack_single(self, clean, label, anchors=(), seed=42):
        """Return a uint8 adversarial tensor and the number of oracle queries."""
        generator = torch.Generator(device=self.device).manual_seed(seed)
        clean = clean.to(self.device).float().clamp(0.0, 1.0)
        label = int(label)
        queries = 0

        if self._predict(clean.unsqueeze(0))[0].item() != label:
            raise ValueError(f"Clean input is not classified as its supplied label {label}.")
        queries += 1

        # Search smooth directions at increasing radii. Keep the smallest
        # successful candidate at each radius instead of accepting the first.
        directions = []
        for size in (8, 12, 16):
            coarse = torch.randn(
                self.directions_per_scale, 3, size, size,
                generator=generator, device=self.device
            )
            smooth = torch.nn.functional.interpolate(
                coarse, size=clean.shape[-2:], mode="bilinear", align_corners=False
            )
            smooth = smooth / (torch.linalg.vector_norm(smooth.flatten(1), dim=1)
                               .view(-1, 1, 1, 1) + 1e-8)
            directions.append(smooth)
        directions = torch.cat(directions, dim=0)

        x_adv = None
        for radius in (0.10, 0.20, 0.35, 0.55, 0.80, 1.20):
            candidates = (clean.unsqueeze(0) + radius * directions).clamp(0.0, 1.0)
            predictions = self._predict(candidates)
            queries += len(directions)
            successful = (predictions != label).nonzero(as_tuple=False).flatten()
            if len(successful):
                # Refine every successful direction and retain the shortest
                # boundary points. Accepting the first crossing wastes SSIM.
                boundary_candidates = []
                ordered = sorted(
                    successful.tolist(),
                    key=lambda index: torch.linalg.vector_norm(candidates[index] - clean).item(),
                )
                for index in ordered[:self.boundary_candidates]:
                    boundary = self._boundary_search(
                        clean, directions[index], label, radius
                    )
                    boundary_candidates.append(boundary)
                    queries += 9

                # Ranking is driven by the post-PNG image, so choose the best
                # verified SSIM rather than the best floating-point image.
                verified = []
                for boundary in boundary_candidates:
                    quantized, extra_queries = self._quantize_and_verify(
                        clean, boundary, label
                    )
                    queries += extra_queries
                    if quantized is not None:
                        verified.append((self._ssim(clean, quantized), quantized))
                if verified:
                    return max(verified, key=lambda item: item[0])[1], queries

        # Cross-class anchors are a deterministic fallback. An anchor is used
        # only after checking its hard label, so the bisection invariant holds.
        if x_adv is None:
            best = None
            for other_label, anchor in enumerate(anchors):
                if other_label == label:
                    continue
                anchor = anchor.to(self.device).float().clamp(0.0, 1.0)
                if not self._is_adversarial(anchor, label):
                    queries += 1
                    continue
                queries += 1
                candidate = self._boundary_search(clean, anchor - clean, label, 1.0)
                queries += 9
                distance = torch.linalg.vector_norm(candidate - clean).item()
                if best is None or distance < best[0]:
                    best = (distance, candidate)
            if best is None:
                raise RuntimeError(f"No adversarial anchor found for label {label}.")
            x_adv = best[1]

        # The anchor fallback also has to satisfy the same disk-reloaded gate.
        quantized, extra_queries = self._quantize_and_verify(clean, x_adv, label)
        queries += extra_queries
        if quantized is None:
            raise RuntimeError(f"8-bit quantization could not preserve the attack for label {label}.")
        return quantized, queries


def generate(model_path, output_dir, clean_dir, mirror_dir=None, device="cpu"):
    """Generate all adversarial PNGs and verify each quantized tensor first."""
    model = torch.jit.load(str(model_path), map_location=device).eval()
    attacker = HardLabelAdversarialAttack(model, device=device)
    output_dir = Path(output_dir)
    clean_dir = Path(clean_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if mirror_dir is not None:
        mirror_dir = Path(mirror_dir)
        mirror_dir.mkdir(parents=True, exist_ok=True)

    anchors = [
        transforms.functional.to_tensor(
            Image.open(clean_dir / f"{name}_0.png").convert("RGB")
        ).to(device)
        for name in CLASS_NAMES
    ]

    for label, name in enumerate(CLASS_NAMES):
        for index in range(IMAGES_PER_CLASS):
            clean_path = clean_dir / f"{name}_{index}.png"
            output_path = output_dir / f"{name}_{index}_adv.png"
            with Image.open(clean_path) as image_file:
                clean = transforms.functional.to_tensor(image_file.convert("RGB"))
            adversarial, queries = attacker.attack_single(
                clean, label, anchors=anchors, seed=index * len(CLASS_NAMES) + label
            )
            output_image = Image.fromarray(adversarial.permute(1, 2, 0).cpu().numpy())
            output_image.save(output_path, format="PNG")
            if mirror_dir is not None:
                output_image.save(mirror_dir / output_path.name, format="PNG")
            print(f"{output_path.name}: {queries} hard-label queries")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    generate(
        root / "p2_data" / "cnn_model.pt",
        root / "adv_images",
        root / "p2_data" / "clean_images",
        root / "p2_data" / "adv_images",
    )
