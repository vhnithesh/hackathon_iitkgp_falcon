"""Search stronger alternatives for the worst-quality saved attacks."""

import argparse
import shutil
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from pytorch_msssim import ssim
from torchvision import transforms

from attack import CLASS_NAMES, HardLabelAdversarialAttack


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "adv_images_refined")
    parser.add_argument("--output", type=Path, default=root / "adv_images_optimized")
    parser.add_argument("--quality-report", type=Path, default=root / "quality_report.csv")
    parser.add_argument("--worst", type=int, default=25)
    args = parser.parse_args()

    clean_dir = root / "p2_data" / "clean_images"
    args.output.mkdir(parents=True, exist_ok=True)
    for source in args.input.glob("*.png"):
        shutil.copyfile(source, args.output / source.name)

    model = torch.jit.load(str(root / "p2_data" / "cnn_model.pt"), map_location="cpu").eval()
    attacker = HardLabelAdversarialAttack(
        model, device="cpu", directions_per_scale=32, boundary_candidates=12
    )
    anchors = [
        transforms.functional.to_tensor(
            Image.open(clean_dir / f"{name}_0.png").convert("RGB")
        )
        for name in CLASS_NAMES
    ]
    quality = pd.read_csv(args.quality_report).sort_values("ssim").head(args.worst)
    replacements = 0

    for row in quality.itertuples(index=False):
        name, index = row.class_name, int(row.index)
        clean_path = clean_dir / f"{name}_{index}.png"
        current_path = args.output / f"{name}_{index}_adv.png"
        with Image.open(clean_path) as clean_file, Image.open(current_path) as current_file:
            clean = transforms.functional.to_tensor(clean_file.convert("RGB"))
            current = transforms.functional.to_tensor(current_file.convert("RGB"))
        label = CLASS_NAMES.index(name)
        candidate, _ = attacker.attack_single(
            clean, label, anchors=anchors, seed=index * len(CLASS_NAMES) + label
        )
        candidate = candidate.float() / 255.0
        current_ssim = ssim(clean.unsqueeze(0), current.unsqueeze(0), data_range=1.0).item()
        candidate_ssim = ssim(clean.unsqueeze(0), candidate.unsqueeze(0), data_range=1.0).item()
        candidate_pred = int(attacker._predict(candidate.unsqueeze(0))[0].item())
        if candidate_pred != label and candidate_ssim > current_ssim:
            Image.fromarray((candidate * 255.0).round().clamp(0, 255).byte().permute(1, 2, 0).numpy()).save(
                current_path, format="PNG"
            )
            replacements += 1
            print(f"replaced {name}_{index}: {current_ssim:.4f} -> {candidate_ssim:.4f}")
        else:
            print(f"kept {name}_{index}: {current_ssim:.4f} (candidate {candidate_ssim:.4f})")

    print(f"Replacements: {replacements}/{len(quality)}")
    print(f"Output directory: {args.output}")


if __name__ == "__main__":
    main()
