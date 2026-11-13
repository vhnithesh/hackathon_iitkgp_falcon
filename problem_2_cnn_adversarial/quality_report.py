"""Rank saved adversarial images by visual quality for targeted refinement."""

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from pytorch_msssim import ssim
from torchvision import transforms

from attack import CLASS_NAMES, IMAGES_PER_CLASS


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adv-dir", type=Path, default=root / "adv_images_refined")
    parser.add_argument("--output", type=Path, default=root / "quality_report.csv")
    args = parser.parse_args()
    clean_dir = root / "p2_data" / "clean_images"
    rows = []
    for name in CLASS_NAMES:
        for index in range(IMAGES_PER_CLASS):
            with Image.open(clean_dir / f"{name}_{index}.png") as clean_file:
                clean = transforms.functional.to_tensor(clean_file.convert("RGB"))
            with Image.open(args.adv_dir / f"{name}_{index}_adv.png") as adv_file:
                adv = transforms.functional.to_tensor(adv_file.convert("RGB"))
            mse = ((adv - clean) ** 2).mean().item()
            rows.append({
                "class_name": name,
                "index": index,
                "ssim": ssim(clean.unsqueeze(0), adv.unsqueeze(0), data_range=1.0).item(),
                "psnr": 10.0 * math.log10(1.0 / max(1e-8, mse)),
                "l2": torch.linalg.vector_norm(adv - clean).item(),
                "linf": (adv - clean).abs().max().item(),
            })
    result = pd.DataFrame(rows).sort_values("ssim")
    result.to_csv(args.output, index=False)
    print(result.head(25).to_string(index=False))
    print(f"Mean SSIM: {result['ssim'].mean():.4f}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
