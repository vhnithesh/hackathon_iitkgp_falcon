"""Probe the ranking-optimized attack on a representative image subset."""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_msssim import ssim
from torchvision import transforms

from attack import CLASS_NAMES, HardLabelAdversarialAttack, IMAGES_PER_CLASS


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=10, help="Images per class to probe")
    args = parser.parse_args()

    device = "cpu"
    clean_dir = root / "p2_data" / "clean_images"
    model = torch.jit.load(str(root / "p2_data" / "cnn_model.pt"), map_location=device).eval()
    attacker = HardLabelAdversarialAttack(model, device=device)
    anchors = [
        transforms.functional.to_tensor(Image.open(clean_dir / f"{name}_0.png").convert("RGB"))
        for name in CLASS_NAMES
    ]

    asr = 0
    ssims, psnrs, l2s, queries = [], [], [], []
    with torch.no_grad():
        for label, name in enumerate(CLASS_NAMES):
            for index in range(min(args.limit, IMAGES_PER_CLASS)):
                with Image.open(clean_dir / f"{name}_{index}.png") as image_file:
                    clean = transforms.functional.to_tensor(image_file.convert("RGB"))
                adv_uint8, query_count = attacker.attack_single(
                    clean, label, anchors=anchors, seed=index * len(CLASS_NAMES) + label
                )
                adv = adv_uint8.float() / 255.0
                normalized = (adv.unsqueeze(0) - attacker.mean_t.unsqueeze(0)) / attacker.std_t.unsqueeze(0)
                prediction = int(model(normalized).reshape(-1)[0].item())
                asr += int(prediction != int(model(((clean.unsqueeze(0) - attacker.mean_t.unsqueeze(0)) / attacker.std_t.unsqueeze(0))).reshape(-1)[0].item()))
                mse = ((adv - clean) ** 2).mean().item()
                ssims.append(ssim(clean.unsqueeze(0), adv.unsqueeze(0), data_range=1.0).item())
                psnrs.append(10.0 * math.log10(1.0 / max(1e-8, mse)))
                l2s.append(torch.linalg.vector_norm(adv - clean).item())
                queries.append(query_count)

    total = len(ssims)
    print(f"Images: {total}")
    print(f"ASR: {100.0 * asr / total:.2f}%")
    print(f"Mean SSIM: {np.mean(ssims):.4f}")
    print(f"Mean PSNR: {np.mean(psnrs):.2f} dB")
    print(f"Mean L2: {np.mean(l2s):.4f}")
    print(f"Mean hard-label queries: {np.mean(queries):.1f}")


if __name__ == "__main__":
    main()
