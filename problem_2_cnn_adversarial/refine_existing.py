"""Reduce distortion of already-successful PNG attacks without changing labels.

This conservative refinement starts from the submitted disk image, verifies its
hard-label attack, and binary-searches toward the clean image. It is safer for
ranking than replacing a proven attack with a new generator: only candidates
that remain adversarial after uint8 quantization are accepted.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from attack import CLASS_NAMES, IMAGES_PER_CLASS, HardLabelAdversarialAttack


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "adv_images")
    parser.add_argument("--output", type=Path, default=root / "adv_images_refined")
    parser.add_argument("--limit", type=int, default=IMAGES_PER_CLASS)
    args = parser.parse_args()

    clean_dir = root / "p2_data" / "clean_images"
    model = torch.jit.load(str(root / "p2_data" / "cnn_model.pt"), map_location="cpu").eval()
    helper = HardLabelAdversarialAttack(model, device="cpu")
    args.output.mkdir(parents=True, exist_ok=True)

    success = 0
    total = 0
    for label, name in enumerate(CLASS_NAMES):
        indices = list(range(min(args.limit, IMAGES_PER_CLASS)))
        cleans, originals = [], []
        for index in indices:
            clean_path = clean_dir / f"{name}_{index}.png"
            adv_path = args.input / f"{name}_{index}_adv.png"
            with Image.open(clean_path) as clean_file, Image.open(adv_path) as adv_file:
                cleans.append(transforms.functional.to_tensor(clean_file.convert("RGB")))
                originals.append(transforms.functional.to_tensor(adv_file.convert("RGB")))

        clean_batch = torch.stack(cleans)
        original_batch = torch.stack(originals)
        clean_labels = helper._predict(clean_batch)
        if not torch.all(clean_labels == label):
            raise RuntimeError(f"Clean label mismatch in class {name}")
        original_predictions = helper._predict(original_batch)
        if not torch.all(original_predictions != clean_labels):
            raise RuntimeError(f"Input attack is not adversarial in class {name}")

        low = torch.zeros(len(indices))
        high = torch.ones(len(indices))
        best = (original_batch * 255.0).round().clamp(0, 255).to(torch.uint8)
        direction = original_batch - clean_batch
        # Quantized bisection is monotonic along this fixed direction in the
        # usual decision-based setting; verify every accepted point in batches.
        for _ in range(12):
            mid = (low + high) / 2.0
            candidate = (clean_batch + mid[:, None, None, None] * direction).clamp(0.0, 1.0)
            uint8 = (candidate * 255.0).round().clamp(0, 255).to(torch.uint8)
            reloaded = uint8.float() / 255.0
            predictions = helper._predict(reloaded)
            successful = predictions != clean_labels
            high[successful] = mid[successful]
            low[~successful] = mid[~successful]
            best[successful] = uint8[successful]

        for offset, index in enumerate(indices):
            output_path = args.output / f"{name}_{index}_adv.png"
            Image.fromarray(best[offset].permute(1, 2, 0).numpy()).save(output_path, format="PNG")
            success += int(helper._is_adversarial(best[offset].float() / 255.0, label))
            total += 1

    print(f"Refined and verified: {success}/{total}")
    print(f"Output directory: {args.output}")
    if success != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
