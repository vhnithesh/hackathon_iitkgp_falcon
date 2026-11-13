"""Apply hard-label HSJA refinement to the lowest-SSIM saved attacks.

Copies the current submission into ``--output`` and then refines the
``--count`` worst images (by SSIM) in resumable chunks. Each refined image is
kept only if it is still adversarial after uint8 quantization and has higher
SSIM than the image it replaces. Per-image oracle query counts are appended
to ``--log`` so the query tiebreaker is measured rather than estimated.
"""

import argparse
import csv
import shutil
from pathlib import Path

import torch
from PIL import Image
from pytorch_msssim import ssim
from torchvision import transforms

from attack import CLASS_NAMES, IMAGES_PER_CLASS
from hsja_refine import refine


def load(path):
    with Image.open(path) as image:
        return transforms.functional.to_tensor(image.convert("RGB"))


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "adv_images")
    parser.add_argument("--output", type=Path, default=root / "adv_images_hsja")
    parser.add_argument("--log", type=Path, default=root / "hsja_queries.csv")
    parser.add_argument("--count", type=int, default=400)
    parser.add_argument("--chunk", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--max-probes", type=int, default=60)
    args = parser.parse_args()

    clean_dir = root / "p2_data" / "clean_images"
    model = torch.jit.load(str(root / "p2_data" / "cnn_model.pt"), map_location="cpu").eval()
    args.output.mkdir(parents=True, exist_ok=True)
    for source in args.input.glob("*.png"):
        target = args.output / source.name
        if not target.exists():
            shutil.copyfile(source, target)

    names = [(name, i) for name in CLASS_NAMES for i in range(IMAGES_PER_CLASS)]
    scores = []
    for name, i in names:
        clean = load(clean_dir / f"{name}_{i}.png")
        adv = load(args.input / f"{name}_{i}_adv.png")
        scores.append(ssim(clean[None], adv[None], data_range=1.0).item())
    order = sorted(range(len(names)), key=lambda k: scores[k])[:args.count]

    done = set()
    if args.log.exists():
        with open(args.log, newline="") as handle:
            done = {row["image"] for row in csv.DictReader(handle)}
    todo = [k for k in order if f"{names[k][0]}_{names[k][1]}" not in done]
    print(f"{len(order) - len(todo)} already refined, {len(todo)} to go", flush=True)

    new_log = not args.log.exists()
    with open(args.log, "a", newline="") as handle:
        writer = csv.writer(handle)
        if new_log:
            writer.writerow(["image", "queries", "ssim_before", "ssim_after"])
        for start in range(0, len(todo), args.chunk):
            chunk = todo[start:start + args.chunk]
            clean = torch.stack([load(clean_dir / f"{names[k][0]}_{names[k][1]}.png") for k in chunk])
            current = torch.stack([
                load(args.output / f"{names[k][0]}_{names[k][1]}_adv.png") for k in chunk
            ])
            current_uint8 = (current * 255).round().to(torch.uint8)
            labels = torch.tensor([CLASS_NAMES.index(names[k][0]) for k in chunk])
            refined, queries = refine(
                model, clean, current_uint8, labels, iterations=args.iterations,
                max_probes=args.max_probes, seed=start,
            )
            before = ssim(clean, current, data_range=1.0, size_average=False)
            after = ssim(clean, refined.float() / 255, data_range=1.0, size_average=False)
            for j, k in enumerate(chunk):
                name, i = names[k]
                Image.fromarray(refined[j].permute(1, 2, 0).numpy()).save(
                    args.output / f"{name}_{i}_adv.png", format="PNG"
                )
                writer.writerow([f"{name}_{i}", int(queries[j]), f"{before[j]:.5f}", f"{after[j]:.5f}"])
            handle.flush()
            print(
                f"chunk {start // args.chunk + 1}/{-(-len(todo) // args.chunk)}: "
                f"SSIM {before.mean():.4f} -> {after.mean():.4f}, "
                f"mean queries {queries.float().mean():.0f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
