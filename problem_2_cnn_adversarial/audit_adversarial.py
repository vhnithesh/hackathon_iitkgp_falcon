import math
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import pytorch_msssim

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
MEAN_T = torch.tensor(MEAN).view(3, 1, 1)
STD_T = torch.tensor(STD).view(3, 1, 1)
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
IMAGES_PER_CLASS = 100
TARGET_ASR = 0.80
INTERNAL_SSIM_GOAL = 0.93


def wilson_interval(successes, trials, z=1.96):
    """Return a 95% Wilson interval for a binomial rate."""
    if trials <= 0:
        return (float("nan"), float("nan"))
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denominator
    radius = z * ((p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) ** 0.5) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)

def _expected_names(suffix, limit):
    return {
        f"{class_name}_{idx}{suffix}.png"
        for class_name in CLASS_NAMES
        for idx in range(limit)
    }


def audit(adv_dir, clean_dir, model_path, limit_per_class=IMAGES_PER_CLASS):
    adv_dir = Path(adv_dir)
    clean_dir = Path(clean_dir)
    model_path = Path(model_path)

    expected_clean = _expected_names("", limit_per_class)
    expected_adv = _expected_names("_adv", limit_per_class)
    actual_clean = {p.name for p in clean_dir.glob("*.png")}
    actual_adv = {p.name for p in adv_dir.glob("*.png")}
    missing_clean = sorted(expected_clean - actual_clean)
    missing_adv = sorted(expected_adv - actual_adv)
    extra_clean = sorted(actual_clean - expected_clean) if limit_per_class == IMAGES_PER_CLASS else []
    extra_adv = sorted(actual_adv - expected_adv) if limit_per_class == IMAGES_PER_CLASS else []
    if missing_clean or missing_adv or extra_clean or extra_adv:
        raise FileNotFoundError(
            f"Missing clean images: {missing_clean[:3]}{', ...' if len(missing_clean) > 3 else ''}; "
            f"missing adversarial images: {missing_adv[:3]}{', ...' if len(missing_adv) > 3 else ''}; "
            f"unexpected clean images: {extra_clean[:3]}{', ...' if len(extra_clean) > 3 else ''}; "
            f"unexpected adversarial images: {extra_adv[:3]}{', ...' if len(extra_adv) > 3 else ''}"
        )

    print("=" * 65)
    print("DISK-RELOADED ADVERSARIAL ATTACK AUDIT")
    print("=" * 65)
    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()

    total = 0
    clean_correct = 0
    misclassified = 0
    l2_list, linf_list, psnr_list, ssim_list = [], [], [], []

    with torch.no_grad():
        for c_idx, c_name in enumerate(CLASS_NAMES):
            for idx in range(limit_per_class):
                total += 1
                f_clean = clean_dir / f"{c_name}_{idx}.png"
                f_adv = adv_dir / f"{c_name}_{idx}_adv.png"

                with Image.open(f_clean) as clean_file, Image.open(f_adv) as adv_file:
                    img_c = transforms.functional.to_tensor(clean_file.convert("RGB"))
                    img_a = transforms.functional.to_tensor(adv_file.convert("RGB"))

                    # Evaluate the clean and disk-reloaded adversarial image under the model.
                    norm_c = (img_c.unsqueeze(0) - MEAN_T.unsqueeze(0)) / STD_T.unsqueeze(0)
                    norm_a = (img_a.unsqueeze(0) - MEAN_T.unsqueeze(0)) / STD_T.unsqueeze(0)
                    clean_pred = int(model(norm_c).reshape(-1)[0].item())
                    adv_pred = int(model(norm_a).reshape(-1)[0].item())
                    clean_correct += int(clean_pred == c_idx)
                    # Untargeted ASR is defined relative to the victim's
                    # original hard label, not an assumed ground-truth label.
                    misclassified += int(adv_pred != clean_pred)

                    ssim_score = pytorch_msssim.ssim(img_c.unsqueeze(0), img_a.unsqueeze(0), data_range=1.0).item()
                    ssim_list.append(ssim_score)

                    l2 = torch.norm(img_a - img_c).item()
                    linf = (img_a - img_c).abs().max().item()
                    l2_list.append(l2)
                    linf_list.append(linf)

                    mse = ((img_a - img_c) ** 2).mean().item()
                    psnr = 10 * math.log10(1.0 / max(1e-8, mse))
                    psnr_list.append(psnr)

    asr = (misclassified / float(total)) * 100.0
    clean_accuracy = clean_correct / float(total)
    clean_ci = wilson_interval(clean_correct, total)
    asr_ci = wilson_interval(misclassified, total)
    mean_ssim = np.mean(ssim_list)
    mean_psnr = np.mean(psnr_list)
    mean_l2 = np.mean(l2_list)
    mean_linf = np.mean(linf_list)

    # The provided official rubric weights ASR and imperceptibility equally,
    # but does not define a numeric SSIM cutoff. Keep 0.93 as an internal
    # quality goal, not an official pass/fail requirement.
    passed = asr / 100.0 > TARGET_ASR and clean_accuracy == 1.0
    print(f"Total Images Audited          : {total}")
    print(f"Clean Model Accuracy          : {clean_accuracy * 100:.2f}%")
    print(f"Clean Accuracy 95% CI         : [{clean_ci[0] * 100:.2f}%, {clean_ci[1] * 100:.2f}%]")
    print(f"Successfully Misclassified    : {misclassified} / {total}")
    print(f"Attack Success Rate (ASR)     : {asr:.2f}% (Target: > {TARGET_ASR * 100:.0f}%)")
    print(f"ASR 95% CI                    : [{asr_ci[0] * 100:.2f}%, {asr_ci[1] * 100:.2f}%]")
    print(f"Mean SSIM (Imperceptibility)  : {mean_ssim:.4f} (Internal quality goal: > {INTERNAL_SSIM_GOAL:.2f})")
    print("Official Problem 2 rubric     : ASR 50% + imperceptibility 50%")
    print(f"Mean PSNR                     : {mean_psnr:.2f} dB")
    print(f"Mean L2 Distortion            : {mean_l2:.4f}")
    print(f"Mean Linf Distortion          : {mean_linf:.4f}")
    print(f"Disk Reload Attack Check      : {'PASSED' if asr == 100.0 else 'FAILED'}")
    print(f"ASR Gate                      : {'PASSED' if asr / 100.0 > TARGET_ASR else 'FAILED'}")
    print(f"Internal SSIM Goal            : {'PASSED' if mean_ssim > INTERNAL_SSIM_GOAL else 'NOT MET'}")
    print(f"Hard-Metric Status            : {'PASSED' if passed else 'FAILED'}")
    print("=" * 65)
    return {
        "total": total,
        "clean_accuracy": clean_accuracy,
        "clean_accuracy_ci": clean_ci,
        "asr": asr / 100.0,
        "asr_ci": asr_ci,
        "ssim": float(mean_ssim),
        "ssim_median": float(np.median(ssim_list)),
        "ssim_min": float(np.min(ssim_list)),
        "ssim_below_0_8": int(np.sum(np.array(ssim_list) < 0.8)),
        "psnr": float(mean_psnr),
        "l2": float(mean_l2),
        "linf": float(mean_linf),
        "linf_median_255": float(np.median(linf_list) * 255),
        "passed": passed,
    }

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Audit disk-reloaded adversarial images.")
    parser.add_argument("--adv-dir", type=Path, default=base_dir / "adv_images")
    parser.add_argument("--clean-dir", type=Path, default=base_dir / "p2_data" / "clean_images")
    parser.add_argument("--model", type=Path, default=base_dir / "p2_data" / "cnn_model.pt")
    parser.add_argument("--limit", type=int, default=IMAGES_PER_CLASS)
    parser.add_argument("--queries", type=Path, default=base_dir / "hsja_queries.csv",
                        help="Per-image query log to summarise, if present")
    parser.add_argument("--json", type=Path, default=base_dir / "audit_results.json",
                        help="Where to write the measured metrics")
    args = parser.parse_args()
    result = audit(
        args.adv_dir,
        args.clean_dir,
        args.model,
        args.limit,
    )
    if args.queries.exists():
        with open(args.queries, newline="") as handle:
            queries = [int(row["queries"]) for row in csv.DictReader(handle)]
        result["refined_images"] = len(queries)
        result["refinement_queries_total"] = int(sum(queries))
        result["refinement_queries_mean"] = float(np.mean(queries))
        print(f"Refinement queries logged     : {sum(queries):,} over {len(queries)} images")
    result = {k: list(v) if isinstance(v, tuple) else v for k, v in result.items()}
    args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0 if result["passed"] else 1)
