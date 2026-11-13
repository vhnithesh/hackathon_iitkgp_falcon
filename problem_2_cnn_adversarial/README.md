# Problem 2: Hard-Label Black-Box Adversarial Attack

One adversarial PNG for each of the 1,000 clean CIFAR-10 images, generated
using only the victim's predicted labels.

**Disk-reloaded audit: 100.00% attack success, mean SSIM 0.9469, mean PSNR
31.17 dB.** See [REPORT.md](REPORT.md) for the method and the
[PDF report](../output/pdf/Problem_2_Black_Box_Adversarial_Attack.pdf).

## Layout

```
problem_2_cnn_adversarial/
├── p2_code.ipynb            # completed notebook: attack code + audit of saved images
├── adv_images/              # 1,000 submitted adversarial PNGs (<class>_<i>_adv.png)
├── p2_data/
│   ├── clean_images/        # 1,000 clean images (provided)
│   ├── adv_images/          # identical copy of adv_images/
│   ├── cnn_model.pt         # victim model (provided)
│   └── readme.txt
├── attack.py                # stage 1: directional search + bisection
├── refine_existing.py       # stage 2: 8-bit bisection toward the clean image
├── hsja_refine.py           # stage 3: HopSkipJump-style hard-label refinement
├── refine_worst.py          # runs stage 3 on the lowest-SSIM images, logs queries
├── targeted_search.py       # stronger stage-1 search for the worst images
├── audit_adversarial.py     # disk-reloaded ASR / SSIM / PSNR audit
├── audit_results.json       # latest audit output
├── hsja_queries.csv         # per-image stage-3 query log
├── quality_report.py        # per-image quality ranking
├── benchmark_attack.py      # stage-1 probe on a subset
├── generate_gallery.py
└── adversarial_gallery.png  # clean vs adversarial vs 10x difference
```

## Verify

```bash
python audit_adversarial.py
```

This reloads every PNG, checks that each is misclassified, and reports SSIM,
PSNR, L2 and L∞. It exits non-zero if the attack success rate is 80% or lower
or if any clean image is misclassified.
