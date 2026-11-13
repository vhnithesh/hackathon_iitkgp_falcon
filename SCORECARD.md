# Submission Security Scorecard

Generated: 2026-10-06

This scorecard separates measured evidence from historical or missing results.

## Artifact integrity

- Clean images: 1000
- Adversarial images: 1000
- Quantization verification: performed by `audit_adversarial.py`

## Problem 1 adaptive robustness

| Scenario | Clean FPR | Backdoor TPR |
|---|---:|---:|
| unmodified | 0.040 | 0.990 |
| global_scale | 0.990 | 1.000 |
| counter_noise | 0.975 | 1.000 |
| clock_drift | 0.990 | 1.000 |
| llc_evasion | 1.000 | 1.000 |
| counter_clipping | 0.025 | 0.990 |

## Ablation evidence

| Detector | Accuracy | F1 | AUROC |
|---|---:|---:|---:|
| raw_counters_iforest | 0.962 | 0.971 | 0.991 |
| engineered_iforest | 0.973 | 0.980 | 0.991 |
| full_ensemble | 0.980 | 0.985 | 0.998 |

## Problem 2 disk-reloaded audit

| Measure | Result |
|---|---:|
| Attack success rate | 100.00% |
| Mean / median SSIM | 0.9469 / 0.9556 |
| Images with SSIM < 0.8 | 15 |
| Mean PSNR | 31.17 dB |
| Mean L2 / Linf | 1.7971 / 0.1140 |
| Lowest-SSIM images sent through HSJA refinement | 320 |
| Logged refinement queries | 353,140 |

The rubric assigns 50% to ASR and 50% to imperceptibility; SSIM is the imperceptibility measure reported here. Queries of the stage-1 attack were not logged for this image set. Confidence change is not observable through the hard-label interface.

## Problem 1 synthetic holdout

98.00% accuracy, 99.00% TPR, 4.00% FPR, 0.9851 F1 and 0.9982 AUROC on 200 clean holdout traces plus 400 synthetic anomalies. The drift table above shows the detector assumes test traces come from the same measurement setup.
