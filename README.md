# IIT Kharagpur ML Security Hackathon 2026 — Submission

**Event:** A Hands-on ML Security Hackathon: Applications and Challenges
(IIT Kharagpur, co-located with AsianHOST 2026)  
**Tracks:** Problem 1 (HPC backdoor detection) and Problem 2 (black-box adversarial attack)

Short reports: [Problem 1 PDF](output/pdf/Problem_1_HPC_Backdoor_Detection.pdf) ·
[Problem 2 PDF](output/pdf/Problem_2_Black_Box_Adversarial_Attack.pdf)

---

## Results

| Track | Metric | Target | Measured |
| :--- | :--- | :--- | :--- |
| **Problem 1** (synthetic holdout) | AUROC | > 0.95 | **0.9982** |
| | F1 | > 0.90 | **0.9851** |
| | TPR | > 90% | **99.00%** |
| | FPR | < 10% | **4.00%** |
| | Accuracy | — | **98.00%** |
| **Problem 2** (1,000 PNGs reloaded from disk) | Attack success rate | 50% of score | **100.00%** |
| | Mean SSIM (imperceptibility) | 50% of score | **0.9469** |
| | Mean PSNR | — | **31.17 dB** |
| | Logged queries (stage-3 refinement) | tiebreaker: minimise | 353,140 over 320 images |

Problem 1 is evaluated on 200 held-out clean traces plus 400 synthetic
anomalies, because no backdoor traces are provided; the organisers' private
set remains the real test. Problem 2 metrics are measured on the saved PNGs
after reloading, so 8-bit quantization is included.

---

## Approach

**Problem 1: anomaly detection on hardware performance counters.**
The three counters (`cache-references`, `cycles`, `LLC-loads`) are expanded
into microarchitectural ratios (LLC-loads per reference, references per cycle,
LLC-loads per cycle, cycles per reference) plus log counts. Six unsupervised
detectors are fit on clean traces only: autoencoder, Isolation Forest,
One-Class SVM, ECOD, GMM and robust covariance. Their min-max-normalised scores
are combined by weighted average and thresholded at the 99th percentile of
clean training scores. An ablation shows that both the engineered features and
the ensemble lower the false-positive rate. A drift stress test shows the
detector assumes test traces come from the same measurement setup.
Details: [problem_1_hpc_backdoor/REPORT.md](problem_1_hpc_backdoor/REPORT.md).

**Problem 2: hard-label (decision-based) attack.**
The attack uses only predicted labels. It runs in three stages:
1. A smooth-direction search with boundary bisection.
2. 8-bit bisection toward the clean image.
3. HopSkipJump-style refinement of the lowest-SSIM images. It estimates the boundary normal from the labels of random probes and walks along the boundary toward the clean image.

Every saved image is re-checked after 8-bit rounding. Details:
[problem_2_cnn_adversarial/REPORT.md](problem_2_cnn_adversarial/REPORT.md).

---

## Repository

```text
├── verify_all.py                  # one-command audit of both problems
├── build_reports.py               # regenerates the PDFs (Problem 2 numbers from audit_results.json)
├── build_scorecard.py             # regenerates SCORECARD.md from the result files
├── SCORECARD.md                   # all measured evidence in one place
├── output/pdf/                    # the two short reports
├── problem_1_hpc_backdoor/        # trace.csv, src/, saved_models/, results CSVs, REPORT.md
└── problem_2_cnn_adversarial/     # p2_code.ipynb, adv_images/, attack + refinement + audit scripts
```

## Reproduce

```bash
python -m pip install -r requirements.txt
python verify_all.py               # Problem 1 holdout benchmark + Problem 2 disk audit
```

Further analyses:

```bash
python problem_1_hpc_backdoor/src/ablation.py
python problem_1_hpc_backdoor/src/adaptive_benchmark.py
python problem_2_cnn_adversarial/audit_adversarial.py
python build_scorecard.py && python build_reports.py
```
