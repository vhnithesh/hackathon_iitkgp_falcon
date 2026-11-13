# Problem 1: Detection of Backdoor Attacks Using Hardware Performance Counters (HPCs)

**Competition:** ML Security Hackathon 2026 — AsianHOST & IIT Kharagpur  
**Track:** Problem 1 — Detection of Backdoor Attacks using HPCs  
**Approach:** Engineered HPC features + six-detector unsupervised anomaly ensemble

---

## 1. Summary

A backdoored model behaves normally until a trigger is presented; executing the
trojan path then changes how the inference uses the cache hierarchy and how
long it takes. We learn the distribution of benign inferences from the 800
clean traces in `trace.csv` (three counters: `cache-references`, `cycles`,
`LLC-loads`) and flag traces that deviate from it. No backdoor traces are used
for training.

The detector is an ensemble of six unsupervised models over engineered
features:

1. **Autoencoder** (PyTorch, 10→32→8→32→10, 50 epochs, Adam 1e-3): score = reconstruction MSE
2. **Isolation Forest** (200 trees)
3. **One-Class SVM** (RBF kernel, ν = 0.05)
4. **ECOD** (PyOD, empirical-CDF outlier score)
5. **Gaussian Mixture Model** (6 components, full covariance): score = negative log-likelihood
6. **Robust covariance / Elliptic Envelope** (FastMCD), which exploits the 0.92 correlation between `cache-references` and `LLC-loads`

### Validation (synthetic anomalies; not the private test set)

| Metric | Target | Measured (200 clean holdout + 400 synthetic anomalies) |
| :--- | :--- | :--- |
| AUROC | > 0.95 | **0.9982** |
| F1 | > 0.90 | **0.9851** |
| TPR | > 90% | **99.00%** |
| FPR | < 10% | **4.00%** |
| Accuracy | — | **98.00%** |

Synthetic anomalies are proxies for real backdoor traces, so these numbers are
evidence that the pipeline works, not a prediction of private-set performance.

---

## 2. Method

```
[cache-references, cycles, LLC-loads]
                 │
                 ▼
  Exploratory analysis (eda.py)
  - correlation heatmap: cache-refs ↔ LLC-loads r = 0.92
  - PCA scatter: clean traces form one tight cluster
                 │
                 ▼
  Feature engineering (feature_engineering.py)
  ├── LLC-loads / cache-references    (fraction of references reaching LLC)
  ├── cache-references / cycles       (memory-event density)
  ├── LLC-loads / cycles              (LLC traffic rate)
  ├── cycles / cache-references       (cycles per memory reference)
  ├── log1p of each raw counter
  └── the 3 raw counters             (10 features in total)
                 │
                 ▼
  StandardScaler (fit on clean training rows)
                 │
                 ▼
  Six detectors, each fit on clean rows only
  AE 0.30 · IsoForest 0.20 · OC-SVM 0.15 · ECOD 0.15 · GMM 0.10 · MCD 0.10
                 │
                 ▼
  Min-max normalise each score with its clean-training range,
  weighted sum → ensemble score
                 │
                 ▼
  Threshold τ = 99th percentile of clean training scores
  (saved model: τ = 0.38380)   score > τ ⇒ backdoor (1)
```

The threshold is set on the same rows the detectors were fit on, so the clean
false-positive rate on unseen traces is higher than the nominal 1%. It was 4.0%
on the disjoint holdout (95% CI 2.0–7.7%), which is still below the 10% target.

---

## 3. Evidence

All experiments fit on 600 clean rows and evaluate on the 200 disjoint clean
rows plus 400 synthetic anomalies drawn from four profiles: elevated cache
activity, execution stalls, LLC-load surges, and a subtle joint shift.

### Ablation (`src/ablation.py`)

| Detector | Accuracy | F1 | AUROC | FPR | TPR |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Isolation Forest on raw counters | 0.962 | 0.972 | 0.991 | 7.5% | 98.0% |
| Isolation Forest on engineered features | 0.973 | 0.980 | 0.991 | 5.0% | 98.5% |
| **Full ensemble** | **0.980** | **0.985** | **0.998** | **4.0%** | **99.0%** |

Engineered features cut the false-positive rate, and the ensemble improves
again on both FPR and AUROC.

### Sensitivity to small shifts

The table gives AUROC for anomalies built by scaling one or all counters of
clean traces by a small factor.

| Shift applied | +2% | +5% | +10% |
| :--- | ---: | ---: | ---: |
| All counters | 0.980 | 1.000 | 1.000 |
| `cycles` only | 0.983 | 1.000 | 1.000 |
| `cache-references` only | 0.796 | 0.983 | 0.998 |
| `LLC-loads` only | 0.629 | 0.955 | 0.994 |

Shifts of 5% or more in any counter are separated with AUROC ≥ 0.95. A 2%
change in LLC-loads alone falls inside normal run-to-run variation.

### Limitation: measurement drift (`src/adaptive_benchmark.py`)

| Scenario applied to clean and anomalous traces | Clean FPR | Backdoor TPR |
| :--- | ---: | ---: |
| None | 0.040 | 0.990 |
| Clip each counter to its 2–98th percentile | 0.025 | 0.990 |
| Global rescaling of all counters (lognormal σ = 0.8) | 0.990 | 1.000 |
| ±8% independent counter noise | 0.975 | 1.000 |
| Clock drift (cycles × 0.65–1.45) | 0.990 | 1.000 |
| LLC-load / cycle manipulation | 1.000 | 1.000 |

The detector models the exact benign distribution in `trace.csv`, including
log raw counts, so clean traces taken on a different machine, frequency or
counter configuration are also flagged. It assumes that test traces are
collected with the same setup as the training traces, as the problem implies.
If they are not, the detector should be refit on clean traces from the new
setup.

---

## 4. Reproduce

```bash
python src/eda.py                  # heatmap + PCA figures
python src/train.py                # holdout benchmark, then refit on all 800 rows and save
python src/ablation.py             # ablation table
python src/adaptive_benchmark.py   # drift / evasion table
python src/evaluate.py --input <test.csv> --output predictions.csv
```

`evaluate.py` writes one row per trace with `predicted_label` (1 = backdoor),
`anomaly_score` (score − τ), and `prob_backdoor`.
