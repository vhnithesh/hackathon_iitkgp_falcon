# Problem 1: Backdoor Detection Using Hardware Performance Counters

Unsupervised detector trained only on the 800 clean traces in `trace.csv`
(`cache-references`, `cycles`, `LLC-loads`). It combines engineered
microarchitectural ratios with an ensemble of six anomaly detectors
(autoencoder, Isolation Forest, One-Class SVM, ECOD, GMM, robust covariance).

Synthetic-anomaly holdout (200 clean + 400 synthetic): **AUROC 0.9982, F1 0.9851,
TPR 99.0%, FPR 4.0%, accuracy 98.0%**. See [REPORT.md](REPORT.md) for the
method, ablation, sensitivity and drift analysis, and the
[PDF report](../output/pdf/Problem_1_HPC_Backdoor_Detection.pdf).

## Layout

```
problem_1_hpc_backdoor/
├── trace.csv                         # 800 clean training traces
├── hpc_correlation_heatmap.png       # EDA
├── hpc_pca_scatter.png               # EDA
├── src/
│   ├── eda.py                        # exploratory analysis and figures
│   ├── feature_engineering.py        # ratio + log features
│   ├── models.py                     # ensemble detector
│   ├── train.py                      # holdout benchmark, final fit, save
│   ├── evaluate.py                   # predict on any trace CSV
│   ├── ablation.py                   # raw vs engineered vs ensemble
│   └── adaptive_benchmark.py         # drift / evasion stress test
├── saved_models/
│   ├── hpc_backdoor_detector.joblib  # final detector (fit on all 800 rows)
│   └── detector_model.pkl            # same detector, pickle format
├── ablation_results.csv
├── adaptive_results.csv
└── REPORT.md
```

## Usage

```bash
python src/train.py
python src/evaluate.py --input <path_to_test.csv> --output predictions.csv
```

The input CSV needs the three counter columns; any other columns are ignored.
`predicted_label` is 1 for a suspected backdoor trace and 0 for clean.

## Evaluating on the private validation set

From the repository root (relative paths are resolved from the current directory):

```bash
python problem_1_hpc_backdoor/src/evaluate.py --input private_traces.csv --output predictions.csv
```

`predictions.csv` has one row per input trace, in input order:
`predicted_label` (0/1) for accuracy, TPR, FPR and F1, and `anomaly_score`
(higher means more anomalous) for AUROC. A label column in the input is
ignored.

To use the saved model from Python, put `problem_1_hpc_backdoor/` on the
import path first, because the pickle references the `src` package:

```python
import sys, joblib, pandas as pd
sys.path.insert(0, "problem_1_hpc_backdoor")
detector = joblib.load("problem_1_hpc_backdoor/saved_models/hpc_backdoor_detector.joblib")
df = pd.read_csv("private_traces.csv")
labels = detector.predict(df)             # 1 = backdoor
scores = detector.decision_function(df)   # for AUROC
```
