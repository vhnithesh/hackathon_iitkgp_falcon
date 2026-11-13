import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class HPCFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Physics-grounded feature engineering based on microarchitectural ratios.
    Extracts derived ratios that capture microarchitectural bottlenecks independently of CPU clock frequency:
    1. LLC Spill Ratio: LLC-loads / cache-references
    2. Event Density (IPC Proxy): cache-references / cycles
    3. Memory Bus Transaction Saturation: LLC-loads / cycles
    4. Memory Stall Penalty (CPI Proxy): cycles / cache-references
    5. Fixed Logarithmic transforms for heavy-tailed columns.
    """
    REQUIRED_COLUMNS = ('cache-references', 'cycles', 'LLC-loads')

    def __init__(self):
        self.feature_names = []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            missing = [column for column in self.REQUIRED_COLUMNS if column not in X.columns]
            if missing:
                raise ValueError(
                    f"Input is missing required HPC columns: {missing}. "
                    f"Expected {list(self.REQUIRED_COLUMNS)}."
                )
            # Ignore unrelated columns such as labels or timestamps. This keeps the
            # feature schema identical at training and inference time.
            df = X.loc[:, self.REQUIRED_COLUMNS].copy()
        else:
            values = np.asarray(X)
            if values.ndim != 2 or values.shape[1] != len(self.REQUIRED_COLUMNS):
                raise ValueError(
                    f"Array input must have shape (n, {len(self.REQUIRED_COLUMNS)}), got {values.shape}."
                )
            df = pd.DataFrame(values, columns=self.REQUIRED_COLUMNS)

        if not np.isfinite(df.to_numpy(dtype=float)).all():
            raise ValueError("HPC input contains NaN or infinite values.")
        if (df.to_numpy(dtype=float) < 0).any():
            raise ValueError("HPC counters must be non-negative.")

        # 1. Derived ratios (microarchitectural metrics)
        df['llc_per_cache_ref'] = df['LLC-loads'] / (df['cache-references'] + 1.0)
        df['cache_ref_per_cycle'] = df['cache-references'] / (df['cycles'] + 1.0)
        df['llc_per_cycle'] = df['LLC-loads'] / (df['cycles'] + 1.0)
        df['cycles_per_cache_ref'] = df['cycles'] / (df['cache-references'] + 1.0)

        # 2. Consistent Log-transforms for raw counters (always fixed columns)
        df['log_cache_refs'] = np.log1p(np.maximum(0, df['cache-references']))
        df['log_cycles'] = np.log1p(np.maximum(0, df['cycles']))
        df['log_llc_loads'] = np.log1p(np.maximum(0, df['LLC-loads']))

        self.feature_names = list(df.columns)
        return df.values
