import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.mixture import GaussianMixture
from sklearn.covariance import EllipticEnvelope
from pyod.models.ecod import ECOD
from .feature_engineering import HPCFeatureExtractor

# 1. PyTorch Deep Autoencoder (reconstruction-error detector)
class PyTorchAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 8)
        )
        self.dec = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        return self.dec(self.enc(x))

class AutoencoderAnomalyDetector:
    def __init__(self, epochs=50, lr=1e-3, random_state=42):
        self.epochs = epochs
        self.lr = lr
        self.random_state = random_state
        self.model = None

    def fit(self, X):
        torch.manual_seed(self.random_state)
        d = X.shape[1]
        self.model = PyTorchAutoencoder(d)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        X_t = torch.tensor(X, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(X_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            for (batch_x,) in loader:
                optimizer.zero_grad()
                recon = self.model(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                optimizer.step()
        return self

    def score_samples(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            recon = self.model(X_t)
            mse = ((recon - X_t) ** 2).mean(dim=1).cpu().numpy()
        return mse

# 2. Complete Ensemble Detector
class HPCBackdoorDetectorChampionship(BaseEstimator, ClassifierMixin):
    """
    Ensemble combining:
    1. Deep Autoencoder (Reconstruction error)
    2. Isolation Forest (Spatial partitioning)
    3. One-Class SVM (RBF kernel)
    4. ECOD (PyOD Empirical Cumulative Distribution)
    5. Gaussian Mixture Model (full-covariance density model)
    6. FastMCD Elliptic Envelope (Covariance anomaly detection)
    """
    def __init__(self, target_fpr=0.01, random_state=42):
        self.target_fpr = target_fpr
        self.random_state = random_state
        self.extractor = HPCFeatureExtractor()
        self.scaler = StandardScaler()

        self.ae = AutoencoderAnomalyDetector(epochs=50, lr=1e-3, random_state=random_state)
        # One worker is deterministic and portable to restricted Windows
        # runners where process/thread pools may be denied.
        self.iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=random_state, n_jobs=1)
        self.ocsvm = OneClassSVM(nu=0.05, kernel='rbf', gamma='auto')
        self.ecod = ECOD()
        self.gmm = GaussianMixture(n_components=6, covariance_type="full", random_state=random_state)
        self.mcd = EllipticEnvelope(contamination=target_fpr, random_state=random_state)

        # Diverse ensemble weighting
        self.weights = np.array([0.30, 0.20, 0.15, 0.15, 0.10, 0.10])
        self.threshold = None
        self.score_min_ = None
        self.score_max_ = None

    def _prepare_data(self, X):
        feats = self.extractor.transform(X)
        scaled = self.scaler.transform(feats)
        return scaled

    def fit(self, X, y=None):
        feats = self.extractor.fit_transform(X)
        scaled = self.scaler.fit_transform(feats)

        # Fit all detectors
        self.ae.fit(scaled)
        self.iso.fit(scaled)
        self.ocsvm.fit(scaled)
        self.ecod.fit(scaled)
        self.gmm.fit(scaled)
        self.mcd.fit(scaled)

        # Compute raw scores
        s_ae = self.ae.score_samples(scaled)
        s_iso = -self.iso.decision_function(scaled)
        s_ocsvm = -self.ocsvm.decision_function(scaled)
        s_ecod = self.ecod.decision_function(scaled)
        s_gmm = -self.gmm.score_samples(scaled)
        s_mcd = -self.mcd.decision_function(scaled)

        raw_matrix = np.column_stack([s_ae, s_iso, s_ocsvm, s_ecod, s_gmm, s_mcd])
        self.score_min_ = raw_matrix.min(axis=0)
        self.score_max_ = raw_matrix.max(axis=0)

        # Min-max normalise each detector score using clean-training ranges
        norm_scores = (raw_matrix - self.score_min_) / (self.score_max_ - self.score_min_ + 1e-9)
        ensemble_scores = np.dot(norm_scores, self.weights)

        # Set threshold at (1.0 - target_fpr) percentile on clean data
        self.threshold = float(np.percentile(ensemble_scores, (1.0 - self.target_fpr) * 100))
        return self

    def _get_ensemble_scores(self, scaled):
        s_ae = self.ae.score_samples(scaled)
        s_iso = -self.iso.decision_function(scaled)
        s_ocsvm = -self.ocsvm.decision_function(scaled)
        s_ecod = self.ecod.decision_function(scaled)
        s_gmm = -self.gmm.score_samples(scaled)
        s_mcd = -self.mcd.decision_function(scaled)

        raw_matrix = np.column_stack([s_ae, s_iso, s_ocsvm, s_ecod, s_gmm, s_mcd])
        norm_scores = (raw_matrix - self.score_min_) / (self.score_max_ - self.score_min_ + 1e-9)
        return np.dot(norm_scores, self.weights)

    def decision_function(self, X):
        scaled = self._prepare_data(X)
        scores = self._get_ensemble_scores(scaled)
        return scores - self.threshold

    def predict_proba(self, X):
        scaled = self._prepare_data(X)
        scores = self._get_ensemble_scores(scaled)
        logits = (scores - self.threshold) * 10.0
        p_backdoor = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
        p_clean = 1.0 - p_backdoor
        return np.column_stack([p_clean, p_backdoor])

    def predict(self, X):
        scaled = self._prepare_data(X)
        scores = self._get_ensemble_scores(scaled)
        return (scores > self.threshold).astype(int)
