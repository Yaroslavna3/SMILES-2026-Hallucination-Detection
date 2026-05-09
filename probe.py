"""
probe.py — Hallucination probe classifier (student-implemented).

Implements ``HallucinationProbe``, a binary MLP that classifies feature
vectors as truthful (0) or hallucinated (1).  Called from ``solution.py``
via ``evaluate.run_evaluation``.  All four public methods (``fit``,
``fit_hyperparameters``, ``predict``, ``predict_proba``) must be implemented
and their signatures must not change.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


class HallucinationProbe(nn.Module):
    """Binary classifier that detects hallucinations from hidden-state features.

    Extends ``torch.nn.Module``; the default architecture is a single
    hidden-layer MLP with ``StandardScaler`` pre-processing.  The network is
    built lazily in ``fit()`` once the feature dimension is known.
    """

    def __init__(self) -> None:
        super().__init__()
        self._net: nn.Sequential | None = None  # built lazily in fit()
        self._scaler = StandardScaler()
        self._pca: PCA | None = None
        self._models: list[LogisticRegression] = []
        self._threshold: float = 0.5  # tuned by fit_hyperparameters()

    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the network definition below.
    # ------------------------------------------------------------------
    def _build_network(self, input_dim: int) -> None:
        """Instantiate the network layers.

        Called once at the start of ``fit()`` when ``input_dim`` is known.

        Args:
            input_dim: Feature vector dimensionality.
        """
        self._net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — returns raw logits of shape ``(n_samples,)``.

        Args:
            x: Float tensor of shape ``(n_samples, feature_dim)``.

        Returns:
            1-D tensor of raw (pre-sigmoid) logits.
        """
        if self._net is None:
            raise RuntimeError(
                "Network has not been built yet. Call fit() before forward()."
            )
        return self._net(x).squeeze(-1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Train the probe on labelled feature vectors.

        Scales features with ``StandardScaler``, builds the network if needed,
        and optimises with Adam + ``BCEWithLogitsLoss``.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.
            y: Integer label vector of shape ``(n_samples,)``; 0 = truthful,
               1 = hallucinated.

        Returns:
            ``self`` (for method chaining).
        """
        X = np.asarray(X, dtype=np.float32)
        y = y.astype(int)

        X_scaled = self._scaler.fit_transform(X)

        max_components = min(X_scaled.shape[0] - 1, X_scaled.shape[1], 160)
        if max_components >= 8 and X_scaled.shape[1] > max_components:
            self._pca = PCA(n_components=max_components, random_state=42)
            X_model = self._pca.fit_transform(X_scaled)
        else:
            self._pca = None
            X_model = X_scaled

        self._models = []
        for c_value in (0.05, 0.1, 0.3, 1.0):
            model = LogisticRegression(
                C=c_value,
                class_weight="balanced",
                max_iter=3000,
                random_state=42,
                solver="liblinear",
            )
            model.fit(X_model, y)
            self._models.append(model)

        self._threshold = self._best_threshold_from_probs(
            self._predict_proba_from_prepared(X_model), y
        )
        return self

    def _predict_proba_from_prepared(self, X_model: np.ndarray) -> np.ndarray:
        probs = [model.predict_proba(X_model)[:, 1] for model in self._models]
        return np.mean(probs, axis=0)

    def _best_threshold_from_probs(self, probs: np.ndarray, y_true: np.ndarray) -> float:
        candidates = np.unique(np.concatenate([probs, np.linspace(0.05, 0.95, 181)]))

        best_threshold = 0.5
        best_accuracy = -1.0
        best_f1 = -1.0
        for t in candidates:
            y_pred_t = (probs >= t).astype(int)
            acc = accuracy_score(y_true, y_pred_t)
            score = f1_score(y_true, y_pred_t, zero_division=0)
            if acc > best_accuracy or (acc == best_accuracy and score > best_f1):
                best_accuracy = acc
                best_f1 = score
                best_threshold = float(t)

        return best_threshold

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune the decision threshold on a validation set to maximise F1.

        The chosen threshold is stored in ``self._threshold`` and used by
        subsequent ``predict`` calls.  Call this after ``fit`` and before
        ``predict``.

        Args:
            X_val: Validation feature matrix of shape
                   ``(n_val_samples, feature_dim)``.
            y_val: Integer label vector of shape ``(n_val_samples,)``;
                   0 = truthful, 1 = hallucinated.

        Returns:
            ``self`` (for method chaining).
        """
        probs = self.predict_proba(X_val)[:, 1]

        self._threshold = self._best_threshold_from_probs(probs, y_val)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels for feature vectors.

        Uses the decision threshold in ``self._threshold`` (default ``0.5``;
        updated by ``fit_hyperparameters``).

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Integer array of shape ``(n_samples,)`` with values in ``{0, 1}``.
        """
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Array of shape ``(n_samples, 2)`` where column 1 contains the
            estimated probability of the hallucinated class (label 1).
            Used to compute AUROC.
        """
        if not self._models:
            raise RuntimeError("Probe has not been fitted yet.")

        X = np.asarray(X, dtype=np.float32)
        X_scaled = self._scaler.transform(X)
        if self._pca is not None:
            X_scaled = self._pca.transform(X_scaled)

        prob_pos = self._predict_proba_from_prepared(X_scaled)
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)

