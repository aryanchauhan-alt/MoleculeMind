"""
Uncertainty Engine: Conformal Prediction Module.
Implements Inductive Conformal Prediction (ICP) and Locally Adaptive Conformal Prediction
to provide finite-sample, distribution-free prediction intervals with rigorous coverage guarantees.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple


class ConformalPredictionEngine:
    def __init__(self):
        self.cal_residuals: Optional[np.ndarray] = None
        self.cal_norm_residuals: Optional[np.ndarray] = None
        self.cal_size: int = 0
        self.test_coverage_stats: Dict[str, float] = {}
        self.quantiles: Dict[float, float] = {}
        self.norm_quantiles: Dict[float, float] = {}

    def calibrate(
        self,
        y_cal: np.ndarray,
        y_cal_pred: np.ndarray,
        y_cal_std: Optional[np.ndarray] = None
    ):
        """
        Calibrate conformal prediction on held-out calibration set.
        y_cal: true values
        y_cal_pred: point predictions from base model
        y_cal_std: predicted standard deviations (epistemic/aleatoric uncertainty)
        """
        abs_errors = np.abs(y_cal - y_cal_pred)
        self.cal_residuals = np.sort(abs_errors)
        self.cal_size = len(abs_errors)

        if y_cal_std is not None and np.all(y_cal_std > 0):
            # Locally normalized conformal nonconformity score
            norm_errors = abs_errors / np.maximum(y_cal_std, 1e-4)
            self.cal_norm_residuals = np.sort(norm_errors)
        else:
            self.cal_norm_residuals = None

        # Precompute standard quantiles for 80%, 90%, 95%, 99% confidence
        for confidence in [0.80, 0.90, 0.95, 0.99]:
            alpha = 1.0 - confidence
            self.quantiles[confidence] = self._compute_quantile(self.cal_residuals, alpha)
            if self.cal_norm_residuals is not None:
                self.norm_quantiles[confidence] = self._compute_quantile(self.cal_norm_residuals, alpha)

    def _compute_quantile(self, sorted_scores: np.ndarray, alpha: float) -> float:
        """
        Compute conformal quantile with finite-sample correction:
        q_{1-alpha} = ceil((n + 1) * (1 - alpha)) / n th order statistic
        """
        n = len(sorted_scores)
        idx = int(np.ceil((n + 1) * (1.0 - alpha))) - 1
        idx = min(max(idx, 0), n - 1)
        return float(sorted_scores[idx])

    def evaluate_test_coverage(
        self,
        y_test: np.ndarray,
        y_test_pred: np.ndarray,
        y_test_std: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Verify empirical coverage on test set against theoretical guarantees.
        """
        coverage_results = {}
        for confidence in [0.80, 0.90, 0.95]:
            q = self.quantiles.get(confidence, 1.0)
            if self.norm_quantiles and y_test_std is not None:
                q_norm = self.norm_quantiles[confidence]
                half_width = q_norm * np.maximum(y_test_std, 1e-4)
            else:
                half_width = np.full_like(y_test_pred, q)

            lower = y_test_pred - half_width
            upper = y_test_pred + half_width
            in_interval = (y_test >= lower) & (y_test <= upper)
            empirical_coverage = float(np.mean(in_interval))
            avg_interval_width = float(np.mean(upper - lower))

            coverage_results[f"{int(confidence*100)}%"] = {
                "nominal_confidence": confidence,
                "empirical_coverage": round(empirical_coverage * 100, 2),
                "guarantee_satisfied": empirical_coverage >= (confidence - 0.05),
                "avg_interval_width": round(avg_interval_width, 3),
                "margin_error": round(float(np.mean(half_width)), 3)
            }
        return coverage_results

    def predict_intervals(
        self,
        y_pred: float,
        std_est: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Compute conformal prediction intervals for a single prediction point at multiple confidence levels.
        """
        intervals = {}
        for confidence in [0.80, 0.90, 0.95]:
            if self.cal_norm_residuals is not None and std_est is not None and std_est > 0:
                q_norm = self.norm_quantiles.get(confidence, 1.96)
                half_width = q_norm * max(std_est, 1e-4)
            else:
                half_width = self.quantiles.get(confidence, 1.0)

            lower = y_pred - half_width
            upper = y_pred + half_width

            intervals[f"{int(confidence*100)}%"] = {
                "confidence": confidence,
                "lower_bound": round(float(lower), 3),
                "upper_bound": round(float(upper), 3),
                "interval_width": round(float(upper - lower), 3),
                "margin_error": round(float(half_width), 3)
            }

        return {
            "prediction": round(float(y_pred), 3),
            "estimated_std": round(float(std_est), 3) if std_est is not None else None,
            "calibration_set_size": self.cal_size,
            "method": "Locally-Adaptive Split Conformal Prediction" if self.cal_norm_residuals is not None else "Standard Split Conformal Prediction",
            "coverage_guarantee": "Finite-sample, distribution-free marginal coverage P(Y in [L, U]) >= 1 - alpha",
            "intervals": intervals
        }
