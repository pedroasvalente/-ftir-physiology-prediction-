import numpy as np
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class PLSTransformer(BaseEstimator, TransformerMixin):
    """
    Wrapper around PLSRegression that conforms to the sklearn Transformer API.
    fit_transform(X, y) returns only X_scores (2D), not the (X_scores, Y_scores) tuple
    that PLSRegression.fit_transform would otherwise return.
    """

    def __init__(self, n_components: int = 10):
        self.n_components = n_components

    def fit(self, X, y=None):
        self._pls = PLSRegression(n_components=self.n_components)
        self._pls.fit(X, y)
        return self

    def transform(self, X):
        return self._pls.transform(X)

    def vip_scores(self) -> np.ndarray:
        """
        Variable Importance in Projection (VIP) scores.

        VIP_j = sqrt(p * sum_h(W*_jh^2 * SSY_h) / SSY_total)

        where W* are the normalised x-weights, and SSY_h is the variance of y
        explained by component h. Scores > 1 are conventionally considered important.
        """
        T = self._pls.x_scores_     # (n_samples, n_components)
        W = self._pls.x_weights_    # (n_features, n_components)
        Q = self._pls.y_loadings_   # (n_targets,  n_components)
        p = W.shape[0]

        s = np.diag(T.T @ T @ Q.T @ Q)
        total_s = s.sum()
        if total_s == 0:
            return np.ones(p)

        col_norms = np.linalg.norm(W, axis=0, keepdims=True)
        col_norms = np.where(col_norms == 0, 1.0, col_norms)
        w_norm = W / col_norms

        return np.sqrt(p * (w_norm ** 2 @ s) / total_s)


class SNVTransformer(BaseEstimator, TransformerMixin):
    """Standard Normal Variate: centres and scales each spectrum individually."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True)
        std = np.where(std == 0, 1.0, std)
        return (X - mean) / std


class SavitzkyGolayDerivative(BaseEstimator, TransformerMixin):
    """2nd derivative via Savitzky-Golay filter — removes baseline and sharpens peaks."""

    def __init__(self, window_length: int = 11, polyorder: int = 2, deriv: int = 2):
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return savgol_filter(X, self.window_length, self.polyorder, deriv=self.deriv, axis=1)


def make_pipeline(model, scale: bool = True, apply_pls: bool = True,
                  n_components: int = 10, spectral_preproc: str = "none") -> Pipeline:
    """
    Build a sklearn Pipeline that wraps preprocessing + model.

    Keeping scaling and PLS inside the pipeline ensures both are refitted on each
    CV fold's training data, preventing spectral variance leakage.
    """
    if not scale and not apply_pls:
        raise ValueError("At least one of scale or apply_pls must be True.")

    steps = []
    if "snv" in spectral_preproc:
        steps.append(("snv", SNVTransformer()))
    if "derivative" in spectral_preproc:
        steps.append(("derivative", SavitzkyGolayDerivative()))
    if scale:
        steps.append(("scaler", StandardScaler()))
    if apply_pls:
        steps.append(("pls", PLSTransformer(n_components=n_components)))
    steps.append(("model", model))
    return Pipeline(steps)


def prefix_params(params: dict, prefix: str = "model__") -> dict:
    """Prefix parameter names for use inside a sklearn Pipeline."""
    return {f"{prefix}{k}": v for k, v in params.items()}


def get_pls_vip(pipeline: Pipeline) -> np.ndarray | None:
    """Return VIP scores from the fitted PLS step, or None if no PLS step."""
    if "pls" in pipeline.named_steps:
        return pipeline.named_steps["pls"].vip_scores()
    return None
