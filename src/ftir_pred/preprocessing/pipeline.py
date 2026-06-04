import numpy as np
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

    @property
    def x_loadings_(self) -> np.ndarray:
        return self._pls.x_loadings_


def make_pipeline(model, scale: bool = True, apply_pls: bool = True, n_components: int = 10) -> Pipeline:
    """
    Build a sklearn Pipeline that wraps preprocessing + model.

    By keeping scaling and PLS inside the pipeline, both are refitted on each
    CV fold's training data — preventing spectral variance leakage.
    """
    if not scale and not apply_pls:
        raise ValueError("At least one of scale or apply_pls must be True.")

    steps = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    if apply_pls:
        steps.append(("pls", PLSTransformer(n_components=n_components)))
    steps.append(("model", model))
    return Pipeline(steps)


def prefix_params(params: dict, prefix: str = "model__") -> dict:
    """Prefix parameter names for use inside a sklearn Pipeline."""
    return {f"{prefix}{k}": v for k, v in params.items()}


def get_pls_loadings(pipeline: Pipeline) -> np.ndarray | None:
    """Extract PLS x-loadings from a fitted pipeline (None if no PLS step)."""
    if "pls" in pipeline.named_steps:
        return pipeline.named_steps["pls"].x_loadings_
    return None
