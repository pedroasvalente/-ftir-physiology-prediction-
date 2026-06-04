from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge

BASELINE_MODELS = {
    "dummy_mean": ("Dummy (mean)", DummyRegressor(strategy="mean")),
    "ridge": ("Ridge", Ridge(alpha=1.0)),
    "pls_baseline": ("PLS (3 comp)", PLSRegression(n_components=3)),
}
