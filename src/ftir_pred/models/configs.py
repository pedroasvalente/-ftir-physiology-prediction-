from dataclasses import dataclass, field

from skopt.space import Categorical, Integer, Real
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from ftir_pred.config import RANDOM_SEED


@dataclass
class ModelConfig:
    name: str
    display_name: str
    model_fn: type
    model_kwargs: dict = field(default_factory=dict)
    grid_params: dict = field(default_factory=dict)
    bayes_params: dict = field(default_factory=dict)

    def get_model(self):
        return self.model_fn(random_state=RANDOM_SEED, **self.model_kwargs)


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "random_forest": ModelConfig(
        name="random_forest",
        display_name="Random Forest",
        model_fn=RandomForestRegressor,
        grid_params={
            "n_estimators": [100, 200],
            "max_depth": [4, 8, 12, None],
            "max_features": ["sqrt", "log2"],
            "criterion": ["squared_error", "absolute_error"],
        },
        bayes_params={
            "n_estimators": Integer(50, 300),
            "max_depth": Integer(3, 15),
            "min_samples_split": Integer(2, 20),
            "min_samples_leaf": Integer(1, 10),
            "bootstrap": Categorical([True, False]),
        },
    ),
    "mlp": ModelConfig(
        name="mlp",
        display_name="MLP Regressor",
        model_fn=MLPRegressor,
        model_kwargs={
            "max_iter": 3000,
            "early_stopping": True,
            "validation_fraction": 0.1,
        },
        grid_params={
            "hidden_layer_sizes": [(50,), (100,), (50, 50)],
            "activation": ["relu", "tanh"],
            "solver": ["adam", "sgd"],
            "alpha": [0.0001, 0.001],
            "learning_rate": ["constant", "adaptive"],
        },
        bayes_params={
            "hidden_layer_sizes": Categorical([(50,), (100,), (50, 50), (100, 50)]),
            "activation": Categorical(["relu", "tanh", "logistic"]),
            "solver": Categorical(["adam", "sgd"]),
            "alpha": Real(1e-6, 1e-1, prior="log-uniform"),
            "learning_rate_init": Real(1e-4, 1e-2, prior="log-uniform"),
        },
    ),
    "decision_tree": ModelConfig(
        name="decision_tree",
        display_name="Decision Tree",
        model_fn=DecisionTreeRegressor,
        grid_params={
            "criterion": ["squared_error", "absolute_error"],
            "splitter": ["best", "random"],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        bayes_params={
            "criterion": Categorical(["squared_error", "absolute_error"]),
            "max_depth": Integer(1, 50),
            "min_samples_split": Integer(2, 50),
            "min_samples_leaf": Integer(1, 20),
            "max_features": Categorical([None, "sqrt", "log2"]),
        },
    ),
    "xgboost": ModelConfig(
        name="xgboost",
        display_name="XGBoost",
        model_fn=XGBRegressor,
        model_kwargs={"objective": "reg:squarederror", "eval_metric": "rmse"},
        grid_params={
            "n_estimators": [50, 100],
            "max_depth": [3, 6, 10],
            "learning_rate": [0.01, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        },
        bayes_params={
            "n_estimators": Integer(50, 500),
            "max_depth": Integer(1, 15),
            "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
            "subsample": Real(0.5, 1.0),
            "colsample_bytree": Real(0.5, 1.0),
            "min_child_weight": Integer(1, 10),
            "gamma": Real(0, 5),
        },
    ),
}
