from sklearn.pipeline import Pipeline
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

MODEL_REGISTRY = {
    "linear_regression": LinearRegression,
    "ridge": Ridge,
    "lasso": Lasso,
    "elasticnet": ElasticNet,
    "decision_tree": DecisionTreeRegressor,
    "random_forest": RandomForestRegressor,
    "xgboost": XGBRegressor,
    "lightgbm": LGBMRegressor,
    "catboost": CatBoostRegressor
}

def build_model(model_name, preprocessor, model_params=None):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")

    if model_params is None:
        model_params = {}

    model_class = MODEL_REGISTRY[model_name]

    model = model_class(**model_params)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])

    return pipeline