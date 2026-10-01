from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline


def build_model(preprocessor, model_params):
    model = CatBoostRegressor(**model_params)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])

    return pipeline