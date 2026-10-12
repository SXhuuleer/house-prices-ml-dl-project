from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from src.models import build_model
from src.preprocessing import build_preprocessor


def build_voting_ensemble(
    X,
    model_configs,
    weights=None
):
    """
    Создаёт ансамбль из нескольких ML-моделей.

    weights=None — обычное усреднение.
    weights=[...] — взвешенное усреднение.
    """

    estimators = []

    for model_config in model_configs:
        model_name = model_config["name"]
        model_params = model_config.get("params", {})

        # Каждая модель получает собственный Pipeline
        preprocessor = build_preprocessor(X)

        model = build_model(
            model_name,
            preprocessor,
            model_params
        )

        estimators.append((model_name, model))

    if weights is not None:
        if len(weights) != len(estimators):
            raise ValueError(
                "Number of weights must match number of models"
            )

        if any(weight < 0 for weight in weights):
            raise ValueError(
                "Weights must be non-negative"
            )

        if sum(weights) <= 0:
            raise ValueError(
                "At least one weight must be positive"
            )

    ensemble = VotingRegressor(
        estimators=estimators,
        weights=weights,
        n_jobs=1
    )

    return ensemble

def build_stacking_ensemble(
    X,
    model_configs,
    ridge_alpha=0.1,
    inner_splits=5,
    random_state=42
):
    """
    Создаёт StackingRegressor.

    Каждая базовая модель имеет собственный Pipeline,
    поэтому preprocessing обучается внутри CV-фолдов.
    """

    estimators = []

    for model_config in model_configs:
        model_name = model_config["name"]
        model_params = model_config.get("params", {})

        preprocessor = build_preprocessor(X)

        model = build_model(
            model_name,
            preprocessor,
            model_params
        )

        estimators.append((model_name, model))

    inner_cv = KFold(
        n_splits=inner_splits,
        shuffle=True,
        random_state=random_state
    )

    stacking = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=ridge_alpha),
        cv=inner_cv,
        passthrough=False,
        n_jobs=1
    )

    return stacking