import numpy as np
from sklearn.model_selection import KFold, RandomizedSearchCV
from src.models import build_model
import pandas as pd

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error
)

def tune_model(
    model_name,
    preprocessor,
    X,
    y,
    param_distributions,
    n_iter=20,
    n_splits=5,
    random_state=42
):
    model = build_model(
        model_name=model_name,
        preprocessor=preprocessor
    )

    # Параметры модели внутри sklearn Pipeline
    search_params = {
        f"regressor__{name}": values
        for name, values in param_distributions.items()
    }

    cv = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=search_params,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        random_state=random_state,
        n_jobs=-1,
        error_score="raise"
    )

    y_log = np.log1p(y)

    search.fit(X, y_log)

    best_params = {
        name.replace("regressor__", ""): value
        for name, value in search.best_params_.items()
    }

    best_index = search.best_index_

    return {
        "model": model_name,
        "best_params": best_params,
        "cv_rmse": -search.best_score_,
        "cv_std": search.cv_results_["std_test_score"][best_index],
        "best_estimator": search.best_estimator_
    }

def evaluate_tuned_model_nested_cv(
    model_name,
    preprocessor,
    X,
    y,
    param_distributions,
    n_iter=5,
    outer_splits=5,
    inner_splits=3,
    random_state=42
):
    outer_cv = KFold(
        n_splits=outer_splits,
        shuffle=True,
        random_state=random_state
    )

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(
        outer_cv.split(X), start=1
    ):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        # Подбираем параметры только на Outer Train
        tuning_result = tune_model(
            model_name=model_name,
            preprocessor=preprocessor,
            X=X_train,
            y=y_train,
            param_distributions=param_distributions,
            n_iter=n_iter,
            n_splits=inner_splits,
            random_state=random_state
        )

        # Уже обученная лучшая модель
        best_model = tuning_result["best_estimator"]

        # Предсказываем на независимом Outer Validation
        pred_log = best_model.predict(X_val)

        pred_price = np.expm1(pred_log)

        rmse = np.sqrt(
            mean_squared_error(
                np.log1p(y_val),
                pred_log
            )
        )

        mape = mean_absolute_percentage_error(
            y_val, pred_price
        ) * 100

        wape = (
            np.abs(y_val.to_numpy() - pred_price).sum()
            / np.abs(y_val.to_numpy()).sum()
        ) * 100

        fold_results.append({
            "Fold": fold,
            "RMSE": rmse,
            "MAPE": mape,
            "WAPE": wape
        })

        print(
            f"Outer fold {fold} | "
            f"RMSE: {rmse:.5f} | "
            f"MAPE: {mape:.2f}% | "
            f"WAPE: {wape:.2f}%"
        )

        print("Best params:", tuning_result["best_params"])

    results_df = pd.DataFrame(fold_results)

    summary = {
        "CV RMSE": results_df["RMSE"].mean(),
        "CV Std": results_df["RMSE"].std(ddof=0),
        "CV MAPE": results_df["MAPE"].mean(),
        "CV WAPE": results_df["WAPE"].mean()
    }

    return results_df, summary