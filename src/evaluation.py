import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error
)

from src.outliers import remove_training_outliers


def evaluate_model_cv(
    model,
    X,
    y,
    n_splits=5,
    random_state=42,
    remove_outliers=False,
    area_threshold=4000,
    price_threshold=300000
):
    cv = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X), start=1
    ):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        removed_count = 0

        # Удаляем выбросы только из тренировочного фолда
        if remove_outliers:
            X_train, y_train, removed_count = (
                remove_training_outliers(
                    X_train=X_train,
                    y_train=y_train,
                    area_threshold=area_threshold,
                    price_threshold=price_threshold
                )
            )

        # Создаём отдельную модель для каждого фолда
        fold_model = clone(model)

        # Обучаем модель на логарифмах цен
        y_train_log = np.log1p(y_train)

        fold_model.fit(X_train, y_train_log)

        # Предсказания для нетронутого validation fold
        pred_log = fold_model.predict(X_val)

        # Возвращаем цены в исходную шкалу
        pred_price = np.expm1(pred_log)

        rmse = np.sqrt(
            mean_squared_error(
                np.log1p(y_val),
                pred_log
            )
        )

        mape = mean_absolute_percentage_error(
            y_val,
            pred_price
        ) * 100

        wape = (
            np.abs(y_val.to_numpy() - pred_price).sum()
            / np.abs(y_val.to_numpy()).sum()
        ) * 100

        fold_result = {
            "Fold": fold,
            "RMSE": rmse,
            "MAPE": mape,
            "WAPE": wape
        }

        if remove_outliers:
            fold_result["Removed Outliers"] = removed_count

        fold_results.append(fold_result)

        print(
            f"Fold {fold} | "
            f"RMSE: {rmse:.5f} | "
            f"MAPE: {mape:.2f}% | "
            f"WAPE: {wape:.2f}%"
            + (
                f" | Removed: {removed_count}"
                if remove_outliers else ""
            )
        )

    results_df = pd.DataFrame(fold_results)

    summary = {
        "CV RMSE": results_df["RMSE"].mean(),
        "CV Std": results_df["RMSE"].std(ddof=0),
        "CV MAPE": results_df["MAPE"].mean(),
        "CV WAPE": results_df["WAPE"].mean()
    }

    return results_df, summary