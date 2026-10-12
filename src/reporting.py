from pathlib import Path

import pandas as pd


def create_all_models_report(
    ml_results,
    reports_dir,
    config
):
    """
    Объединяет результаты текущих ML-экспериментов
    и сохранённые результаты DNN и ансамблей.
    """

    reports_dir = Path(reports_dir)

    # ML-модели текущего запуска
    current_results = ml_results.copy()
    current_results["Source"] = "current_run"

    tables = [current_results]

    # Сохранённые результаты остальных моделей
    saved_experiments = {
        "dnn": (
            reports_dir / "folds" / "dnn_folds.csv",
            config.get("deep_learning", {}).get("enabled", False)
        ),
        "averaging": (
            reports_dir / "folds" / "averaging_folds.csv",
            config.get("ensembles", {}).get("enabled", False)
        ),
        "weighted_averaging": (
            reports_dir / "folds" / "weighted_averaging_folds.csv",
            config.get("ensembles", {}).get("enabled", False)
        ),
        "stacking": (
            reports_dir / "folds" / "stacking_folds.csv",
            config.get("stacking", {}).get("enabled", False)
        )
    }

    expected_folds = config["cv"]["n_splits"]

    for model_name, (file_path, enabled) in saved_experiments.items():

        if not file_path.exists():
            print(f"Report not found: {file_path}")
            continue

        folds = pd.read_csv(file_path)

        required_columns = {"Fold", "RMSE", "MAPE", "WAPE"}

        if not required_columns.issubset(folds.columns):
            print(f"Invalid report format: {file_path}")
            continue

        if sorted(folds["Fold"].tolist()) != list(
            range(1, expected_folds + 1)
        ):
            print(f"Incompatible CV fold count: {file_path}")
            continue

        summary = pd.DataFrame([{
            "Model": model_name,
            "CV RMSE": folds["RMSE"].mean(),
            "CV Std": folds["RMSE"].std(ddof=0),
            "CV MAPE": folds["MAPE"].mean(),
            "CV WAPE": folds["WAPE"].mean(),
            "Source": (
                "current_run"
                if enabled
                else "previous_run"
            )
        }])

        tables.append(summary)

    all_results = pd.concat(
        tables,
        ignore_index=True
    )

    all_results = (
        all_results
        .sort_values("CV RMSE")
        .reset_index(drop=True)
    )

    all_results.to_csv(
        reports_dir / "all_cv_results.csv",
        index=False
    )

    return all_results