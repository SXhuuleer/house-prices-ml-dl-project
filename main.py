import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import json

from src.tuning import (
    tune_model,
    evaluate_tuned_model_nested_cv
)
from src.preprocessing import build_preprocessor
from src.models import build_model
from src.data import load_data
from src.evaluation import evaluate_model_cv
from src.dl import evaluate_mlp_cv
from src.features import (
    add_basic_features,
    add_interaction_features
)
from src.ensembles import (
    build_voting_ensemble,
    build_stacking_ensemble
)
from src.reporting import create_all_models_report
from src.submission import save_submission

def main():
    with open("config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config["target"].get("log_transform") is not True:
        raise ValueError(
            "В текущем пайплайне log_transform должен быть true. "
            "Все ML, DL и CV эксперименты используют log1p(SalePrice)."
        )

    X_train, y_train, X_test, test_ids = load_data(
        train_path=config["paths"]["train"],
        test_path=config["paths"]["test"],
        target=config["target"]["name"]
    )

    y_train_original = y_train.copy()
    y_train = np.log1p(y_train)

    preprocessor = build_preprocessor(X_train)

    # Все модели для эксперимента
    model_configs = (
        config["experiments"]["models"]
        + [config["model"]]
    )

    cv_config = config["cv"]

    model_results = []

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    folds_dir = reports_dir / "folds"
    folds_dir.mkdir(exist_ok=True)


    for model_config in model_configs:
        model_name = model_config["name"]
        model_params = model_config.get("params", {})

        print(f"\n{'=' * 40}")
        print(f"Model: {model_name}")
        print(f"{'=' * 40}")

        model = build_model(
            model_name,
            preprocessor,
            model_params
        )

        fold_results, cv_summary = evaluate_model_cv(
            model=model,
            X=X_train,
            y=y_train_original,
            n_splits=cv_config["n_splits"],
            random_state=cv_config["random_state"]
        )

        fold_results.to_csv(
            folds_dir / f"{model_name}_folds.csv",
            index=False
        )

        model_results.append({
            "Model": model_name,
            **cv_summary
        })

        print(f"Mean RMSE: {cv_summary['CV RMSE']:.5f}")


    # Общая таблица результатов
    results = pd.DataFrame(model_results)

    results = (
        results
        .sort_values("CV RMSE")
        .reset_index(drop=True)
    )

    results.to_csv(
        reports_dir / "ml_cv_results.csv",
        index=False
    )

    print("\nML models comparison:")
    print(results)

    # Deep Learning: оценка нейросети
    dl_config = config.get("deep_learning", {})

    if dl_config.get("enabled", False):

        print("\n=== Deep Learning: 5-Fold CV ===")

        dl_folds, dl_summary = evaluate_mlp_cv(
            X=X_train,
            y=y_train_original,
            n_splits=cv_config["n_splits"],
            random_state=cv_config["random_state"],
            epochs=dl_config["epochs"],
            batch_size=dl_config["batch_size"],
            learning_rate=dl_config["learning_rate"]
        )

        # Сохраняем результаты каждого фолда
        dl_folds.to_csv(
            folds_dir / "dnn_folds.csv",
            index=False
        )

        # Добавляем DNN к результатам девяти ML-моделей
        all_results = pd.concat(
            [
                results,
                pd.DataFrame([{
                    "Model": "dnn",
                    **dl_summary
                }])
            ],
            ignore_index=True
        )

        all_results = (
            all_results
            .sort_values("CV RMSE")
            .reset_index(drop=True)
        )

        # Общая таблица ML + DL
        all_results.to_csv(
            reports_dir / "ml_dl_cv_results.csv",
            index=False
        )

        print("\n=== ML + DL Comparison ===")
        print(all_results.to_string(index=False))

    # Подбор гиперпараметров
    tuning_config = config.get("tuning", {})

    if tuning_config.get("enabled", False):
        tuning_results = []

        for tuning_model in tuning_config["models"]:
            model_name = tuning_model["name"]

            print(f"\nTuning model: {model_name}")

            tuning_result = tune_model(
                model_name=model_name,
                preprocessor=preprocessor,
                X=X_train,
                y=y_train_original,
                param_distributions=tuning_model["params"],
                n_iter=tuning_config["n_iter"],
                n_splits=tuning_config["n_splits"],
                random_state=cv_config["random_state"]
            )

            print("Best parameters:", tuning_result["best_params"])
            print("Best CV RMSE:", tuning_result["cv_rmse"])
            print("CV Std:", tuning_result["cv_std"])

            tuning_results.append({
                "Model": model_name,
                "Best Params": tuning_result["best_params"],
                "Search CV RMSE": float(tuning_result["cv_rmse"]),
                "Search CV Std": float(tuning_result["cv_std"])
            })

        with open(
            reports_dir / "tuning_results.json",
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                tuning_results,
                file,
                indent=4,
                ensure_ascii=False
            )

    # Вложенная кросс-валидация
    nested_config = config.get("nested_cv", {})

    if nested_config.get("enabled", False):

        # Пространства поиска из config.yaml
        tuning_spaces = {
            model["name"]: model["params"]
            for model in config["tuning"]["models"]
        }

        nested_results = []

        for model_name in nested_config["models"]:

            if model_name not in tuning_spaces:
                raise ValueError(
                    f"No tuning params for model: {model_name}"
                )

            print(f"\nNested CV: {model_name}")

            fold_results, nested_summary = (
                evaluate_tuned_model_nested_cv(
                    model_name=model_name,
                    preprocessor=preprocessor,
                    X=X_train,
                    y=y_train_original,
                    param_distributions=tuning_spaces[model_name],
                    n_iter=nested_config["n_iter"],
                    outer_splits=nested_config["outer_splits"],
                    inner_splits=nested_config["inner_splits"],
                    random_state=cv_config["random_state"]
                )
            )

            # Результаты по внешним фолдам
            fold_results.to_csv(
                folds_dir / f"{model_name}_nested_folds.csv",
                index=False
            )

            nested_results.append({
                "Model": model_name,
                **nested_summary
            })

            print("\nNested CV summary:")

            for metric, value in nested_summary.items():
                print(f"{metric}: {value:.5f}")

        # Итоговая таблица Nested CV
        pd.DataFrame(nested_results).to_csv(
            reports_dir / "nested_cv_results.csv",
            index=False
        )

    # Эксперимент с удалением выбросов
    outliers_config = config.get("outliers", {})

    if outliers_config.get("enabled", False):

        print("\n=== Outliers Experiment ===")

        # CatBoost с прежними параметрами
        outliers_model = build_model(
            config["model"]["name"],
            preprocessor,
            config["model"]["params"]
        )

        # Оцениваем модель с удалением выбросов
        filtered_folds, filtered_summary = evaluate_model_cv(
            model=outliers_model,
            X=X_train,
            y=y_train_original,
            n_splits=cv_config["n_splits"],
            random_state=cv_config["random_state"],
            remove_outliers=True,
            area_threshold=outliers_config["area_threshold"],
            price_threshold=outliers_config["price_threshold"]
        )

        # Baseline уже рассчитан в общем цикле ML-моделей
        baseline_row = results.loc[
            results["Model"] == config["model"]["name"]
        ].iloc[0]

        comparison = pd.DataFrame([
            {
                "Experiment": "Baseline",
                **{
                    metric: baseline_row[metric]
                    for metric in filtered_summary
                }
            },
            {
                "Experiment": "Without Outliers",
                **filtered_summary
            }
        ])

        print("\nOutliers Comparison:")
        print(comparison.to_string(index=False))

        rmse_difference = (
            filtered_summary["CV RMSE"]
            - baseline_row["CV RMSE"]
        )

        print(f"\nRMSE difference: {rmse_difference:+.5f}")

        # Сохраняем результаты
        comparison.to_csv(
            reports_dir / "outliers_comparison.csv",
            index=False
        )

        filtered_folds.to_csv(
            folds_dir / "catboost_outliers_folds.csv",
            index=False
        )

    # Эксперимент с Feature Engineering
    fe_config = config.get("feature_engineering", {})

    if fe_config.get("enabled", False):

        print("\n=== Feature Engineering Experiment ===")

        # Создаём пять новых признаков
        X_fe = add_basic_features(X_train)

        print("Original features:", X_train.shape[1])
        print("Features after FE:", X_fe.shape[1])

        # Отдельный preprocessing для расширенного датасета
        fe_preprocessor = build_preprocessor(X_fe)

        fe_model = build_model(
            config["model"]["name"],
            fe_preprocessor,
            config["model"]["params"]
        )

        # Оцениваем CatBoost с новыми признаками
        fe_folds, fe_summary = evaluate_model_cv(
            model=fe_model,
            X=X_fe,
            y=y_train_original,
            n_splits=cv_config["n_splits"],
            random_state=cv_config["random_state"]
        )

        # Результат исходного CatBoost уже есть в results
        baseline_row = results.loc[
            results["Model"] == config["model"]["name"]
        ].iloc[0]

        baseline_summary = {
            metric: float(baseline_row[metric])
            for metric in fe_summary
        }

        comparison = pd.DataFrame([
            {
                "Experiment": "Baseline",
                **baseline_summary
            },
            {
                "Experiment": "Feature Engineering",
                **fe_summary
            }
        ])

        print("\nFeature Engineering Comparison:")
        print(comparison.to_string(index=False))

        rmse_difference = (
            fe_summary["CV RMSE"]
            - baseline_summary["CV RMSE"]
        )

        print(f"\nRMSE difference: {rmse_difference:+.5f}")

        # Сохраняем результаты
        comparison.to_csv(
            reports_dir / "features_comparison.csv",
            index=False
        )

        fe_folds.to_csv(
            folds_dir / "catboost_fe_folds.csv",
            index=False
        )

    # Эксперимент с признаками взаимодействия
    interaction_config = config.get("interaction_features", {})

    if interaction_config.get("enabled", False):

        print("\n=== Interaction Features Experiment ===")

        # Добавляем четыре признака
        X_interactions = add_interaction_features(X_train)

        print("Original features:", X_train.shape[1])
        print("Features after interactions:", X_interactions.shape[1])

        # Создаём preprocessing для новых признаков
        interaction_preprocessor = build_preprocessor(
            X_interactions
        )

        interaction_model = build_model(
            config["model"]["name"],
            interaction_preprocessor,
            config["model"]["params"]
        )

        # Cross-validation
        interaction_folds, interaction_summary = evaluate_model_cv(
            model=interaction_model,
            X=X_interactions,
            y=y_train_original,
            n_splits=cv_config["n_splits"],
            random_state=cv_config["random_state"]
        )

        # Получаем уже рассчитанный CatBoost Baseline
        baseline_row = results.loc[
            results["Model"] == config["model"]["name"]
        ].iloc[0]

        baseline_summary = {
            metric: float(baseline_row[metric])
            for metric in interaction_summary
        }

        comparison = pd.DataFrame([
            {
                "Experiment": "Baseline",
                **baseline_summary
            },
            {
                "Experiment": "Interaction Features",
                **interaction_summary
            }
        ])

        print("\nInteraction Features Comparison:")
        print(comparison.to_string(index=False))

        rmse_difference = (
            interaction_summary["CV RMSE"]
            - baseline_summary["CV RMSE"]
        )

        print(f"\nRMSE difference: {rmse_difference:+.5f}")

        # Сохраняем результаты
        comparison.to_csv(
            reports_dir / "interactions_comparison.csv",
            index=False
        )

        interaction_folds.to_csv(
            folds_dir / "catboost_interactions_folds.csv",
            index=False
        )

        # Эксперименты с ансамблями
    ensembles_config = config.get("ensembles", {})

    if ensembles_config.get("enabled", False):

        print("\n=== Ensembles Experiments ===")

        # Конфигурации всех доступных ML-моделей
        configs_by_name = {
            model_config["name"]: model_config
            for model_config in model_configs
        }

        ensemble_model_configs = [
            configs_by_name[name]
            for name in ensembles_config["models"]
        ]

        ensemble_results = []

        for experiment in ensembles_config["experiments"]:
            experiment_name = experiment["name"]
            weights = experiment.get("weights")

            print(f"\n=== {experiment_name} ===")

            ensemble = build_voting_ensemble(
                X=X_train,
                model_configs=ensemble_model_configs,
                weights=weights
            )

            fold_results, ensemble_summary = evaluate_model_cv(
                model=ensemble,
                X=X_train,
                y=y_train_original,
                n_splits=cv_config["n_splits"],
                random_state=cv_config["random_state"]
            )

            fold_results.to_csv(
                folds_dir / f"{experiment_name}_folds.csv",
                index=False
            )

            ensemble_results.append({
                "Model": experiment_name,
                **ensemble_summary
            })

        # Таблица ансамблей
        ensembles_df = pd.DataFrame(ensemble_results)

        ensembles_df.to_csv(
            reports_dir / "ensembles_cv_results.csv",
            index=False
        )

        print("\n=== Ensembles Comparison ===")
        print(ensembles_df.to_string(index=False))

        # Объединяем ML, DL (если включён) и ансамбли
        base_results = (
            all_results
            if dl_config.get("enabled", False)
            else results
        )

        combined_results = pd.concat(
            [base_results, ensembles_df],
            ignore_index=True
        )

        combined_results = (
            combined_results
            .sort_values("CV RMSE")
            .reset_index(drop=True)
        )

        combined_results.to_csv(
            reports_dir / "all_cv_results.csv",
            index=False
        )

        print("\n=== All Models Comparison ===")
        print(combined_results.to_string(index=False))

        # Эксперимент со Stacking
    stacking_config = config.get("stacking", {})

    if stacking_config.get("enabled", False):

        print("\n=== Stacking Experiment ===")

        # Конфигурации доступных ML-моделей
        configs_by_name = {
            model_config["name"]: model_config
            for model_config in model_configs
        }

        # Используем базовые модели из раздела ensembles
        stacking_model_configs = [
            configs_by_name[name]
            for name in config["ensembles"]["models"]
        ]

        # Создаём StackingRegressor
        stacking_model = build_stacking_ensemble(
            X=X_train,
            model_configs=stacking_model_configs,
            ridge_alpha=stacking_config["ridge_alpha"],
            inner_splits=stacking_config["inner_splits"],
            random_state=cv_config["random_state"]
        )

        # Внешняя 5-fold CV
        stacking_folds, stacking_summary = evaluate_model_cv(
            model=stacking_model,
            X=X_train,
            y=y_train_original,
            n_splits=cv_config["n_splits"],
            random_state=cv_config["random_state"]
        )

        # Сохраняем результаты по фолдам
        stacking_folds.to_csv(
            folds_dir / "stacking_folds.csv",
            index=False
        )

        # Результат CatBoost из основного ML-цикла
        baseline_row = results.loc[
            results["Model"] == "catboost"
        ].iloc[0]

        comparison = pd.DataFrame([
            {
                "Model": "catboost",
                **{
                    metric: float(baseline_row[metric])
                    for metric in stacking_summary
                }
            },
            {
                "Model": "stacking",
                **stacking_summary
            }
        ])

        print("\n=== Stacking vs CatBoost ===")
        print(comparison.to_string(index=False))

        rmse_difference = (
            stacking_summary["CV RMSE"]
            - float(baseline_row["CV RMSE"])
        )

        print(f"\nRMSE difference: {rmse_difference:+.5f}")

        # Сохраняем сравнение
        comparison.to_csv(
            reports_dir / "stacking_comparison.csv",
            index=False
        )

        # Формируем общую таблицу текущего запуска
        if ensembles_config.get("enabled", False):
            base_results = combined_results
        elif dl_config.get("enabled", False):
            base_results = all_results
        else:
            base_results = results

        final_results = pd.concat(
            [
                base_results,
                pd.DataFrame([{
                    "Model": "stacking",
                    **stacking_summary
                }])
            ],
            ignore_index=True
        )

        final_results = (
            final_results
            .sort_values("CV RMSE")
            .reset_index(drop=True)
        )

        final_results.to_csv(
            reports_dir / "all_cv_results.csv",
            index=False
        )

        print("\n=== All Models Comparison ===")
        print(final_results.to_string(index=False))

    # Единая таблица всех проведённых экспериментов
    print("\n=== Final Experiments Summary ===")

    all_cv_results = create_all_models_report(
        ml_results=results,
        reports_dir=reports_dir,
        config=config
    )

    print(all_cv_results.to_string(index=False))

    # Финальная модель для Kaggle
    final_model = build_model(
        config["model"]["name"],
        preprocessor,
        config["model"]["params"]
    )

    # Обучаем финальную модель на всех тренировочных данных
    final_model.fit(X_train, y_train)

    # Предсказываем стоимость домов из test.csv
    predictions = final_model.predict(X_test)

    predictions = np.expm1(predictions)

    # Формируем submission
    submission = pd.DataFrame({
        "Id": test_ids,
        "SalePrice": predictions
    })

    submission_path = Path(config["paths"]["submission"])

    # Создаём папку автоматически, если её ещё нет
    submission_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    submission.to_csv(
        submission_path,
        index=False
    )

    print("\nSubmission saved successfully.")
    print(f"Submission shape: {submission.shape}")
    print(f"Saved to: {config['paths']['submission']}")

if __name__ == "__main__":
    main()