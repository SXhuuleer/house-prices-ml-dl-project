import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src.data import load_data
from src.features import (
    add_basic_features,
    add_interaction_features,
)
from src.preprocessing import build_preprocessor
from src.evaluation import evaluate_model_cv


class TestHousePricesPipeline(unittest.TestCase):

    def test_load_data(self):
        """Проверяем загрузку данных и исключение Id."""

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            train = pd.DataFrame({
                "Id": [1, 2],
                "Area": [100, 200],
                "SalePrice": [150000, 250000],
            })

            test = pd.DataFrame({
                "Id": [3],
                "Area": [180],
            })

            train_path = temp_dir / "train.csv"
            test_path = temp_dir / "test.csv"

            train.to_csv(train_path, index=False)
            test.to_csv(test_path, index=False)

            X, y, X_test, test_ids = load_data(
                train_path,
                test_path,
                target="SalePrice",
            )

            self.assertEqual(X.shape, (2, 1))
            self.assertEqual(X_test.shape, (1, 1))

            self.assertNotIn("Id", X.columns)
            self.assertNotIn("SalePrice", X.columns)

            self.assertEqual(y.tolist(), [150000, 250000])
            self.assertEqual(test_ids.tolist(), [3])

    def test_feature_engineering(self):
        """Проверяем правильность новых признаков."""

        X = pd.DataFrame({
            "TotalBsmtSF": [500],
            "1stFlrSF": [1000],
            "2ndFlrSF": [200],
            "FullBath": [2],
            "HalfBath": [1],
            "BsmtFullBath": [1],
            "BsmtHalfBath": [0],
            "WoodDeckSF": [50],
            "OpenPorchSF": [20],
            "EnclosedPorch": [10],
            "3SsnPorch": [0],
            "ScreenPorch": [30],
            "YrSold": [2010],
            "YearBuilt": [2000],
            "YearRemodAdd": [2005],
            "OverallQual": [7],
            "OverallCond": [5],
            "GrLivArea": [1200],
            "GarageCars": [2],
            "GarageArea": [400],
        })

        basic = add_basic_features(X)
        interactions = add_interaction_features(X)

        # Проверяем базовые признаки
        self.assertEqual(basic["TotalSF"].iloc[0], 1700)
        self.assertEqual(basic["TotalBathrooms"].iloc[0], 3.5)
        self.assertEqual(basic["TotalPorchSF"].iloc[0], 110)
        self.assertEqual(basic["HouseAge"].iloc[0], 10)
        self.assertEqual(basic["RemodAge"].iloc[0], 5)

        # Проверяем признаки взаимодействия
        self.assertEqual(
            interactions["QualGrLivArea"].iloc[0], 8400
        )

        self.assertEqual(
            interactions["QualBsmtSF"].iloc[0], 3500
        )

        self.assertEqual(
            interactions["GarageScore"].iloc[0], 800
        )

        self.assertEqual(
            interactions["OverallScore"].iloc[0], 35
        )

        # Исходные данные не должны изменяться
        self.assertEqual(X.shape[1], 20)

    def test_cross_validation(self):
        """Проверяем работу CV и расчёт метрик."""

        X = pd.DataFrame({
            "Area": np.arange(1000, 1180, 10),
            "Neighborhood": ["A", "B"] * 9,
        })

        y = pd.Series(
            np.linspace(100000, 200000, len(X))
        )

        model = Pipeline([
            ("preprocessor", build_preprocessor(X)),
            ("regressor", Ridge(alpha=1.0)),
        ])

        folds, summary = evaluate_model_cv(
            model=model,
            X=X,
            y=y,
            n_splits=3,
            random_state=42,
        )

        self.assertEqual(len(folds), 3)

        expected_metrics = [
            "CV RMSE",
            "CV Std",
            "CV MAPE",
            "CV WAPE",
        ]

        for metric in expected_metrics:
            self.assertIn(metric, summary)
            self.assertTrue(np.isfinite(summary[metric]))
            self.assertGreaterEqual(summary[metric], 0)


if __name__ == "__main__":
    unittest.main()