# House Prices ML/DL Project

Проект по задаче регрессии на датасете Kaggle House Prices.

Цель проекта — построить и сравнить несколько моделей машинного обучения и нейронную сеть для предсказания стоимости домов, провести EDA, preprocessing, feature engineering, кросс-валидацию и ансамблирование моделей.

## Dataset

Используется датасет соревнования Kaggle House Prices.

Целевая переменная:
- `SalePrice` — стоимость дома.

Для обучения моделей использовалось логарифмическое преобразование:

`log1p(SalePrice)`

Перед формированием submission выполнялось обратное преобразование:

`expm1(prediction)`

## EDA

В ноутбуке `notebooks/01_eda.ipynb` проведены:

- анализ размеров и структуры данных;
- анализ пропусков;
- исследование распределения `SalePrice`;
- сравнение исходного `SalePrice` и `log1p(SalePrice)`;
- анализ корреляций числовых признаков;
- анализ категориальных признаков;
- анализ выбросов;
- исследование зависимости цены от площади, качества дома, района и других признаков.

## Preprocessing

Для числовых признаков используются:

- заполнение пропусков медианой;
- StandardScaler.

Для категориальных признаков:

- заполнение пропусков наиболее частым значением;
- OneHotEncoder с `handle_unknown="ignore"`.

Для оценки качества используется 5-fold `KFold` с перемешиванием и `random_state=42`.

Основная метрика — RMSE на `log1p(SalePrice)`.

## Models

В проекте были протестированы:

- Linear Regression
- Ridge
- Lasso
- ElasticNet
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- DNN на PyTorch

Также были протестированы ансамбли:

- Averaging
- Weighted Averaging
- Stacking с Ridge в качестве метамодели

## Deep Learning

Была реализована MLP на PyTorch.

В ходе экспериментов проверялись:

- размеры скрытых слоёв;
- ReLU, LeakyReLU и GELU;
- Adam, AdamW и SGD;
- learning rate;
- batch size;
- количество эпох;
- Batch Normalization;
- Dropout.

Лучшей конфигурацией на holdout стала:

- hidden layers: `64 → 32`
- activation: `GELU`
- optimizer: `AdamW`
- learning rate: `0.003`
- batch size: `32`
- epochs: `20`
- без BatchNorm и Dropout

Для финальной оценки DNN была проведена 5-fold cross-validation.

## Feature Engineering

Были протестированы агрегированные признаки:

- `TotalSF`
- `TotalBathrooms`
- `TotalPorchSF`
- `HouseAge`
- `RemodAge`

Также были протестированы interaction-признаки:

- `QualGrLivArea`
- `QualBsmtSF`
- `GarageScore`
- `OverallScore`

Дополнительные признаки не улучшили результат CatBoost, поэтому финальная модель использует исходный набор признаков.

## Results

| Model | CV RMSE | CV Std |
|---|---:|---:|
| Stacking Ridge | 0.12267 | 0.02009 |
| Weighted Averaging | 0.12273 | 0.01981 |
| CatBoost | 0.12292 | 0.01728 |
| Averaging Ensemble | 0.12468 | 0.02476 |
| XGBoost | 0.12551 | 0.02196 |
| LightGBM | 0.12999 | 0.01899 |
| Random Forest | 0.14016 | 0.01675 |
| ElasticNet | 0.14285 | 0.04020 |
| Lasso | 0.14428 | 0.04247 |
| Ridge | 0.14678 | 0.03933 |
| Linear Regression | 0.15274 | 0.04702 |
| DNN | 0.15632 | 0.05704 |
| Decision Tree | 0.19017 | 0.01215 |

Полные результаты также сохранены в:

`reports/results.csv`

## Kaggle results

Для Kaggle были протестированы две финальные модели:

| Model | Kaggle Score |
|---|---:|
| CatBoost | **0.12515** |
| Stacking Ridge | 0.12571 |

Несмотря на немного лучший CV у Stacking Ridge, на Kaggle лучший результат показал CatBoost.

Поэтому финальная модель проекта — CatBoost.

## Project structure

```text
house-prices-ml-dl-project/
├── artifacts/
│   └── submissions/
│       ├── catboost_submission.csv
│       └── stacking_submission.csv
├── data/
│   └── raw/
├── notebooks/
│   └── 01_eda.ipynb
├── reports/
│   └── results.csv
├── src/
│   ├── models.py
│   └── preprocessing.py
├── config.yaml
├── main.py
├── README.md
└── requirements.txt