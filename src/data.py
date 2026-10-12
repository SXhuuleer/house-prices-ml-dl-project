import pandas as pd

def load_data(train_path, test_path, target="SalePrice"):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    y = train_df[target].copy()

    # Убираем целевую переменную и идентификатор
    X = train_df.drop(columns=[target, "Id"])

    # Запоминаем Id для Kaggle submission
    test_ids = test_df["Id"].copy()

    # Берём в test те же признаки, что и в train
    X_test = test_df[X.columns].copy()

    return X, y, X_test, test_ids