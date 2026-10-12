import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error
)

from src.preprocessing import build_preprocessor

class HousePriceMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),

            nn.Linear(64, 32),
            nn.GELU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

def to_float32_dense(X):
    """
    Преобразует данные в плотный массив float32.
    Нужен, поскольку OneHotEncoder может возвращать
    разреженную матрицу.
    """
    if hasattr(X, "toarray"):
        X = X.toarray()

    return np.ascontiguousarray(X, dtype=np.float32)


def train_mlp(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    learning_rate=0.003,
    random_state=42
):
    """
    Обучает MLP на подготовленных признаках.

    y_train должен содержать log1p(SalePrice).
    """

    torch.manual_seed(random_state)

    X_tensor = torch.from_numpy(
        to_float32_dense(X_train)
    )

    y_tensor = torch.from_numpy(
        np.asarray(
            y_train,
            dtype=np.float32
        ).reshape(-1, 1).copy()
    )

    dataset = TensorDataset(X_tensor, y_tensor)

    generator = torch.Generator()
    generator.manual_seed(random_state)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator
    )

    model = HousePriceMLP(
        input_dim=X_tensor.shape[1]
    )

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate
    )

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            predictions = model(X_batch)

            loss = criterion(
                predictions,
                y_batch
            )

            loss.backward()
            optimizer.step()

            total_loss += (
                loss.item() * len(X_batch)
            )

        epoch_loss = total_loss / len(dataset)

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train MSE: {epoch_loss:.5f}"
            )

    return model


def predict_mlp(model, X):
    """
    Возвращает предсказания в логарифмической шкале.
    """

    X_tensor = torch.from_numpy(
        to_float32_dense(X)
    )

    model.eval()

    with torch.no_grad():
        predictions = model(X_tensor)

    return predictions.squeeze(-1).cpu().numpy()


def evaluate_mlp_cv(
    X,
    y,
    n_splits=5,
    random_state=42,
    epochs=20,
    batch_size=32,
    learning_rate=0.003
):
    """
    Оценивает нейросеть через KFold.

    Preprocessing обучается заново на каждом
    тренировочном фолде.
    """

    cv = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X), start=1
    ):
        print(f"\n{'=' * 40}")
        print(f"DNN Fold {fold}/{n_splits}")
        print(f"{'=' * 40}")

        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        # Отдельный preprocessing для каждого фолда
        preprocessor = build_preprocessor(X_train)

        X_train_processed = preprocessor.fit_transform(
            X_train
        )

        X_val_processed = preprocessor.transform(
            X_val
        )

        # Обучаем нейросеть на log1p(SalePrice)
        model = train_mlp(
            X_train=X_train_processed,
            y_train=np.log1p(y_train),
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            random_state=random_state
        )

        # Предсказания на validation fold
        pred_log = predict_mlp(
            model,
            X_val_processed
        )

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
            np.abs(
                y_val.to_numpy() - pred_price
            ).sum()
            / np.abs(y_val.to_numpy()).sum()
        ) * 100

        fold_results.append({
            "Fold": fold,
            "RMSE": rmse,
            "MAPE": mape,
            "WAPE": wape
        })

        print(
            f"\nDNN Fold {fold} | "
            f"RMSE: {rmse:.5f} | "
            f"MAPE: {mape:.2f}% | "
            f"WAPE: {wape:.2f}%"
        )

    results_df = pd.DataFrame(fold_results)

    summary = {
        "CV RMSE": results_df["RMSE"].mean(),
        "CV Std": results_df["RMSE"].std(ddof=0),
        "CV MAPE": results_df["MAPE"].mean(),
        "CV WAPE": results_df["WAPE"].mean()
    }

    return results_df, summary
