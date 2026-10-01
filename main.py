import numpy as np
import pandas as pd
import yaml

from src.preprocessing import build_preprocessor
from src.models import build_model


def main():
    with open("config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    train_df = pd.read_csv(config["paths"]["train"])
    test_df = pd.read_csv(config["paths"]["test"])

    target = config["target"]["name"]

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]

    X_test = test_df.copy()

    if config["target"]["log_transform"]:
        y_train = np.log1p(y_train)

    preprocessor = build_preprocessor(X_train)

    model_params = config["model"]["params"]

    final_model = build_model(
        preprocessor,
        model_params
    )

    final_model.fit(X_train, y_train)

    predictions = final_model.predict(X_test)

    if config["target"]["log_transform"]:
        predictions = np.expm1(predictions)

    submission = pd.DataFrame({
        "Id": test_df["Id"],
        "SalePrice": predictions
    })

    submission.to_csv(
        config["paths"]["submission"],
        index=False
    )

    print("Submission saved successfully.")
    print(f"Submission shape: {submission.shape}")
    print(f"Saved to: {config['paths']['submission']}")


if __name__ == "__main__":
    main()