
def add_basic_features(X):
    """
    Создаёт дополнительные признаки для House Prices.

    Исходный DataFrame не изменяется.
    """

    X = X.copy()

    # Общая площадь дома
    X["TotalSF"] = (
        X["TotalBsmtSF"].fillna(0)
        + X["1stFlrSF"].fillna(0)
        + X["2ndFlrSF"].fillna(0)
    )

    # Общее количество ванных комнат
    X["TotalBathrooms"] = (
        X["FullBath"]
        + 0.5 * X["HalfBath"]
        + X["BsmtFullBath"].fillna(0)
        + 0.5 * X["BsmtHalfBath"].fillna(0)
    )

    # Общая площадь веранд, террас и крылец
    porch_columns = [
        "WoodDeckSF",
        "OpenPorchSF",
        "EnclosedPorch",
        "3SsnPorch",
        "ScreenPorch"
    ]

    X["TotalPorchSF"] = (
        X[porch_columns].fillna(0).sum(axis=1)
    )

    # Возраст дома на момент продажи
    X["HouseAge"] = (
        X["YrSold"] - X["YearBuilt"]
    )

    # Сколько лет прошло после реконструкции
    X["RemodAge"] = (
        X["YrSold"] - X["YearRemodAdd"]
    )

    return X

def add_interaction_features(X):
    """
    Создаёт четыре признака взаимодействия.
    Исходный DataFrame не изменяется.
    """

    X = X.copy()

    X["QualGrLivArea"] = (
        X["OverallQual"] * X["GrLivArea"]
    )

    X["QualBsmtSF"] = (
        X["OverallQual"] * X["TotalBsmtSF"].fillna(0)
    )

    X["GarageScore"] = (
        X["GarageCars"].fillna(0)
        * X["GarageArea"].fillna(0)
    )

    X["OverallScore"] = (
        X["OverallQual"] * X["OverallCond"]
    )

    return X