def remove_training_outliers(
    X_train,
    y_train,
    area_threshold=4000,
    price_threshold=300000
):
    """
    Удаляет подозрительные объекты только
    из тренировочной выборки.

    X_train: исходные признаки
    y_train: исходные цены (без log1p)
    """

    outlier_mask = (
        (X_train["GrLivArea"] > area_threshold)
        & (y_train < price_threshold)
    )

    removed_count = int(outlier_mask.sum())

    X_filtered = X_train.loc[~outlier_mask].copy()
    y_filtered = y_train.loc[~outlier_mask].copy()

    return X_filtered, y_filtered, removed_count