def test_package_imports():
    import ml_from_scratch

    assert ml_from_scratch is not None


def test_top_level_exports_include_pca():
    from ml_from_scratch import PCA

    assert PCA is not None


def test_top_level_exports_include_all_models():
    from ml_from_scratch import (
        DecisionTreeClassifier,
        DecisionTreeRegressor,
        KMeans,
        LinearRegression,
        LogisticRegression,
        PCA,
        RandomForestClassifier,
        RandomForestRegressor,
    )

    assert LinearRegression is not None
    assert LogisticRegression is not None
    assert DecisionTreeClassifier is not None
    assert DecisionTreeRegressor is not None
    assert RandomForestClassifier is not None
    assert RandomForestRegressor is not None
    assert KMeans is not None
    assert PCA is not None
