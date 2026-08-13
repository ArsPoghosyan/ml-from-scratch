import numpy as np
import pytest
from ml_from_scratch.decomposition import PCA


def _make_correlated_data() -> np.ndarray:
    return np.array(
        [
            [1.0, 1.0],
            [2.0, 2.1],
            [3.0, 2.9],
            [4.0, 4.2],
            [5.0, 5.1],
        ]
    )


def test_pca_fit_transform_reduces_dimension():
    X = _make_correlated_data()

    model = PCA(n_components=1)
    fitted = model.fit(X)
    transformed = model.transform(X)

    assert fitted is model
    assert transformed.shape == (X.shape[0], 1)
    assert model.components_.shape == (1, X.shape[1])
    assert model.n_features_in_ == X.shape[1]
    assert model.n_components_ == 1
    assert model.explained_variance_.shape == (1,)
    assert model.explained_variance_ratio_.shape == (1,)
    assert np.all(model.explained_variance_ratio_ >= 0.0)


def test_pca_inverse_transform_reconstructs_full_rank_data():
    X = _make_correlated_data()

    model = PCA(n_components=2).fit(X)
    transformed = model.transform(X)
    reconstructed = model.inverse_transform(transformed)

    np.testing.assert_allclose(reconstructed, X, atol=1e-10)


def test_pca_whiten_scales_variance():
    X = _make_correlated_data()

    transformed = PCA(n_components=1, whiten=True).fit_transform(X)

    assert transformed.shape == (X.shape[0], 1)
    assert np.isfinite(transformed).all()
    assert np.var(transformed, ddof=1) == pytest.approx(1.0, rel=1e-6, abs=1e-6)


def test_pca_predict_aliases_transform():
    X = _make_correlated_data()

    model = PCA(n_components=1).fit(X)

    np.testing.assert_allclose(model.predict(X), model.transform(X))


def test_pca_rejects_feature_mismatch_and_unfitted_use():
    model = PCA(n_components=1)

    with pytest.raises(ValueError, match="not fitted yet"):
        model.transform([[1.0, 2.0]])

    model.fit(_make_correlated_data())

    with pytest.raises(ValueError, match="different number of features"):
        model.transform([[1.0, 2.0, 3.0]])


def test_pca_rejects_invalid_n_components():
    X = _make_correlated_data()

    with pytest.raises(ValueError, match="at least 1"):
        PCA(n_components=0).fit(X)

    with pytest.raises(ValueError, match="cannot be greater"):
        PCA(n_components=3).fit(X)
