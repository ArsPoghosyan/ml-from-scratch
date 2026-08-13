import numpy as np
import pytest

from ml_from_scratch.cluster import KMeans


def _make_two_cluster_data(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed=seed)

    cluster_a = rng.normal(loc=(-2.0, -2.0), scale=0.25, size=(40, 2))
    cluster_b = rng.normal(loc=(3.5, 3.0), scale=0.3, size=(40, 2))

    X = np.vstack((cluster_a, cluster_b))
    y = np.concatenate(
        (
            np.zeros(cluster_a.shape[0], dtype=int),
            np.ones(cluster_b.shape[0], dtype=int),
        )
    )
    return X, y


def test_k_means_fit_predicts_separated_clusters():
    X, y = _make_two_cluster_data()

    model = KMeans(
        n_clusters=2,
        init="k-means++",
        n_init=5,
        max_iter=100,
        random_state=0,
    )
    fitted = model.fit(X)
    labels = model.predict(X)

    assert fitted is model
    assert model.cluster_centers_.shape == (2, 2)
    assert model.labels_.shape == (X.shape[0],)
    assert labels.shape == (X.shape[0],)
    assert model.n_features_in_ == 2
    assert model.n_iter_ >= 1
    assert model.inertia_ >= 0.0

    np.testing.assert_array_equal(labels, model.labels_)
    assert np.unique(labels[: y.shape[0] // 2]).shape[0] == 1
    assert np.unique(labels[y.shape[0] // 2 :]).shape[0] == 1
    assert labels[0] != labels[-1]

    true_centers = np.vstack((X[y == 0].mean(axis=0), X[y == 1].mean(axis=0)))
    distances = np.linalg.norm(
        model.cluster_centers_[:, np.newaxis, :] - true_centers[np.newaxis, :, :],
        axis=2,
    )
    assert np.all(np.min(distances, axis=0) < 0.5)


def test_k_means_fit_predict_returns_labels():
    X, _ = _make_two_cluster_data()

    labels = KMeans(n_clusters=2, random_state=0).fit_predict(X)

    assert labels.shape == (X.shape[0],)


def test_k_means_predict_rejects_feature_mismatch():
    model = KMeans(n_clusters=2, random_state=0).fit([[0.0, 0.0], [1.0, 1.0]])

    with pytest.raises(ValueError, match="different number of features"):
        model.predict([[0.0, 1.0, 2.0]])


def test_k_means_predict_requires_fit():
    with pytest.raises(ValueError, match="not fitted yet"):
        KMeans().predict([[0.0, 0.0]])


def test_k_means_rejects_invalid_configuration():
    X, _ = _make_two_cluster_data()

    with pytest.raises(ValueError, match="n_clusters must be at least 1"):
        KMeans(n_clusters=0).fit(X)

    with pytest.raises(ValueError, match="init must be either 'random' or 'k-means\\+\\+'"):
        KMeans(n_clusters=2, init="bad").fit(X)

