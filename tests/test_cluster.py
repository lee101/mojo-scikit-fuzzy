import numpy as np
import pytest
import skfuzzy

import mojofuzzy


def dataset(seed=4, features=3, samples=180):
    rng = np.random.default_rng(seed)
    data = np.ascontiguousarray(rng.normal(size=(features, samples)))
    init = rng.random((4, samples))
    init /= init.sum(axis=0)
    return data, init


def assert_result_parity(actual, expected, atol=5e-12, rtol=2e-12):
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want, rtol=rtol, atol=atol)


def test_cmeans_fixed_initialization_parity():
    data, init = dataset()
    expected = skfuzzy.cluster.cmeans(data, 4, 2.0, 1e-9, 100, init=init)
    actual = mojofuzzy.cluster.cmeans(data, 4, 2.0, 1e-9, 100, init=init)
    assert_result_parity(actual, expected)


@pytest.mark.parametrize("m", [1.5, 2.0, 3.0])
def test_cmeans_exponents(m):
    data, init = dataset(samples=90)
    expected = skfuzzy.cluster.cmeans(data, 4, m, 1e-8, 60, init=init)
    actual = mojofuzzy.cluster.cmeans(data, 4, m, 1e-8, 60, init=init)
    assert_result_parity(actual, expected, atol=3e-10, rtol=2e-9)
    np.testing.assert_allclose(actual[1].sum(axis=0), 1.0, atol=2e-15)


def test_cmeans_seeded_initialization_parity():
    data, _ = dataset(samples=75)
    expected = skfuzzy.cluster.cmeans(data, 3, 2, 1e-8, 70, seed=12)
    actual = mojofuzzy.cluster.cmeans(data, 3, 2, 1e-8, 70, seed=12)
    assert_result_parity(actual, expected)


def test_cmeans_predict_parity():
    data, init = dataset(samples=120)
    centers = skfuzzy.cluster.cmeans(data, 4, 2, 1e-9, 100, init=init)[0]
    test_data, prediction_init = dataset(seed=8, samples=65)
    expected = skfuzzy.cluster.cmeans_predict(
        test_data, centers, 2, 1e-9, 10, init=prediction_init
    )
    actual = mojofuzzy.cluster.cmeans_predict(
        test_data, centers, 2, 1e-9, 10, init=prediction_init
    )
    assert_result_parity(actual, expected)
    assert actual[3].shape == expected[3].shape == (2,)


def test_cmeans_single_cluster():
    data, _ = dataset(samples=30)
    init = np.ones((1, 30))
    expected = skfuzzy.cluster.cmeans(data, 1, 2, 1e-9, 10, init=init)
    actual = mojofuzzy.cluster.cmeans(data, 1, 2, 1e-9, 10, init=init)
    assert_result_parity(actual, expected)


def test_cmeans_simd_tail_parity():
    data, init = dataset(samples=67)
    expected = skfuzzy.cluster.cmeans(data, 4, 2, 1e-9, 20, init=init)
    actual = mojofuzzy.cluster.cmeans(data, 4, 2, 1e-9, 20, init=init)
    assert_result_parity(actual, expected)


@pytest.mark.parametrize("samples", [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17])
def test_cmeans_all_simd_tail_boundaries(samples):
    rng = np.random.default_rng(samples)
    data = rng.normal(size=(2, samples))
    init = rng.random((1, samples))
    expected = skfuzzy.cluster.cmeans(data, 1, 2, 0, 2, init=init)
    actual = mojofuzzy.cluster.cmeans(data[:, ::-1], 1, 2, 0, 2, init=init[:, ::-1])
    reversed_expected = skfuzzy.cluster.cmeans(
        data[:, ::-1], 1, 2, 0, 2, init=init[:, ::-1]
    )
    assert_result_parity(actual, reversed_expected)


def test_cmeans_parallel_threshold_parity():
    data, init = dataset(features=2, samples=16_385)
    expected = skfuzzy.cluster.cmeans(data, 4, 2, 0, 1, init=init)
    actual = mojofuzzy.cluster.cmeans(data, 4, 2, 0, 1, init=init)
    assert_result_parity(actual, expected)


@pytest.mark.parametrize(
    "kwargs,exception",
    [
        ({"m": 1}, ValueError),
        ({"metric": "cityblock"}, NotImplementedError),
        ({"c": 0}, ValueError),
        ({"c": 2.5}, ValueError),
        ({"error": -1}, ValueError),
        ({"maxiter": 0}, ValueError),
    ],
)
def test_cmeans_validation(kwargs, exception):
    data, init = dataset(samples=20)
    arguments = dict(c=4, m=2, error=1e-5, maxiter=10, init=init)
    arguments.update(kwargs)
    with pytest.raises(exception):
        mojofuzzy.cluster.cmeans(data, **arguments)


def test_cmeans_rejects_unsafe_empty_and_invalid_initialization():
    with pytest.raises(ValueError):
        mojofuzzy.cluster.cmeans(np.empty((2, 0)), 1, 2, 1e-5, 10)
    data, init = dataset(samples=20)
    init[:, 0] = 0
    with pytest.raises(ValueError):
        mojofuzzy.cluster.cmeans(data, 4, 2, 1e-5, 10, init=init)


def test_predict_allows_more_clusters_than_samples():
    rng = np.random.default_rng(22)
    data = rng.normal(size=(2, 3))
    centers = rng.normal(size=(5, 2))
    init = rng.random((5, 3))
    init /= init.sum(axis=0)
    expected = skfuzzy.cluster.cmeans_predict(data, centers, 2, 0, 2, init=init)
    actual = mojofuzzy.cluster.cmeans_predict(data, centers, 2, 0, 2, init=init)
    assert_result_parity(actual, expected)
