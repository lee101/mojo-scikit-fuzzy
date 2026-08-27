from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import skfuzzy
from skfuzzy import control as skcontrol

import mojofuzzy
from mojofuzzy import control


X = np.linspace(-4, 4, 257)


@pytest.mark.parametrize(
    "name,args",
    [
        ("trimf", ([-3, 0.2, 3],)),
        ("trapmf", ([-3, -2, 1, 3],)),
        ("gaussmf", (0.3, 1.2)),
        ("gauss2mf", (-1, 0.5, 1, 0.8)),
        ("gbellmf", (1.5, 2.2, 0.4)),
        ("sigmf", (0.2, -3.0)),
        ("dsigmf", (-1, 3, 1, 4)),
        ("psigmf", (-1, 3, 1, -4)),
        ("smf", (-2, 2)),
        ("zmf", (-2, 2)),
        ("pimf", (-3, -1, 1, 3)),
    ],
)
def test_membership_function_parity(name, args):
    expected = getattr(skfuzzy, name)(X, *args)
    actual = getattr(mojofuzzy, name)(X, *args)
    np.testing.assert_allclose(actual, expected, rtol=5e-9, atol=1e-10)


@pytest.mark.parametrize("mode", ["centroid", "bisector", "mom", "som", "lom"])
def test_defuzz_parity(mode):
    membership = np.maximum(
        np.minimum(0.71, skfuzzy.trimf(X, [-3, -0.4, 2.3])),
        np.minimum(0.28, skfuzzy.gaussmf(X, 1.7, 0.5)),
    )
    actual = mojofuzzy.defuzz(X, membership, mode)
    expected = skfuzzy.defuzz(X, membership, mode)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_dcentroid_parity():
    membership = skfuzzy.gaussmf(X, 0.2, 1.1)
    np.testing.assert_allclose(
        mojofuzzy.dcentroid(X, membership, 1.75),
        skfuzzy.dcentroid(X, membership, 1.75),
        atol=1e-13,
    )


def test_interp_membership_vector_and_scalar():
    membership = skfuzzy.trimf(X, [-3, 0, 2])
    query = np.linspace(-5, 5, 301)
    np.testing.assert_allclose(
        mojofuzzy.interp_membership(X, membership, query),
        skfuzzy.interp_membership(X, membership, query),
        atol=1e-15,
    )
    assert mojofuzzy.interp_membership(X, membership, 0.125) == pytest.approx(
        skfuzzy.interp_membership(X, membership, 0.125)
    )


def test_interp_membership_endpoint_extrapolation():
    membership = skfuzzy.sigmf(X, 0, 2)
    query = np.array([-8.0, 8.0])
    np.testing.assert_allclose(
        mojofuzzy.interp_membership(X, membership, query, zero_outside_x=False),
        skfuzzy.interp_membership(X, membership, query, zero_outside_x=False),
    )


def test_interp_membership_empty_query_and_invalid_universe():
    actual = mojofuzzy.interp_membership(X, np.ones_like(X), np.array([]))
    assert actual.shape == (0,)
    with pytest.raises(ValueError):
        mojofuzzy.interp_membership([0], [1], 0)
    with pytest.raises(ValueError):
        mojofuzzy.interp_membership([0, 0, 1], [0, 1, 0], 0.5)


def test_membership_functions_accept_empty_and_strided_inputs():
    assert mojofuzzy.gaussmf(np.array([]), 0, 1).shape == (0,)
    np.testing.assert_allclose(
        mojofuzzy.trimf(X[::3], [-3, 0.2, 3]),
        skfuzzy.trimf(X[::3], [-3, 0.2, 3]),
    )
    with pytest.raises(TypeError):
        mojofuzzy.gaussmf(np.array([1 + 2j]), 0, 1)


def test_gaussmf_simd_tail_and_parallel_threshold():
    x = np.linspace(-6, 6, 65_539)
    np.testing.assert_allclose(
        mojofuzzy.gaussmf(x, 0.3, 1.7),
        skfuzzy.gaussmf(x, 0.3, 1.7),
        rtol=5e-9,
        atol=1e-10,
    )


def test_gaussmf_gpu_or_cpu_fallback():
    x = np.linspace(-6, 6, 259)
    np.testing.assert_allclose(
        mojofuzzy.gaussmf(x, 0.3, 1.7, device="gpu"),
        skfuzzy.gaussmf(x, 0.3, 1.7),
        rtol=5e-9,
        atol=1e-10,
    )
    with pytest.raises(ValueError):
        mojofuzzy.gaussmf([], 0, 1, device="accelerator")


@pytest.mark.parametrize("ordered", [True, False])
def test_interp_membership_parallel_threshold(ordered):
    x = np.linspace(-4, 4, 1003)
    membership = skfuzzy.gaussmf(x, 0.2, 1.1)
    query = np.linspace(-5, 5, 65_539)
    if not ordered:
        query = np.random.default_rng(14).permutation(query)
    np.testing.assert_allclose(
        mojofuzzy.interp_membership(x, membership, query),
        skfuzzy.interp_membership(x, membership, query),
        atol=1e-15,
    )


def test_parallel_runtime_initializes_on_calling_thread():
    x = np.linspace(-4, 4, 1003)
    membership = skfuzzy.gaussmf(x, 0.2, 1.1)
    query = np.linspace(-5, 5, 65_539)
    with ThreadPoolExecutor(max_workers=1) as executor:
        actual = executor.submit(
            mojofuzzy.interp_membership, x, membership, query
        ).result()
    np.testing.assert_allclose(
        actual,
        skfuzzy.interp_membership(x, membership, query),
        atol=1e-15,
    )


def test_interp_universe_parity():
    membership = skfuzzy.trimf(X, [-3, 0.1, 3])
    for level in [0.0, 0.2, 0.7, 1.0]:
        assert sorted(mojofuzzy.interp_universe(X, membership, level)) == pytest.approx(
            sorted(skfuzzy.interp_universe(X, membership, level))
        )


def test_lambda_cut_helpers_parity():
    membership = skfuzzy.trimf(X, [-3, 0.1, 3])
    for level in [0, 0.4, 1]:
        np.testing.assert_array_equal(
            mojofuzzy.lambda_cut(membership, level),
            skfuzzy.lambda_cut(membership, level),
        )
    np.testing.assert_allclose(
        mojofuzzy.lambda_cut_series(X, membership, 7),
        skfuzzy.lambda_cut_series(X, membership, 7),
    )
    for level in [0, 0.4, 1]:
        actual = mojofuzzy.arglcut(membership, level)
        expected = skfuzzy.arglcut(membership, level)
        np.testing.assert_array_equal(actual[0], expected[0])


def build_tip(fuzzy, ctrl):
    quality = ctrl.Antecedent(np.arange(0, 11), "quality")
    service = ctrl.Antecedent(np.arange(0, 11), "service")
    tip = ctrl.Consequent(np.arange(0, 26), "tip")
    quality.automf(3)
    service.automf(3)
    tip["low"] = fuzzy.trimf(tip.universe, [0, 0, 13])
    tip["medium"] = fuzzy.trimf(tip.universe, [0, 13, 25])
    tip["high"] = fuzzy.trimf(tip.universe, [13, 25, 25])
    rules = [
        ctrl.Rule(quality["poor"] | service["poor"], tip["low"]),
        ctrl.Rule(service["average"], tip["medium"] % 0.85),
        ctrl.Rule(service["good"] | quality["good"], tip["high"]),
    ]
    return ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))


@pytest.mark.parametrize("quality,service", [(6.5, 9.8), (1, 1), (5, 5), (9, 3)])
def test_control_system_parity(quality, service):
    expected = build_tip(skfuzzy, skcontrol)
    actual = build_tip(mojofuzzy, control)
    for simulation in (expected, actual):
        simulation.input["quality"] = quality
        simulation.input["service"] = service
        simulation.compute()
    assert actual.output["tip"] == pytest.approx(expected.output["tip"], abs=2e-12)


def test_control_clips_inputs_and_upsamples_cut_intersections():
    expected = build_tip(skfuzzy, skcontrol)
    actual = build_tip(mojofuzzy, control)
    for simulation in (expected, actual):
        simulation.input["quality"] = 50
        simulation.input["service"] = 2.3
        simulation.compute()
    assert actual.input["quality"] == 10
    assert actual.output["tip"] == pytest.approx(expected.output["tip"], abs=2e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"number": 3, "variable_type": "quant"},
        {"number": 5, "invert": True},
        {"names": ["cold", "mild", "hot"]},
    ],
)
def test_automf_options_match_upstream(kwargs):
    expected = skcontrol.Antecedent(np.arange(0, 11), "value")
    actual = control.Antecedent(np.arange(0, 11), "value")
    expected.automf(**kwargs)
    actual.automf(**kwargs)
    assert list(actual.terms) == list(expected.terms)
    for label in expected.terms:
        np.testing.assert_allclose(actual[label].mf, expected[label].mf)


def test_control_not_and_multiple_consequents():
    x = np.linspace(0, 10, 51)

    def make(fuzzy, ctrl):
        antecedent = ctrl.Antecedent(x, "x")
        left = ctrl.Consequent(x, "left")
        right = ctrl.Consequent(x, "right", defuzzify_method="bisector")
        antecedent["low"] = fuzzy.trimf(x, [0, 0, 10])
        antecedent["high"] = fuzzy.trimf(x, [0, 10, 10])
        left["low"] = fuzzy.trimf(x, [0, 0, 8])
        right["high"] = fuzzy.trimf(x, [2, 10, 10])
        rule = ctrl.Rule(~antecedent["low"] & antecedent["high"], (left["low"], right["high"]))
        simulation = ctrl.ControlSystemSimulation(ctrl.ControlSystem([rule]))
        simulation.input["x"] = 8
        simulation.compute()
        return simulation

    expected = make(skfuzzy, skcontrol)
    actual = make(mojofuzzy, control)
    assert actual.output == pytest.approx(expected.output, abs=2e-12)


def test_empty_defuzz_raises():
    with pytest.raises(mojofuzzy.EmptyMembershipError):
        mojofuzzy.defuzz(X, np.zeros_like(X), "centroid")
