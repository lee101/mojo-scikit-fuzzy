"""Common membership-function generators."""

from __future__ import annotations

import numpy as np

from ._lib import addr, f64, lib


def _result(x):
    values = f64(x)
    return values, np.empty_like(values)


def trimf(x, abc):
    values, result = _result(x)
    a, b, c = abc
    if not a <= b <= c:
        raise AssertionError("abc requires a <= b <= c")
    if values.size:
        lib().msf_trimf(addr(values), addr(result), values.size, a, b, c)
    return result


def trapmf(x, abcd):
    values, result = _result(x)
    a, b, c, d = abcd
    if not a <= b <= c <= d:
        raise AssertionError("abcd requires a <= b <= c <= d")
    if values.size:
        lib().msf_trapmf(addr(values), addr(result), values.size, a, b, c, d)
    return result


def gaussmf(x, mean, sigma):
    values, result = _result(x)
    if values.size:
        lib().msf_gaussmf(addr(values), addr(result), values.size, mean, sigma)
    return result


def gauss2mf(x, mean1, sigma1, mean2, sigma2):
    if mean1 > mean2:
        raise AssertionError("mean1 must be <= mean2")
    values = f64(x)
    result = np.ones_like(values)
    left = values <= mean1
    right = values > mean2
    result[left] = gaussmf(values[left], mean1, sigma1)
    result[right] = gaussmf(values[right], mean2, sigma2)
    return result


def gbellmf(x, a, b, c):
    values, result = _result(x)
    if values.size:
        lib().msf_gbellmf(addr(values), addr(result), values.size, a, b, c)
    return result


def sigmf(x, b, c):
    values, result = _result(x)
    if values.size:
        lib().msf_sigmf(addr(values), addr(result), values.size, b, c)
    return result


def dsigmf(x, b1, c1, b2, c2):
    return sigmf(x, b1, c1) - sigmf(x, b2, c2)


def psigmf(x, b1, c1, b2, c2):
    return sigmf(x, b1, c1) * sigmf(x, b2, c2)


def smf(x, a, b):
    if a > b:
        raise AssertionError("a must be <= b")
    values = f64(x)
    result = np.zeros_like(values)
    midpoint = (a + b) / 2
    middle_left = (values > a) & (values <= midpoint)
    middle_right = (values > midpoint) & (values < b)
    result[middle_left] = 2 * ((values[middle_left] - a) / (b - a)) ** 2
    result[middle_right] = 1 - 2 * ((values[middle_right] - b) / (b - a)) ** 2
    result[values >= b] = 1
    return result


def zmf(x, a, b):
    return 1.0 - smf(x, a, b)


def pimf(x, a, b, c, d):
    if not a <= b <= c <= d:
        raise AssertionError("a <= b <= c <= d is required")
    return smf(x, a, b) * zmf(x, c, d)
