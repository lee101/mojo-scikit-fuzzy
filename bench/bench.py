"""Benchmarks against scikit-fuzzy on identical inputs."""

from __future__ import annotations

import math
import os
import platform
import subprocess
import sys
import time

import numpy as np
import skfuzzy
from skfuzzy import control as skcontrol

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
)

import mojofuzzy  # noqa: E402
from mojofuzzy import control  # noqa: E402


def timeit(function, repeat=3):
    function()
    start = time.perf_counter()
    function()
    estimate = time.perf_counter() - start
    number = max(1, min(100, math.ceil(0.2 / max(estimate, 1e-6))))
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        for _ in range(number):
            function()
        best = min(best, (time.perf_counter() - start) / number)
    return best


def duration(value):
    if value < 1e-3:
        return f"{value * 1e6:.1f} us"
    return f"{value * 1e3:.2f} ms"


def machine():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as source:
            for line in source:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def cmeans_case():
    rng = np.random.default_rng(1)
    data = np.ascontiguousarray(rng.normal(size=(8, 20_000)))
    init = rng.random((6, data.shape[1]))
    init /= init.sum(axis=0)
    return (
        lambda: mojofuzzy.cluster.cmeans(data, 6, 2, 0, 15, init=init),
        lambda: skfuzzy.cluster.cmeans(data, 6, 2, 0, 15, init=init),
    )


def predict_case():
    rng = np.random.default_rng(2)
    data = np.ascontiguousarray(rng.normal(size=(8, 80_000)))
    centers = np.ascontiguousarray(rng.normal(size=(8, 8)))
    init = rng.random((8, data.shape[1]))
    init /= init.sum(axis=0)
    return (
        lambda: mojofuzzy.cluster.cmeans_predict(data, centers, 2, 1e-9, 10, init=init),
        lambda: skfuzzy.cluster.cmeans_predict(data, centers, 2, 1e-9, 10, init=init),
    )


def centroid_case():
    x = np.linspace(-10, 10, 1_000_000)
    membership = skfuzzy.gaussmf(x, 0.7, 2.1)
    return (
        lambda: mojofuzzy.centroid(x, membership),
        lambda: skfuzzy.centroid(x, membership),
    )


def interpolation_case():
    x = np.linspace(-10, 10, 100_001)
    membership = skfuzzy.gaussmf(x, 0.3, 2.2)
    query = np.linspace(-12, 12, 1_000_000)
    return (
        lambda: mojofuzzy.interp_membership(x, membership, query),
        lambda: skfuzzy.interp_membership(x, membership, query),
    )


def gauss_case():
    x = np.linspace(-10, 10, 2_000_000)
    return (
        lambda: mojofuzzy.gaussmf(x, 0.3, 2.2),
        lambda: skfuzzy.gaussmf(x, 0.3, 2.2),
    )


def gauss_gpu_case():
    x = np.linspace(-10, 10, 2_000_000)
    return (
        lambda: mojofuzzy.gaussmf(x, 0.3, 2.2, device="gpu"),
        lambda: skfuzzy.gaussmf(x, 0.3, 2.2),
    )


def controller(fuzzy, ctrl):
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
        ctrl.Rule(service["average"], tip["medium"]),
        ctrl.Rule(service["good"] | quality["good"], tip["high"]),
    ]
    return ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules), cache=False)


def control_case():
    ours = controller(mojofuzzy, control)
    upstream = controller(skfuzzy, skcontrol)
    values = np.random.default_rng(3).uniform(0, 10, size=(250, 2))

    def run(simulation):
        for quality, service in values:
            simulation.input["quality"] = quality
            simulation.input["service"] = service
            simulation.compute()

    return lambda: run(ours), lambda: run(upstream)


CASES = [
    ("cmeans, 6 clusters (8 x 20k, 15 iter)", cmeans_case),
    ("cmeans_predict (8 x 80k)", predict_case),
    ("centroid (1M points)", centroid_case),
    ("interp_membership (1M queries)", interpolation_case),
    ("gaussmf (2M points)", gauss_case),
    ("gaussmf GPU (2M points)", gauss_gpu_case),
    ("Mamdani controller (250 evaluations)", control_case),
]


def gpu_memory_free_mib():
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=2,
        )
        return int(output.splitlines()[0].strip())
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def main():
    print(f"Machine: {machine()}")
    print()
    print("| operation | mojo-scikit-fuzzy | scikit-fuzzy | upstream / Mojo |")
    print("|---|---:|---:|---:|")
    for name, prepare in CASES:
        if "GPU" in name:
            free_mib = gpu_memory_free_mib()
            if free_mib is None or free_mib < 4000:
                detail = "unavailable" if free_mib is None else f"{free_mib} MiB free"
                print(f"| {name} | skipped ({detail}) | - | - |")
                continue
        ours, upstream = prepare()
        mojo_time = timeit(ours)
        upstream_time = timeit(upstream)
        print(
            f"| {name} | {duration(mojo_time)} | {duration(upstream_time)} "
            f"| {upstream_time / mojo_time:.2f}x |"
        )


if __name__ == "__main__":
    main()
