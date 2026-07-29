# mojo-scikit-fuzzy

`mojo-scikit-fuzzy` is a standalone Mojo port of the compute-heavy fuzzy
c-means and Mamdani inference paths in
[scikit-fuzzy](https://github.com/scikit-fuzzy/scikit-fuzzy). The public Python
package is `mojofuzzy`; covered functions keep scikit-fuzzy's names, argument
order, array orientation, and return layout.

This is a focused port, not a reimplementation of every scikit-fuzzy module.
It is useful today for clustering, membership generation, defuzzification,
and conventional scalar fuzzy controllers.

## Covered API

- `cluster.cmeans` and `cluster.cmeans_predict`, including seeded or supplied
  membership initialization, convergence history, distances, and FPC.
- `trimf`, `trapmf`, `gaussmf`, `gauss2mf`, `gbellmf`, `sigmf`, `dsigmf`,
  `psigmf`, `smf`, `zmf`, and `pimf`.
- `centroid`, `dcentroid`, `bisector`, and `defuzz` modes `centroid`,
  `bisector`, `mom`, `som`, and `lom`.
- `interp_membership`, `interp_universe`, `lambda_cut`, `arglcut`, and
  `lambda_cut_series`.
- Mamdani `Antecedent`, `Consequent`, `Rule`, `ControlSystem`, and
  `ControlSystemSimulation`, with `automf`, `&`, `|`, `~`, weighted and
  multiple consequents, min implication, max accumulation, clipping, and
  exact universe upsampling at cut intersections.

The clustering port currently covers Euclidean distance only. The control
port covers scalar inputs and the usual min/max Mamdani path; array-valued
simulation, network graph inspection, caching semantics, custom implication
or accumulation functions, visualization, and the remainder of
scikit-fuzzy's image, filters, relations, and miscellaneous fuzzy arithmetic
are not covered.

All compiled kernels operate on `float64`. Python inputs of other real NumPy
dtypes are copied into C-contiguous `float64` buffers before the FFI call;
complex-valued fuzzy sets are not supported. This release targets Linux
x86-64 and builds the shared library locally rather than distributing a
prebuilt wheel.

## Install

The repository pins the tested Mojo nightly and installs NumPy, pytest, and
the real scikit-fuzzy package used by parity tests:

```bash
pixi install
pixi run build
```

The shared library is written to `dist/libmojo-scikit-fuzzy.so`.

## Usage

This complete controller example uses the same construction pattern as
`skfuzzy.control`:

```python
import numpy as np
import mojofuzzy as fuzz
from mojofuzzy import control as ctrl

quality = ctrl.Antecedent(np.arange(0, 11), "quality")
service = ctrl.Antecedent(np.arange(0, 11), "service")
tip = ctrl.Consequent(np.arange(0, 26), "tip")

quality.automf(3)
service.automf(3)
tip["low"] = fuzz.trimf(tip.universe, [0, 0, 13])
tip["medium"] = fuzz.trimf(tip.universe, [0, 13, 25])
tip["high"] = fuzz.trimf(tip.universe, [13, 25, 25])

rules = [
    ctrl.Rule(quality["poor"] | service["poor"], tip["low"]),
    ctrl.Rule(service["average"], tip["medium"]),
    ctrl.Rule(service["good"] | quality["good"], tip["high"]),
]

simulation = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))
simulation.input["quality"] = 6.5
simulation.input["service"] = 9.8
simulation.compute()
print(simulation.output["tip"])  # 19.847607361963192
```

Fuzzy c-means preserves scikit-fuzzy's features-by-samples orientation:

```python
data = np.random.default_rng(0).normal(size=(4, 10_000))
centers, u, u0, distances, objective, iterations, fpc = (
    fuzz.cluster.cmeans(data, c=5, m=2, error=1e-6, maxiter=100, seed=0)
)
```

## Benchmarks

Measured on 2026-07-29 on an Intel Xeon E5-2697 v4 at 2.30 GHz with the
pinned Mojo `1.0.0b3.dev2026072406`, Python 3.13, and scikit-fuzzy 0.5.0.
Times are the best of three warmed batches. `upstream / Mojo` above 1 means
Mojo is faster.

| operation | mojo-scikit-fuzzy | scikit-fuzzy | upstream / Mojo |
|---|---:|---:|---:|
| cmeans, 6 clusters (8 x 20k, 15 iter) | 95.40 ms | 484.00 ms | 5.07x |
| cmeans_predict (8 x 80k) | 59.24 ms | 201.90 ms | 3.41x |
| centroid (1M points) | 7.17 ms | 2000.50 ms | 279.08x |
| interp_membership (1M queries) | 2.73 ms | 9.14 ms | 3.35x |
| gaussmf (2M points) | 24.98 ms | 50.66 ms | 2.03x |
| Mamdani controller (250 evaluations) | 151.62 ms | 553.11 ms | 3.65x |

The very large centroid result comes from replacing scikit-fuzzy's Python
loop over adjacent trapezoids with one compiled loop. Bulk interpolation
detects ordered queries and scans each independent chunk linearly; unordered
queries retain binary search. Reproduce the table with:

```bash
pixi run bench
```

## How it works

`src/capi.mojo` is one compilation unit exporting non-parametric C ABI
functions. Python owns every allocation. Contiguous `float64` NumPy buffers
cross `ctypes` as integer addresses, and Mojo reconstructs mutable
`UnsafePointer[Float64, AnyOrigin[mut=True]]` values inside each exported
function. The Python wrappers validate non-empty kernel inputs and keep every
NumPy owner alive until the synchronous call returns; no allocation crosses
the language boundary.

C-means data remains in scikit-fuzzy's row-major `(features, samples)`
layout, while centers are `(clusters, features)` and memberships and
distances are `(clusters, samples)`. Each iteration performs membership
normalization, weighted center reduction, distance calculation, the
numerically stabilized negative-power update, and objective accumulation in
Mojo. These passes use the host's compile-time `float64` SIMD width with
scalar remainder loops. Workloads above 65,536 element-work units are split
into independent CPU tasks; smaller inputs stay serial. Convergence norm and
state copying are fused into one SIMD pass, avoiding per-iteration NumPy
temporaries. The Python layer reproduces the upstream return tuple.

No GPU path is included. The covered clustering distance and interpolation
kernels perform well under two arithmetic operations per byte moved for the
benchmarked shapes, so transfer and launch costs would dominate rather than
accelerate them.

Inference interpolates crisp antecedents, evaluates the lightweight rule
tree in Python, and sends consequent clipping and max aggregation through
Mojo. Cut intersections are inserted into the consequent universe before
compiled centroid or bisector integration, matching scikit-fuzzy on coarse
universes as well as dense ones.

## Verification

```bash
pixi run build
pixi run test
```

The parity suite compares against the installed scikit-fuzzy 0.5.0
implementation, not a hand-written reference. It checks complete c-means and
prediction return tuples, membership functions, defuzzification,
interpolation, lambda cuts, and end-to-end control-system outputs.
