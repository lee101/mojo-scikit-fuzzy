"""Numerical kernels exposed to Python through a small C ABI."""

from std.algorithm import map
from std.math import exp, pow, sqrt
from std.sys.info import simd_width_of

comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime EPS = 2.220446049250313e-16
comptime PARALLEL_TASKS = 16
comptime PARALLEL_THRESHOLD = 65536


def p(addr: Int) -> Ptr:
    return Ptr(unsafe_from_address=addr)


@export("KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice")
def initialize_cpu_runtime() abi("C") -> Int:
    """Keep the legacy loader contract after AsyncRT left the Mojo stdlib."""
    return 1


def normalize_memberships_simd(
    u: Ptr,
    um: Ptr,
    clusters: Int,
    samples: Int,
    m: Float64,
    block_start: Int,
    block_end: Int,
):
    comptime W = simd_width_of[DType.float64]()
    for block in range(block_start, block_end):
        var j = block * W
        var total = SIMD[DType.float64, W](0.0)
        for k in range(clusters):
            total += u.load[width=W](k * samples + j)
        for k in range(clusters):
            var value = u.load[width=W](k * samples + j) / total
            value = max(value, SIMD[DType.float64, W](EPS))
            um.store(k * samples + j, pow(value, m))


def normalize_memberships_tail(
    u: Ptr, um: Ptr, clusters: Int, samples: Int, m: Float64, start: Int
):
    for j in range(start, samples):
        var total = 0.0
        for k in range(clusters):
            total += u[k * samples + j]
        for k in range(clusters):
            var value = u[k * samples + j] / total
            if value < EPS:
                value = EPS
            um[k * samples + j] = pow(value, m)


def normalize_memberships(u: Ptr, um: Ptr, clusters: Int, samples: Int, m: Float64):
    comptime W = simd_width_of[DType.float64]()
    var blocks = samples // W
    if clusters * samples >= PARALLEL_THRESHOLD:
        var u_addr = Int(u)
        var um_addr = Int(um)

        @__parameter
        def work(task: Int):
            var start = task * blocks // PARALLEL_TASKS
            var end = (task + 1) * blocks // PARALLEL_TASKS
            normalize_memberships_simd(
                p(u_addr), p(um_addr), clusters, samples, m, start, end
            )

        map[work](PARALLEL_TASKS)
    else:
        normalize_memberships_simd(u, um, clusters, samples, m, 0, blocks)
    normalize_memberships_tail(u, um, clusters, samples, m, blocks * W)


def compute_center(
    data: Ptr,
    centers: Ptr,
    um: Ptr,
    cluster: Int,
    features: Int,
    samples: Int,
):
    comptime W = simd_width_of[DType.float64]()
    var row = cluster * samples
    var vector_end = samples - samples % W
    var vector_weight = SIMD[DType.float64, W](0.0)
    for j in range(0, vector_end, W):
        vector_weight += um.load[width=W](row + j)
    var weight = vector_weight.reduce_add()
    for j in range(vector_end, samples):
        weight += um[row + j]
    for f in range(features):
        var vector_total = SIMD[DType.float64, W](0.0)
        for j in range(0, vector_end, W):
            vector_total += (
                um.load[width=W](row + j)
                * data.load[width=W](f * samples + j)
            )
        var total = vector_total.reduce_add()
        for j in range(vector_end, samples):
            total += um[row + j] * data[f * samples + j]
        centers[cluster * features + f] = total / weight


def compute_centers(
    data: Ptr,
    centers: Ptr,
    um: Ptr,
    clusters: Int,
    features: Int,
    samples: Int,
):
    if clusters * features * samples >= PARALLEL_THRESHOLD:
        var tasks = min(clusters, PARALLEL_TASKS)
        var data_addr = Int(data)
        var centers_addr = Int(centers)
        var um_addr = Int(um)

        @__parameter
        def work(task: Int):
            var start = task * clusters // tasks
            var end = (task + 1) * clusters // tasks
            for k in range(start, end):
                compute_center(
                    p(data_addr),
                    p(centers_addr),
                    p(um_addr),
                    k,
                    features,
                    samples,
                )

        map[work](tasks)
    else:
        for k in range(clusters):
            compute_center(data, centers, um, k, features, samples)


def compute_distances_for_cluster(
    data: Ptr,
    centers: Ptr,
    distances: Ptr,
    cluster: Int,
    features: Int,
    samples: Int,
):
    comptime W = simd_width_of[DType.float64]()
    var vector_end = samples - samples % W
    var row = cluster * samples
    for j in range(0, vector_end, W):
        var distance2 = SIMD[DType.float64, W](0.0)
        for f in range(features):
            var delta = (
                data.load[width=W](f * samples + j)
                - centers[cluster * features + f]
            )
            distance2 += delta * delta
        distances.store(
            row + j,
            max(
                sqrt(distance2),
                SIMD[DType.float64, W](EPS),
            ),
        )
    for j in range(vector_end, samples):
        var distance2 = 0.0
        for f in range(features):
            var delta = data[f * samples + j] - centers[cluster * features + f]
            distance2 += delta * delta
        distances[row + j] = max(sqrt(distance2), EPS)


def compute_distances(
    data: Ptr,
    centers: Ptr,
    distances: Ptr,
    clusters: Int,
    features: Int,
    samples: Int,
):
    if clusters * features * samples >= PARALLEL_THRESHOLD:
        var tasks = min(clusters, PARALLEL_TASKS)
        var data_addr = Int(data)
        var centers_addr = Int(centers)
        var distances_addr = Int(distances)

        @__parameter
        def work(task: Int):
            var start = task * clusters // tasks
            var end = (task + 1) * clusters // tasks
            for k in range(start, end):
                compute_distances_for_cluster(
                    p(data_addr),
                    p(centers_addr),
                    p(distances_addr),
                    k,
                    features,
                    samples,
                )

        map[work](tasks)
    else:
        for k in range(clusters):
            compute_distances_for_cluster(
                data, centers, distances, k, features, samples
            )


def objective_sum(um: Ptr, distances: Ptr, count: Int) -> Float64:
    comptime W = simd_width_of[DType.float64]()
    var vector_end = count - count % W
    var vector_total = SIMD[DType.float64, W](0.0)
    for i in range(0, vector_end, W):
        var distance = distances.load[width=W](i)
        vector_total += um.load[width=W](i) * distance * distance
    var total = vector_total.reduce_add()
    for i in range(vector_end, count):
        total += um[i] * distances[i] * distances[i]
    return total


def update_memberships_simd(
    u: Ptr,
    distances: Ptr,
    clusters: Int,
    samples: Int,
    exponent: Float64,
    block_start: Int,
    block_end: Int,
):
    comptime W = simd_width_of[DType.float64]()
    for block in range(block_start, block_end):
        var j = block * W
        var max_distance = distances.load[width=W](j)
        for k in range(1, clusters):
            max_distance = max(
                max_distance, distances.load[width=W](k * samples + j)
            )
        var min_scaled = SIMD[DType.float64, W](1.0)
        for k in range(clusters):
            var scaled = max(
                distances.load[width=W](k * samples + j) / max_distance,
                SIMD[DType.float64, W](EPS),
            )
            min_scaled = min(min_scaled, scaled)
        var total = SIMD[DType.float64, W](0.0)
        for k in range(clusters):
            var scaled = max(
                distances.load[width=W](k * samples + j) / max_distance,
                SIMD[DType.float64, W](EPS),
            )
            var value = pow(scaled / min_scaled, exponent)
            u.store(k * samples + j, value)
            total += value
        for k in range(clusters):
            var offset = k * samples + j
            u.store(offset, u.load[width=W](offset) / total)


def update_memberships_tail(
    u: Ptr,
    distances: Ptr,
    clusters: Int,
    samples: Int,
    exponent: Float64,
    start: Int,
):
    for j in range(start, samples):
        var max_distance = distances[j]
        for k in range(1, clusters):
            if distances[k * samples + j] > max_distance:
                max_distance = distances[k * samples + j]
        var min_scaled = 1.0
        for k in range(clusters):
            var scaled = max(distances[k * samples + j] / max_distance, EPS)
            if scaled < min_scaled:
                min_scaled = scaled
        var total = 0.0
        for k in range(clusters):
            var scaled = max(distances[k * samples + j] / max_distance, EPS)
            var value = pow(scaled / min_scaled, exponent)
            u[k * samples + j] = value
            total += value
        for k in range(clusters):
            u[k * samples + j] /= total


def update_memberships(
    u: Ptr, distances: Ptr, clusters: Int, samples: Int, m: Float64
):
    comptime W = simd_width_of[DType.float64]()
    var exponent = -2.0 / (m - 1.0)
    var blocks = samples // W
    if clusters * samples >= PARALLEL_THRESHOLD:
        var u_addr = Int(u)
        var distances_addr = Int(distances)

        @__parameter
        def work(task: Int):
            var start = task * blocks // PARALLEL_TASKS
            var end = (task + 1) * blocks // PARALLEL_TASKS
            update_memberships_simd(
                p(u_addr),
                p(distances_addr),
                clusters,
                samples,
                exponent,
                start,
                end,
            )

        map[work](PARALLEL_TASKS)
    else:
        update_memberships_simd(
            u, distances, clusters, samples, exponent, 0, blocks
        )
    update_memberships_tail(
        u, distances, clusters, samples, exponent, blocks * W
    )


def distances_and_memberships(
    data: Ptr,
    centers: Ptr,
    u: Ptr,
    um: Ptr,
    distances: Ptr,
    clusters: Int,
    features: Int,
    samples: Int,
    m: Float64,
) -> Float64:
    compute_distances(
        data, centers, distances, clusters, features, samples
    )
    var objective = objective_sum(um, distances, clusters * samples)
    update_memberships(u, distances, clusters, samples, m)
    return objective


@export("msf_cmeans_step")
def msf_cmeans_step(
    data_addr: Int,
    u_addr: Int,
    centers_addr: Int,
    distances_addr: Int,
    um_addr: Int,
    clusters: Int,
    features: Int,
    samples: Int,
    m: Float64,
) abi("C") -> Float64:
    var data = p(data_addr)
    var u = p(u_addr)
    var centers = p(centers_addr)
    var um = p(um_addr)
    normalize_memberships(u, um, clusters, samples, m)
    compute_centers(data, centers, um, clusters, features, samples)
    return distances_and_memberships(
        data,
        centers,
        u,
        um,
        p(distances_addr),
        clusters,
        features,
        samples,
        m,
    )


@export("msf_cmeans_predict_step")
def msf_cmeans_predict_step(
    data_addr: Int,
    centers_addr: Int,
    u_addr: Int,
    distances_addr: Int,
    um_addr: Int,
    clusters: Int,
    features: Int,
    samples: Int,
    m: Float64,
) abi("C") -> Float64:
    var u = p(u_addr)
    var um = p(um_addr)
    normalize_memberships(u, um, clusters, samples, m)
    return distances_and_memberships(
        p(data_addr),
        p(centers_addr),
        u,
        um,
        p(distances_addr),
        clusters,
        features,
        samples,
        m,
    )


@export("msf_normdiff_and_copy")
def msf_normdiff_and_copy(
    current_addr: Int, previous_addr: Int, n: Int
) abi("C") -> Float64:
    comptime W = simd_width_of[DType.float64]()
    var current = p(current_addr)
    var previous = p(previous_addr)
    var vector_end = n - n % W
    var vector_total = SIMD[DType.float64, W](0.0)
    for i in range(0, vector_end, W):
        var value = current.load[width=W](i)
        var delta = value - previous.load[width=W](i)
        vector_total += delta * delta
        previous.store(i, value)
    var total = vector_total.reduce_add()
    for i in range(vector_end, n):
        var value = current[i]
        var delta = value - previous[i]
        total += delta * delta
        previous[i] = value
    return sqrt(total)


@export("msf_fpc")
def msf_fpc(u_addr: Int, clusters: Int, samples: Int) abi("C") -> Float64:
    comptime W = simd_width_of[DType.float64]()
    var u = p(u_addr)
    var n = clusters * samples
    var vector_end = n - n % W
    var vector_total = SIMD[DType.float64, W](0.0)
    for i in range(0, vector_end, W):
        var value = u.load[width=W](i)
        vector_total += value * value
    var total = vector_total.reduce_add()
    for i in range(vector_end, n):
        total += u[i] * u[i]
    return total / Float64(samples)


def interpolate_binary_range(
    x: Ptr,
    mf: Ptr,
    query: Ptr,
    result: Ptr,
    n: Int,
    start: Int,
    end: Int,
    zero_outside: Int,
):
    for q in range(start, end):
        var value = query[q]
        if value < x[0]:
            result[q] = 0.0 if zero_outside != 0 else mf[0]
            continue
        if value > x[n - 1]:
            result[q] = 0.0 if zero_outside != 0 else mf[n - 1]
            continue
        if value == x[n - 1]:
            result[q] = mf[n - 1]
            continue
        var lo = 0
        var hi = n - 1
        while lo + 1 < hi:
            var mid = (lo + hi) // 2
            if x[mid] <= value:
                lo = mid
            else:
                hi = mid
        var width = x[lo + 1] - x[lo]
        result[q] = mf[lo] + (value - x[lo]) * (mf[lo + 1] - mf[lo]) / width


def interpolate_sorted_range(
    x: Ptr,
    mf: Ptr,
    query: Ptr,
    result: Ptr,
    n: Int,
    start: Int,
    end: Int,
    zero_outside: Int,
):
    if start >= end:
        return
    var lo = 0
    var first = query[start]
    if first >= x[0] and first < x[n - 1]:
        var hi = n - 1
        while lo + 1 < hi:
            var mid = (lo + hi) // 2
            if x[mid] <= first:
                lo = mid
            else:
                hi = mid
    for q in range(start, end):
        var value = query[q]
        if value < x[0]:
            result[q] = 0.0 if zero_outside != 0 else mf[0]
            continue
        if value > x[n - 1]:
            result[q] = 0.0 if zero_outside != 0 else mf[n - 1]
            continue
        if value == x[n - 1]:
            result[q] = mf[n - 1]
            continue
        while x[lo + 1] <= value:
            lo += 1
        var width = x[lo + 1] - x[lo]
        result[q] = mf[lo] + (value - x[lo]) * (mf[lo + 1] - mf[lo]) / width


def interpolate_queries(
    x_addr: Int,
    mf_addr: Int,
    query_addr: Int,
    result_addr: Int,
    n: Int,
    queries: Int,
    zero_outside: Int,
    sorted: Bool,
):
    if queries >= PARALLEL_THRESHOLD:

        @__parameter
        def work(task: Int):
            var start = task * queries // PARALLEL_TASKS
            var end = (task + 1) * queries // PARALLEL_TASKS
            if sorted:
                interpolate_sorted_range(
                    p(x_addr),
                    p(mf_addr),
                    p(query_addr),
                    p(result_addr),
                    n,
                    start,
                    end,
                    zero_outside,
                )
            else:
                interpolate_binary_range(
                    p(x_addr),
                    p(mf_addr),
                    p(query_addr),
                    p(result_addr),
                    n,
                    start,
                    end,
                    zero_outside,
                )

        map[work](PARALLEL_TASKS)
    elif sorted:
        interpolate_sorted_range(
            p(x_addr),
            p(mf_addr),
            p(query_addr),
            p(result_addr),
            n,
            0,
            queries,
            zero_outside,
        )
    else:
        interpolate_binary_range(
            p(x_addr),
            p(mf_addr),
            p(query_addr),
            p(result_addr),
            n,
            0,
            queries,
            zero_outside,
        )


@export("msf_interp_membership")
def msf_interp_membership(
    x_addr: Int,
    mf_addr: Int,
    query_addr: Int,
    result_addr: Int,
    n: Int,
    queries: Int,
    zero_outside: Int,
) abi("C"):
    var query = p(query_addr)
    var sorted = True
    for q in range(1, queries):
        if not query[q] >= query[q - 1]:
            sorted = False
            break
    interpolate_queries(
        x_addr,
        mf_addr,
        query_addr,
        result_addr,
        n,
        queries,
        zero_outside,
        sorted,
    )


@export("msf_centroid")
def msf_centroid(x_addr: Int, mf_addr: Int, n: Int) abi("C") -> Float64:
    var x = p(x_addr)
    var mf = p(mf_addr)
    if n == 1:
        return x[0] * mf[0] / max(mf[0], EPS)
    var moment_area = 0.0
    var total_area = 0.0
    for i in range(1, n):
        var x1 = x[i - 1]
        var x2 = x[i]
        var y1 = mf[i - 1]
        var y2 = mf[i]
        if (y1 == 0.0 and y2 == 0.0) or x1 == x2:
            continue
        var area: Float64
        var moment: Float64
        if y1 == y2:
            moment = 0.5 * (x1 + x2)
            area = (x2 - x1) * y1
        elif y1 == 0.0:
            moment = 2.0 / 3.0 * (x2 - x1) + x1
            area = 0.5 * (x2 - x1) * y2
        elif y2 == 0.0:
            moment = 1.0 / 3.0 * (x2 - x1) + x1
            area = 0.5 * (x2 - x1) * y1
        else:
            moment = 2.0 / 3.0 * (x2 - x1) * (y2 + 0.5 * y1) / (y1 + y2) + x1
            area = 0.5 * (x2 - x1) * (y1 + y2)
        moment_area += moment * area
        total_area += area
    return moment_area / max(total_area, EPS)


@export("msf_bisector")
def msf_bisector(x_addr: Int, mf_addr: Int, n: Int) abi("C") -> Float64:
    var x = p(x_addr)
    var mf = p(mf_addr)
    if n == 1:
        return x[0]
    var total_area = 0.0
    for i in range(1, n):
        total_area += 0.5 * (x[i] - x[i - 1]) * (mf[i - 1] + mf[i])
    var target = total_area * 0.5
    var accumulated = 0.0
    for i in range(1, n):
        var x1 = x[i - 1]
        var x2 = x[i]
        var y1 = mf[i - 1]
        var y2 = mf[i]
        var area = 0.5 * (x2 - x1) * (y1 + y2)
        if accumulated + area >= target:
            var subarea = target - accumulated
            var width = x2 - x1
            if y1 == y2:
                return x1 + subarea / y1
            if y1 == 0.0:
                return x1 + sqrt(2.0 * subarea * width / y2)
            if y2 == 0.0:
                return x2 - sqrt(width * width - 2.0 * subarea * width / y1)
            var slope = (y2 - y1) / width
            return x1 + (sqrt(y1 * y1 + 2.0 * slope * subarea) - y1) / slope
        accumulated += area
    return x[n - 1]


@export("msf_aggregate")
def msf_aggregate(
    strengths_addr: Int,
    membership_addr: Int,
    result_addr: Int,
    rules: Int,
    n: Int,
) abi("C"):
    var strengths = p(strengths_addr)
    var membership = p(membership_addr)
    var result = p(result_addr)
    for i in range(n):
        var value = 0.0
        for r in range(rules):
            var clipped = min(strengths[r], membership[r * n + i])
            if clipped > value:
                value = clipped
        result[i] = value


@export("msf_trimf")
def msf_trimf(x_addr: Int, result_addr: Int, n: Int, a: Float64, b: Float64, c: Float64) abi("C"):
    var x = p(x_addr)
    var result = p(result_addr)
    for i in range(n):
        var value = 0.0
        if x[i] == b:
            value = 1.0
        elif x[i] > a and x[i] < b:
            value = (x[i] - a) / (b - a)
        elif x[i] > b and x[i] < c:
            value = (c - x[i]) / (c - b)
        result[i] = value


@export("msf_trapmf")
def msf_trapmf(
    x_addr: Int,
    result_addr: Int,
    n: Int,
    a: Float64,
    b: Float64,
    c: Float64,
    d: Float64,
) abi("C"):
    var x = p(x_addr)
    var result = p(result_addr)
    for i in range(n):
        var value = 0.0
        if x[i] >= b and x[i] <= c:
            value = 1.0
        elif x[i] > a and x[i] < b:
            value = (x[i] - a) / (b - a)
        elif x[i] > c and x[i] < d:
            value = (d - x[i]) / (d - c)
        result[i] = value


@export("msf_gaussmf")
def msf_gaussmf(
    x_addr: Int, result_addr: Int, n: Int, mean: Float64, sigma: Float64
) abi("C"):
    var x = p(x_addr)
    var result = p(result_addr)
    for i in range(n):
        var z = (x[i] - mean) / sigma
        result[i] = exp(-0.5 * z * z)


@export("msf_gbellmf")
def msf_gbellmf(
    x_addr: Int, result_addr: Int, n: Int, a: Float64, b: Float64, c: Float64
) abi("C"):
    var x = p(x_addr)
    var result = p(result_addr)
    for i in range(n):
        result[i] = 1.0 / (1.0 + pow(abs((x[i] - c) / a), 2.0 * b))


@export("msf_sigmf")
def msf_sigmf(
    x_addr: Int, result_addr: Int, n: Int, b: Float64, c: Float64
) abi("C"):
    var x = p(x_addr)
    var result = p(result_addr)
    for i in range(n):
        result[i] = 1.0 / (1.0 + exp(-c * (x[i] - b)))
