"""ctypes loader for the Mojo kernels."""

from __future__ import annotations

import ctypes
import os
import threading

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJOFUZZY_LIB", os.path.join(ROOT, "dist", "libmojo-scikit-fuzzy.so"))

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "msf_cmeans_step": ([I] * 8 + [F], F),
    "msf_cmeans_predict_step": ([I] * 8 + [F], F),
    "msf_cmeans_predict_cached_step": ([I] * 5 + [F], F),
    "msf_normdiff_and_copy": ([I, I, I], F),
    "msf_fpc": ([I, I, I], F),
    "msf_interp_membership": ([I] * 7, None),
    "msf_centroid": ([I, I, I], F),
    "msf_bisector": ([I, I, I], F),
    "msf_aggregate": ([I, I, I, I, I], None),
    "msf_trimf": ([I, I, I, F, F, F], None),
    "msf_trapmf": ([I, I, I, F, F, F, F], None),
    "msf_gaussmf": ([I, I, I, F, F], None),
    "msf_gaussmf_gpu": ([I, I, I, F, F], I),
    "msf_gbellmf": ([I, I, I, F, F, F], None),
    "msf_sigmf": ([I, I, I, F, F], None),
}

_handle: ctypes.CDLL | None = None
_handle_lock = threading.Lock()
_runtime = threading.local()


def lib() -> ctypes.CDLL:
    global _handle
    if _handle is None:
        with _handle_lock:
            if _handle is None:
                if not os.path.isfile(LIB):
                    raise RuntimeError("Mojo library is not built; run `pixi run build`")
                handle = ctypes.CDLL(LIB)
                for name, (argtypes, restype) in _SIGNATURES.items():
                    function = getattr(handle, name)
                    function.argtypes = argtypes
                    function.restype = restype
                _handle = handle
    if not hasattr(_runtime, "cpu_device"):
        initialize = _handle.KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice
        initialize.argtypes = []
        initialize.restype = ctypes.c_void_p
        device = initialize()
        if not device:
            raise RuntimeError("Mojo CPU runtime initialization failed")
        _runtime.cpu_device = device
    return _handle


def f64(value, *, copy: bool = False) -> np.ndarray:
    if (
        not copy
        and isinstance(value, np.ndarray)
        and value.dtype == np.float64
        and value.flags.c_contiguous
    ):
        return value
    source = np.asarray(value)
    if np.issubdtype(source.dtype, np.complexfloating):
        raise TypeError("complex inputs are not supported")
    if copy:
        return np.array(source, dtype=np.float64, order="C", copy=True)
    return np.ascontiguousarray(source, dtype=np.float64)


def addr(array: np.ndarray) -> int:
    if (
        not isinstance(array, np.ndarray)
        or array.dtype != np.float64
        or not array.flags.c_contiguous
        or array.size == 0
    ):
        raise ValueError("FFI buffers must be non-empty, C-contiguous float64 arrays")
    address = int(array.ctypes.data)
    if address == 0:
        raise RuntimeError("NumPy returned a null buffer address")
    return address
