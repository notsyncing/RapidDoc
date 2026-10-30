"""One GPU, many host threads.

The pipeline is a sequence of model calls interleaved with host work (crops,
table HTML assembly, middle-json conversion, PDF rendering, file writes).  The
device is idle during the host half of every window, and the host is idle
during every model call.  Running two documents concurrently in one process
lets each thread's host work run while the other holds the device.

The device calls themselves must not overlap: every session shares one in-order
queue per device and several model objects keep per-call scratch.  A single
process-wide re-entrant lock serialises the *model entry points* only - the
host code around them stays concurrent.  ctypes releases the GIL inside a
``syi_onnx_run_*`` call, so the other thread really does run Python host work
while the device is busy.
"""
import functools
import os
import threading
import time

_LOCK = threading.RLock()
_LOCAL = threading.local()

_stats_on = os.getenv("RAPID_DOC_GPU_STATS") == "1"
STATS = {"hold": 0.0, "wait": 0.0, "calls": 0}


def gpu_lock():
    return _LOCK


def serialized(fn):
    """Run fn under the process-wide device lock."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not _stats_on:
            with _LOCK:
                return fn(*args, **kwargs)
        # Count only the outermost acquisition: the model entry points nest
        # (table predict -> table call -> session run), and counting each
        # level would multiply the same device time.
        depth = getattr(_LOCAL, "depth", 0)
        t0 = t1 = None
        if depth == 0:
            t0 = time.perf_counter()
        _LOCK.acquire()
        if depth == 0:
            t1 = time.perf_counter()
        _LOCAL.depth = depth + 1
        try:
            return fn(*args, **kwargs)
        finally:
            _LOCAL.depth = depth
            _LOCK.release()
            if depth == 0:
                now = time.perf_counter()
                STATS["hold"] += now - t1
                STATS["wait"] += t1 - t0
                STATS["calls"] += 1
                owner = getattr(fn, "__self__", None)
                name = getattr(owner, "model", None)
                if name:
                    models = STATS.setdefault("models", {})
                    m = models.setdefault(name, [0.0, 0])
                    m[0] += now - t1
                    m[1] += 1
    return wrapper


def serialized_per_instance(fn):
    """Serialise calls to one bound object, but not across objects.

    The C API already holds a per-model mutex, and every model shares one
    in-order queue.  Different models can therefore be in flight at once: one
    session's host submission runs while another's kernels execute, which is
    the only way to fill the device's sub-100% bus because the Python host
    work is GIL-bound.  Calls to the *same* session stay serialised (its
    executor pool and output buffers are not re-entrant).
    """
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        lock = getattr(self, "_run_lock", None)
        if lock is None:
            lock = _LOCK
        with lock:
            return fn(self, *args, **kwargs)
    return wrapper


def doc_workers():
    """How many documents to process concurrently (RAPID_DOC_DOC_WORKERS)."""
    try:
        return max(1, int(os.getenv("RAPID_DOC_DOC_WORKERS", "6")))
    except (TypeError, ValueError):
        return 1
