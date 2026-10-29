# Copyright (c) Opendatalab. All rights reserved.
import os
import threading

try:
    import openvino as ov
    try:
        from openvino import Core
    except ImportError:
        from openvino.runtime import Core  # 兼容旧版本
except ImportError:
    ov = None
    Core = None


_CORE = None
_CORE_LOCK = threading.Lock()

_CACHE_DIR_ENV = "RAPID_DOC_OPENVINO_CACHE_DIR"
_CACHE_LAYOUT_VERSION = "static-shape-v1"


def get_gpu_cache_dir():
    """Return the opt-in OpenVINO GPU disk cache directory, or ``None``.

    The GPU plugin's on-disk blob cache is disabled by default.  A blob can be
    left behind by an interrupted compile (container restart, deploy, SIGKILL)
    or belong to a previous model/build; the plugin then loads it and silently
    returns empty OCR detections instead of raising, which empties table cells.
    The in-memory ``StaticShapeRunner`` cache already avoids recompiling a
    bucket within a process, so the disk cache only affects cold starts.
    Set ``RAPID_DOC_OPENVINO_CACHE_DIR`` to opt back in.
    """
    base = os.getenv(_CACHE_DIR_ENV)
    if not base:
        return None
    # Namespace by cache layout so blobs written by older builds (before the
    # static-shape runner) can never be picked up again.
    return os.path.join(base, _CACHE_LAYOUT_VERSION)


def get_openvino_core():
    """Return the process-wide OpenVINO Core.

    The GPU plugin keeps state inside Core. Creating and dropping multiple
    Core objects while compiled models are still alive can leave dangling
    plugin state and crash during later model compilation, so all inference
    engines share one long-lived instance.
    """
    global _CORE
    if Core is None:
        raise ImportError("openvino is not installed")
    with _CORE_LOCK:
        if _CORE is None:
            _CORE = Core()
            cache_dir = get_gpu_cache_dir()
            if cache_dir:
                try:
                    os.makedirs(cache_dir, exist_ok=True)
                    _CORE.set_property("GPU", {"CACHE_DIR": cache_dir})
                except Exception:
                    pass
        return _CORE
