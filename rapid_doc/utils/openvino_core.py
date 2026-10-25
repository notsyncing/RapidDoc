# Copyright (c) Opendatalab. All rights reserved.
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
        return _CORE
