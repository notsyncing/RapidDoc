"""Restart the process when the SYCL/Level-Zero device is gone.

A ``UR_RESULT_ERROR_DEVICE_LOST`` is not a per-model failure: the process's
Unified Runtime context and every queue/executor built on it are dead.  The
sycl-infer library has no reset entry point, and recreating one session does
not help because the per-device queue is cached for the process lifetime.
Exiting is therefore the only way to get a fresh GPU context; the container
restart policy (or docker/start_with_env.sh) starts a clean process.

The exit is deliberately immediate.  The failed request cannot be recovered,
and on some driver versions teardown while the context is lost hangs.
"""
from __future__ import annotations

import os
import sys
import threading
from typing import Union

from loguru import logger

__all__ = [
    "DEFAULT_RESTART_EXIT_CODE",
    "is_device_lost",
    "restart_enabled",
    "restart_exit_code",
    "restart_process",
]

DEFAULT_RESTART_EXIT_CODE = 75

# Strings the Level Zero / Unified Runtime stack has used for this failure.
# Matching is case-insensitive; "device lost" covers the adapter prefixes.
_DEVICE_LOST_MARKERS = (
    "ur_result_error_device_lost",
    "ze_result_error_device_lost",
    "level_zero backend failed with error: 20",
    "device lost",
)

_EXIT_LOCK = threading.Lock()
_EXITING = False


def is_device_lost(error: Union[BaseException, str, None]) -> bool:
    """True when *error* reports an unrecoverable device-lost failure."""
    if error is None:
        return False
    text = str(error).lower()
    return any(marker in text for marker in _DEVICE_LOST_MARKERS)


def restart_enabled() -> bool:
    """RAPID_DOC_RESTART_ON_DEVICE_LOST (default: on)."""
    value = os.getenv("RAPID_DOC_RESTART_ON_DEVICE_LOST", "1")
    return value.strip().lower() not in {"0", "false", "no", "off", "disabled"}


def restart_exit_code() -> int:
    """The exit status the supervisor treats as "restart me"."""
    try:
        return int(os.getenv(
            "RAPID_DOC_DEVICE_LOST_EXIT_CODE", str(DEFAULT_RESTART_EXIT_CODE)))
    except (TypeError, ValueError):
        return DEFAULT_RESTART_EXIT_CODE


def restart_process(reason: Union[BaseException, str], *,
                    source: str = "sycl-infer") -> bool:
    """Exit so the API process is started again with a clean device context.

    Returns ``False`` when automatic restart is disabled.  When enabled the
    call does not return: ``os._exit`` avoids atexit handlers, Python cleanup
    and driver teardown that can block on a lost device.
    """
    global _EXITING
    if not restart_enabled():
        return False

    with _EXIT_LOCK:
        if _EXITING:
            return True
        _EXITING = True
        code = restart_exit_code()
        logger.error(
            "[{}] GPU device lost: {}. Restarting the process "
            "(exit {}); set RAPID_DOC_RESTART_ON_DEVICE_LOST=0 to disable.",
            source, reason, code)
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:
                pass
        os._exit(code)

    return True  # unreachable; keeps type checkers happy
