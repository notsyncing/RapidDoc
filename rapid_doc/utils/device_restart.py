"""Restart the process when the SYCL/Level-Zero device is unusable.

Two failures reach this module:

* ``UR_RESULT_ERROR_DEVICE_LOST`` - the device is gone, and with it the
  process's Unified Runtime context and every queue/executor built on it.
  sycl-infer has no reset entry point, and recreating one session does not
  help because the per-device queue is cached for the process lifetime.
* a device *allocation* failure (``device allocation failed``) - the card has
  no room left for the next allocation.  The executor already trims its pool
  and retries at the allocation site (``executor::acquire``), so a failure
  that escapes to Python means the process's resident models plus the call's
  working set do not fit in the card, and every later call allocates against
  the same exhausted budget.  Only a fresh process returns that memory.

Exiting is therefore the recovery for both; the container restart policy (or
docker/start_with_env.sh) starts a clean process.

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
    "is_device_memory_exhausted",
    "needs_restart",
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

# A failed ``sycl::malloc_device`` is the executor's own wording, and it
# appears in three phrasings there ("device allocation failed", "the device
# allocation for the byte cast of '...' failed", "device allocation of N
# bytes failed after trimming"); matching the common prefix covers all of
# them.  The Unified Runtime names are what a Level Zero / OpenCL allocation
# refusal carries.  A generic "out of memory" is deliberately not matched: a
# host-side ``std::bad_alloc`` while parsing a model is not fixed by a
# restart.
_DEVICE_MEMORY_MARKERS = (
    "device allocation",
    "ur_result_error_out_of_device_memory",
    "ur_result_error_out_of_resources",
    "out_of_device_memory",
)

_EXIT_LOCK = threading.Lock()
_EXITING = False


def is_device_lost(error: Union[BaseException, str, None]) -> bool:
    """True when *error* reports an unrecoverable device-lost failure."""
    if error is None:
        return False
    text = str(error).lower()
    return any(marker in text for marker in _DEVICE_LOST_MARKERS)


def is_device_memory_exhausted(error: Union[BaseException, str, None]) -> bool:
    """True when *error* reports a device allocation failure (out of VRAM)."""
    if error is None:
        return False
    text = str(error).lower()
    return any(marker in text for marker in _DEVICE_MEMORY_MARKERS)


def needs_restart(error: Union[BaseException, str, None]) -> bool:
    """True when no in-process retry can recover the device.

    Covers a lost device and a failed device allocation: the first has no
    usable context left, and the second has no memory left for this process
    (the executor has already trimmed and retried in place).
    """
    return is_device_lost(error) or is_device_memory_exhausted(error)


def restart_enabled() -> bool:
    """RAPID_DOC_RESTART_ON_DEVICE_LOST (default: on).

    The switch covers both failure kinds above; the name is kept for the
    existing deployments.
    """
    value = os.getenv("RAPID_DOC_RESTART_ON_DEVICE_LOST", "1")
    return value.strip().lower() not in {"0", "false", "no", "off", "disabled"}


def restart_exit_code() -> int:
    """The exit status the supervisor treats as "restart me"."""
    try:
        return int(os.getenv(
            "RAPID_DOC_DEVICE_LOST_EXIT_CODE", str(DEFAULT_RESTART_EXIT_CODE)))
    except (TypeError, ValueError):
        return DEFAULT_RESTART_EXIT_CODE


def _failure_kind(reason: Union[BaseException, str, None]) -> str:
    """A label for the log line: why this process is being restarted."""
    if is_device_lost(reason):
        return "device lost"
    if is_device_memory_exhausted(reason):
        return "device memory exhausted"
    return "device failure"


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
            "[{}] GPU {}: {}. Restarting the process "
            "(exit {}); set RAPID_DOC_RESTART_ON_DEVICE_LOST=0 to disable.",
            source, _failure_kind(reason), reason, code)
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:
                pass
        os._exit(code)

    return True  # unreachable; keeps type checkers happy
