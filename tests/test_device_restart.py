"""Automatic restart on an unusable SYCL/Level-Zero device.

Covers both a lost device and a device allocation failure (out of VRAM).
"""
import pytest

from rapid_doc.utils import device_restart

_ALLOCATION_FAILURES = [
    "the operator 1 (Conv) failed while preparing its inputs: device allocation failed",
    "device allocation failed",
    "the device allocation for the byte cast of 'output' failed",
    "level_zero backend failed with error: 40 (UR_RESULT_ERROR_OUT_OF_RESOURCES)",
    "UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY",
]

_OTHER_FAILURES = [
    "",
    None,
    "the ONNX run failed",
    "out of memory while loading model.onnx",
    "cannot open model.onnx",
]


@pytest.mark.parametrize(
    "message",
    [
        "level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)",
        "ur_result_error_device_lost",
        "ZE_RESULT_ERROR_DEVICE_LOST",
        "The device lost during the kernel",
    ],
)
def test_is_device_lost_recognizes_driver_messages(message):
    assert device_restart.is_device_lost(message)


@pytest.mark.parametrize("message", _OTHER_FAILURES)
def test_is_device_lost_ignores_other_failures(message):
    assert not device_restart.is_device_lost(message)


@pytest.mark.parametrize("message", _ALLOCATION_FAILURES)
def test_is_device_memory_exhausted_recognizes_allocation_failures(message):
    assert device_restart.is_device_memory_exhausted(message)
    # Distinct from a lost device: the context is still usable, only the
    # memory budget is gone.  Keeping the predicates apart lets the caller
    # report which one happened.
    assert not device_restart.is_device_lost(message)


@pytest.mark.parametrize("message", _OTHER_FAILURES)
def test_is_device_memory_exhausted_ignores_other_failures(message):
    assert not device_restart.is_device_memory_exhausted(message)


@pytest.mark.parametrize(
    "message", ["UR_RESULT_ERROR_DEVICE_LOST", *_ALLOCATION_FAILURES])
def test_needs_restart_covers_lost_and_exhausted(message):
    assert device_restart.needs_restart(message)


@pytest.mark.parametrize("message", _OTHER_FAILURES)
def test_needs_restart_ignores_other_failures(message):
    assert not device_restart.needs_restart(message)


@pytest.mark.parametrize(
    "message", ["UR_RESULT_ERROR_DEVICE_LOST", *_ALLOCATION_FAILURES])
def test_restart_process_exits_with_restart_code(monkeypatch, message):
    exits = []
    monkeypatch.setattr(device_restart, "_EXITING", False)
    monkeypatch.setenv("RAPID_DOC_RESTART_ON_DEVICE_LOST", "1")
    monkeypatch.setenv("RAPID_DOC_DEVICE_LOST_EXIT_CODE", "75")
    monkeypatch.setattr(device_restart.os, "_exit", exits.append)

    assert device_restart.restart_process(message) is True
    assert exits == [75]


def test_restart_process_can_be_disabled(monkeypatch):
    exits = []
    monkeypatch.setattr(device_restart, "_EXITING", False)
    monkeypatch.setenv("RAPID_DOC_RESTART_ON_DEVICE_LOST", "0")
    monkeypatch.setattr(device_restart.os, "_exit", exits.append)

    assert device_restart.restart_process("UR_RESULT_ERROR_DEVICE_LOST") is False
    assert exits == []


def _record_restarts(monkeypatch):
    """Patch the session's restart hook and return the recorded calls."""
    from rapid_doc.model import sycl_infer_onnx as session

    calls = []
    monkeypatch.setattr(
        session, "restart_process",
        lambda reason, **kw: calls.append((str(reason), kw)))
    return session, calls


def test_session_restarts_on_device_allocation_failure(monkeypatch):
    session, calls = _record_restarts(monkeypatch)
    message = ("the operator 1 (Conv) failed while preparing its inputs: "
               "device allocation failed")

    with pytest.raises(session.SyclOutOfMemoryError):
        session._raise_failure(message, source="sycl-infer:layout")

    assert calls == [(message, {"source": "sycl-infer:layout"})]


def test_session_restarts_on_device_lost(monkeypatch):
    session, calls = _record_restarts(monkeypatch)
    message = "level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)"

    with pytest.raises(session.SyclDeviceLostError):
        session._raise_failure(message, source="sycl-infer:layout")

    assert calls == [(message, {"source": "sycl-infer:layout"})]


def test_session_does_not_restart_on_a_model_failure(monkeypatch):
    session, calls = _record_restarts(monkeypatch)

    with pytest.raises(session.SyclInferError) as excinfo:
        session._raise_failure("cannot open model.onnx")

    assert not isinstance(excinfo.value, session.SyclOutOfMemoryError)
    assert calls == []
