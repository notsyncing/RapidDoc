"""Automatic restart on an unrecoverable SYCL/Level-Zero device loss."""
import pytest

from rapid_doc.utils import device_restart


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


@pytest.mark.parametrize(
    "message",
    [
        "",
        None,
        "the ONNX run failed",
        "out of memory while loading model.onnx",
        "cannot open model.onnx",
    ],
)
def test_is_device_lost_ignores_other_failures(message):
    assert not device_restart.is_device_lost(message)


def test_restart_process_exits_with_restart_code(monkeypatch):
    exits = []
    monkeypatch.setattr(device_restart, "_EXITING", False)
    monkeypatch.setenv("RAPID_DOC_RESTART_ON_DEVICE_LOST", "1")
    monkeypatch.setenv("RAPID_DOC_DEVICE_LOST_EXIT_CODE", "75")
    monkeypatch.setattr(device_restart.os, "_exit", exits.append)

    assert device_restart.restart_process("UR_RESULT_ERROR_DEVICE_LOST") is True
    assert exits == [75]


def test_restart_process_can_be_disabled(monkeypatch):
    exits = []
    monkeypatch.setattr(device_restart, "_EXITING", False)
    monkeypatch.setenv("RAPID_DOC_RESTART_ON_DEVICE_LOST", "0")
    monkeypatch.setattr(device_restart.os, "_exit", exits.append)

    assert device_restart.restart_process("UR_RESULT_ERROR_DEVICE_LOST") is False
    assert exits == []
