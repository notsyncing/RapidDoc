import pytest

import rapid_doc.utils.openvino_core as openvino_core


class _FakeCore:
    def __init__(self):
        self.properties = {}

    def set_property(self, device, config):
        self.properties[device] = config


@pytest.fixture
def fake_core(monkeypatch):
    monkeypatch.setattr(openvino_core, "Core", _FakeCore)
    monkeypatch.setattr(openvino_core, "_CORE", None)
    yield
    monkeypatch.setattr(openvino_core, "_CORE", None)


def test_gpu_disk_cache_is_disabled_by_default(monkeypatch, fake_core):
    monkeypatch.delenv("RAPID_DOC_OPENVINO_CACHE_DIR", raising=False)

    assert openvino_core.get_gpu_cache_dir() is None
    core = openvino_core.get_openvino_core()

    assert core.properties == {}


def test_gpu_disk_cache_requires_explicit_opt_in(monkeypatch, tmp_path, fake_core):
    cache_dir = tmp_path / "ov-cache"
    monkeypatch.setenv("RAPID_DOC_OPENVINO_CACHE_DIR", str(cache_dir))

    versioned = cache_dir / "static-shape-v1"
    assert openvino_core.get_gpu_cache_dir() == str(versioned)
    core = openvino_core.get_openvino_core()

    assert core.properties["GPU"]["CACHE_DIR"] == str(versioned)
    assert versioned.is_dir()
