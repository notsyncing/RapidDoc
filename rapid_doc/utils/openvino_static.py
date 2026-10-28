"""Static-shape OpenVINO execution with a process-wide GPU lock.

Every model is reshaped to the exact bucket shape before compilation.  A
static compiled model has no dynamic dimensions, so the Intel GPU plugin does
not start asynchronous kernel compilation while a later request is running.

Compilation and the first inference for a shape are performed while holding
``GPU_INFER_LOCK``.  The same lock serialises inference, so a new shape can
never be compiled in the background while another request uses the device.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from loguru import logger

try:
    import openvino as ov
except ImportError:  # pragma: no cover - OpenVINO is optional on CPU installs
    ov = None


GPU_INFER_LOCK = threading.RLock()

# A guard against accidentally compiling thousands of near-identical shapes.
# Bucket helpers keep the real number far below this; the cap is per model.
DEFAULT_MAX_STATIC_SHAPES = 64
# The same OCR model is used by the page engine and the table engine.  Share
# the compiled static model between those sessions; the global GPU lock makes
# concurrent use safe, and each session keeps its own InferRequest.
MAX_SHARED_STATIC_MODELS = 160
_SHARED_MODELS: OrderedDict = OrderedDict()


def _config_signature(ov_config: dict) -> tuple:
    return tuple(
        sorted((str(key), repr(value)) for key, value in (ov_config or {}).items())
    )


def _model_stamp(model_path: str) -> tuple[int, int]:
    try:
        stat = Path(model_path).stat()
        return stat.st_mtime_ns, stat.st_size
    except OSError:
        return 0, 0


class StaticShapeRunner:
    """Compile one static OpenVINO model per exact input-shape tuple."""

    def __init__(
        self,
        core,
        model_path,
        device: str,
        ov_config: dict | None = None,
        max_shapes: int = DEFAULT_MAX_STATIC_SHAPES,
        label: str | None = None,
        static_mode: str = "full",
        spatial_shape: tuple[int, ...] | None = None,
    ):
        if ov is None:
            raise ImportError("openvino is not installed")
        self.core = core
        self.model_path = str(model_path)
        self.device = device
        self.ov_config = dict(ov_config or {})
        self.max_shapes = max(1, int(max_shapes))
        self.label = label or Path(self.model_path).name
        self.static_mode = static_mode
        self.spatial_shape = tuple(spatial_shape) if spatial_shape else None
        self._signature = _model_stamp(self.model_path)
        # key -> (compiled_model, infer_request, tensor_names)
        self._entries: OrderedDict = OrderedDict()

    def _key(self, inputs: dict[str, np.ndarray]) -> tuple:
        return self._input_shape_map(inputs)

    @staticmethod
    def _default_aux_shape(partial_shape, batch: int) -> tuple[int, ...]:
        """Concrete shape for an input the caller did not provide."""
        dims = []
        for axis, dim in enumerate(partial_shape):
            if dim.is_static:
                dims.append(int(dim.get_length()))
            elif axis == 0:
                dims.append(batch)
            elif partial_shape.rank().is_static and partial_shape.rank().get_length() == 2:
                # Layout's im_shape / scale_factor are (batch, 2).
                dims.append(2)
            else:
                dims.append(1)
        return tuple(dims)

    def _input_shape_map(self, inputs: dict[str, np.ndarray]) -> tuple:
        shapes = []
        for name, array in sorted(inputs.items()):
            shape = tuple(int(d) for d in np.shape(array))
            if self.static_mode == "spatial" and shape:
                # Batch stays dynamic in the compiled model; every batch size
                # reuses the same CompiledModel/InferRequest.
                shape = (-1,) + shape[1:]
            shapes.append((name, shape))
        return tuple(shapes)

    def _shared_key(self, inputs: dict[str, np.ndarray]) -> tuple:
        return (
            self.model_path,
            str(self.device),
            _config_signature(self.ov_config),
            self.static_mode,
            self.spatial_shape,
            self._input_shape_map(inputs),
        )

    def _compile(self, inputs: dict[str, np.ndarray]):
        key = self._shared_key(inputs)
        shared = _SHARED_MODELS.get(key)
        if shared is not None:
            _SHARED_MODELS.move_to_end(key)
            # Sharing the InferRequest (serialised by GPU_INFER_LOCK) avoids
            # duplicating the large recognition output buffer between the
            # page OCR session and the table OCR session.
            compiled, request, output_names = shared
            return compiled, request, output_names, 0.0

        model = self.core.read_model(model=self.model_path)
        main_shape = None
        for array in inputs.values():
            main_shape = tuple(int(d) for d in np.shape(array))
            break
        batch = main_shape[0] if main_shape else 1
        reshape = {}
        for model_input in model.inputs:
            name = model_input.get_any_name()
            if name in inputs:
                concrete = [int(d) for d in np.shape(inputs[name])]
                if self.static_mode == "spatial":
                    if len(concrete) != 4:
                        raise ValueError(
                            f"spatial static mode requires a 4-D input, got {concrete}"
                        )
                    spatial = self.spatial_shape or tuple(concrete[1:])
                    reshape[name] = ov.PartialShape([-1, *spatial])
                else:
                    reshape[name] = ov.PartialShape(concrete)
            else:
                reshape[name] = ov.PartialShape(
                    self._default_aux_shape(model_input.get_partial_shape(), batch)
                )
        model.reshape(reshape)

        started = time.perf_counter()
        compiled = self.core.compile_model(model, self.device, self.ov_config)
        compile_elapsed = time.perf_counter() - started
        request = compiled.create_infer_request()
        output_names = [
            output.get_any_name() for output in compiled.outputs
        ]
        _SHARED_MODELS[key] = (compiled, request, output_names)
        while len(_SHARED_MODELS) > MAX_SHARED_STATIC_MODELS:
            _SHARED_MODELS.popitem(last=False)
        return compiled, request, output_names, compile_elapsed

    @staticmethod
    def _infer(compiled, request, output_names, inputs: dict[str, np.ndarray]):
        for name, array in inputs.items():
            request.set_tensor(name, ov.Tensor(np.ascontiguousarray(array)))
        request.start_async()
        request.wait()
        return [
            np.array(request.get_tensor(name).data) for name in output_names
        ]

    def run(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        """Execute ``inputs`` on a statically compiled bucket model."""
        if not inputs:
            raise ValueError("no inputs supplied")
        key = self._key(inputs)
        with GPU_INFER_LOCK:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
                compiled, request, output_names = entry
                return self._infer(compiled, request, output_names, inputs)

            compiled, request, output_names, compile_elapsed = self._compile(inputs)
            # First inference happens under the lock as well.  If the plugin
            # defers any work until the first call, it cannot overlap a
            # request that is already running.
            outputs = self._infer(compiled, request, output_names, inputs)
            self._entries[key] = (compiled, request, output_names)
            while len(self._entries) > self.max_shapes:
                evicted_key, _ = self._entries.popitem(last=False)
                logger.warning(
                    "OpenVINO static shape cache evicted {} {} (cap {})",
                    self.label,
                    evicted_key,
                    self.max_shapes,
                )
            if compile_elapsed > 0:
                logger.info(
                    "OpenVINO static compile {} {} {:.2f}s",
                    self.label,
                    tuple(key[0][1]) if len(key) == 1 else key,
                    compile_elapsed,
                )
            else:
                logger.info(
                    "OpenVINO static reuse {} {}",
                    self.label,
                    tuple(key[0][1]) if len(key) == 1 else key,
                )
            return outputs

    def precompile(self, inputs: dict[str, np.ndarray]) -> None:
        """Compile and warm one bucket without going through a dynamic model."""
        self.run(inputs)

    def cached_shapes(self) -> list[tuple]:
        with GPU_INFER_LOCK:
            return list(self._entries.keys())
