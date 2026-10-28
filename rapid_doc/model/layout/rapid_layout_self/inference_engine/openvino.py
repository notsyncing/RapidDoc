import os
import traceback
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
from importlib.metadata import version
from packaging.version import Version

try:
    import openvino as ov
    try:
        from openvino import Core
    except ImportError:
        from openvino.runtime import Core  # 兼容旧版本
except ImportError:
    raise ImportError(
        "openvino is not installed. Please install it with: pip install openvino"
    )
openvino_version = Version(version("openvino"))
if openvino_version < Version("2025.4.0"):
    raise ImportError(
        f"openvino version must be >= 2025.4.0, but found {openvino_version}. "
        "Please upgrade with: pip install -U openvino"
    )

from ..model_handler.utils import ModelProcessor
from ..utils.logger import Logger
from rapid_doc.utils.openvino_core import get_openvino_core
from rapid_doc.utils.openvino_static import StaticShapeRunner
from ..utils.typings import RapidLayoutInput
from .base import InferSession

class OpenVINOInferSession(InferSession):
    def __init__(self, cfg: RapidLayoutInput):
        super().__init__(cfg)
        self.logger = Logger(logger_name=__name__).get_log()

        if cfg.model_dir_or_path is None:
            model_path = ModelProcessor.get_model_path(cfg.model_type)
        else:
            model_path = Path(cfg.model_dir_or_path)

        self._verify_model(model_path)
        self.model_path = model_path
        self.logger.info(f"Using {model_path}")

        device = cfg.engine_cfg.get('device', 'CPU') if cfg.engine_cfg else 'CPU'
        ov_config = self._init_config(cfg)
        self.device = device

        self.core = get_openvino_core()
        # Keep the unreshaped model for labels/RT metadata, but run every
        # inference on a static-shape compiled bucket.
        self.model = self.core.read_model(model=str(self.model_path))
        self.input_tensor = self.model.inputs[0]
        self.output_tensors = self.model.outputs
        self.runner = StaticShapeRunner(
            self.core,
            self.model_path,
            device,
            ov_config,
            label="layout",
        )

    def _init_config(self, cfg: RapidLayoutInput) -> Dict[Any, Any]:
        config = {}
        engine_cfg = cfg.engine_cfg

        performance_hint = engine_cfg.get("performance_hint", None)
        if performance_hint is not None:
            config["PERFORMANCE_HINT"] = str(performance_hint)

        performance_num_requests = engine_cfg.get("performance_num_requests", -1)
        if performance_num_requests != -1:
            config["PERFORMANCE_HINT_NUM_REQUESTS"] = str(performance_num_requests)

        cache_dir = engine_cfg.get("cache_dir", None)
        if cache_dir is not None:
            config["CACHE_DIR"] = str(cache_dir)

        # CPU-only properties — skip for GPU to avoid plugin errors
        device = engine_cfg.get('device', 'CPU') if engine_cfg else 'CPU'
        if device.upper() == 'GPU':
            # Intel GPU 默认 FP16；部分设备/驱动上 FP16 conv 会产生错误输出
            # （layout/OCR/table 全空）。默认用 FP32，可用环境变量覆盖。
            precision = engine_cfg.get("inference_precision") or os.getenv(
                "RAPID_DOC_INFERENCE_PRECISION", "f32"
            )
            if precision:
                config["INFERENCE_PRECISION_HINT"] = str(precision)
            # Only opt in for a genuinely large single allocation. Enabling
            # this globally switches the GPU allocator mode and can destabilize
            # the context for ordinary layout inputs.
            enable_large_allocations = engine_cfg.get("enable_large_allocations", False)
            config["GPU_ENABLE_LARGE_ALLOCATIONS"] = bool(enable_large_allocations)
        else:
            infer_num_threads = engine_cfg.get("inference_num_threads", -1)
            if infer_num_threads != -1 and 1 <= infer_num_threads <= os.cpu_count():
                config["INFERENCE_NUM_THREADS"] = str(infer_num_threads)

            enable_cpu_pinning = engine_cfg.get("enable_cpu_pinning", None)
            if enable_cpu_pinning is not None:
                config["ENABLE_CPU_PINNING"] = str(enable_cpu_pinning)

            num_streams = engine_cfg.get("num_streams", -1)
            if num_streams != -1:
                config["NUM_STREAMS"] = str(num_streams)

            enable_hyper_threading = engine_cfg.get("enable_hyper_threading", None)
            if enable_hyper_threading is not None:
                config["ENABLE_HYPER_THREADING"] = str(enable_hyper_threading)

            scheduling_core_type = engine_cfg.get("scheduling_core_type", None)
            if scheduling_core_type is not None:
                config["SCHEDULING_CORE_TYPE"] = str(scheduling_core_type)

        self.logger.info(f"Using OpenVINO config: {config}")
        return config

    def __call__(self, input_content: np.ndarray, scale_factor: np.ndarray = None) -> Any:
        try:
            inputs = {"image": np.asarray(input_content)}
            if scale_factor is not None:
                inputs["scale_factor"] = np.asarray(scale_factor)
            if "im_shape" in self.get_input_names():
                h, w = input_content.shape[-2:]
                batch = input_content.shape[0]
                inputs["im_shape"] = np.asarray(
                    [[h, w]], dtype=np.float32
                ).repeat(batch, axis=0)
            return self.runner.run(inputs)
        except Exception as e:
            error_info = traceback.format_exc()
            raise OpenVIONError(error_info) from e

    def precompile(self, batch: int = 1) -> None:
        h = w = 800
        inputs = {
            "image": np.zeros((batch, 3, h, w), dtype=np.float32),
            "im_shape": np.asarray([[h, w]], dtype=np.float32).repeat(batch, axis=0),
            "scale_factor": np.ones((batch, 2), dtype=np.float32),
        }
        names = set(self.get_input_names())
        self.runner.precompile({k: v for k, v in inputs.items() if k in names})

    def get_input_names(self) -> List[str]:
        return [tensor.get_any_name() for tensor in self.model.inputs]

    def get_output_names(self) -> List[str]:
        return [tensor.get_any_name() for tensor in self.model.outputs]

    @property
    def characters(self):
        return self.get_character_list()

    def get_character_list(self, key: str = "character") -> List[str]:
        val = self.model.get_rt_info()["framework"][key]
        return val.value.splitlines()

    def have_key(self, key: str = "character") -> bool:
        try:
            rt_info = self.model.get_rt_info()
            return key in rt_info
        except:
            return False


class OpenVIONError(Exception):
    pass
