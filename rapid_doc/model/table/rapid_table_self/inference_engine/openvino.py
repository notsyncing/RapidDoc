import os
import traceback
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
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

from ..model_processor.main import ModelProcessor
from ..utils.logger import Logger
from .base import InferSession
from rapid_doc.utils.openvino_core import get_openvino_core
from rapid_doc.utils.openvino_static import StaticShapeRunner

class OpenVINOInferSession(InferSession):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.logger = Logger(logger_name=__name__).get_log()

        if cfg["model_dir_or_path"] is None:
            model_path = ModelProcessor.get_model_path(cfg['model_type'])
        else:
            model_path = Path(cfg["model_dir_or_path"])

        self._verify_model(model_path)
        self.model_path = model_path
        self.logger.info(f"Using {model_path}")

        device = cfg.get('engine_cfg', {}).get('device', 'CPU')
        ov_config = self._init_config(cfg)
        self.device = device
        self.ov_config = ov_config

        self.core = get_openvino_core()
        # Keep the unreshaped model for character/RT metadata, but run every
        # inference on a static-shape compiled bucket.
        self.model = self.core.read_model(model=str(self.model_path))
        self.input_tensor = self.model.inputs[0]
        self.input_name = self.input_tensor.get_any_name()
        self.output_tensors = self.model.outputs
        model_type = str(cfg.get("model_type", "table"))
        # SLANET's graph cannot be compiled with fully static H/W (the GPU
        # plugin crashes), but it accepts a spatial-static, batch-dynamic
        # model.  All other table models are compiled fully static.
        is_slanet = "slanet" in model_type.lower()
        self.runner = StaticShapeRunner(
            self.core,
            self.model_path,
            device,
            ov_config,
            label=f"table-{model_type}",
            static_mode="spatial" if is_slanet else "full",
            spatial_shape=(3, 488, 488) if is_slanet else None,
        )

    def _init_config(self, cfg) -> Dict[Any, Any]:
        config = {}
        engine_cfg = cfg['engine_cfg']

        performance_hint = engine_cfg.get("performance_hint", None)
        if performance_hint is not None:
            config["PERFORMANCE_HINT"] = str(performance_hint)

        performance_num_requests = engine_cfg.get("performance_num_requests", -1)
        if performance_num_requests != -1:
            config["PERFORMANCE_HINT_NUM_REQUESTS"] = str(performance_num_requests)

        cache_dir = engine_cfg.get("cache_dir", None)
        if cache_dir is not None:
            config["CACHE_DIR"] = str(cache_dir)

        device = cfg.get('engine_cfg', {}).get('device', 'CPU')
        if device.upper() != 'GPU':
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

        device = cfg.get('engine_cfg', {}).get('device', 'CPU')
        if device.upper() == 'GPU':
            # Table models stay on the plugin default precision. Forcing f32
            # here triggers stack corruption on some Intel GPU builds.
            # Only opt in for a genuinely large single allocation. Enabling
            # this globally switches the GPU allocator mode and can destabilize
            # the context for ordinary table inputs.
            enable_large_allocations = engine_cfg.get("enable_large_allocations", False)
            config["GPU_ENABLE_LARGE_ALLOCATIONS"] = bool(enable_large_allocations)

        self.logger.info(f"Using OpenVINO config: {config}")
        return config

    def __call__(self, input_content: np.ndarray) -> Any:
        try:
            return self.runner.run(
                {self.input_name: np.asarray(input_content)}
            )
        except Exception as e:
            error_info = traceback.format_exc()
            raise OpenVIONError(error_info) from e

    def precompile(self, shape) -> None:
        self.runner.precompile(
            {self.input_name: np.zeros(shape, dtype=np.float32)}
        )

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
