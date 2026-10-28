# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import os
import traceback
from pathlib import Path
from typing import Any, Dict

import numpy as np
from omegaconf import DictConfig

try:
    import openvino as ov
    try:
        from openvino import Core
    except ImportError:
        from openvino.runtime import Core #兼容旧版本
except ImportError:
    raise ImportError(
        "openvino is not installed. Please install it with: pip install openvino"
    )

from rapidocr.utils.download_file import DownloadFile, DownloadFileInput
from rapidocr.utils.log import logger
from rapidocr.inference_engine.base import FileInfo, InferSession
from rapid_doc.utils.openvino_core import get_openvino_core
from rapid_doc.utils.openvino_static import StaticShapeRunner


_DEFAULT_OCR_MODEL_DIR = None

def _get_ocr_model_dir():
    global _DEFAULT_OCR_MODEL_DIR
    if _DEFAULT_OCR_MODEL_DIR is None:
        try:
            import rapidocr
            _DEFAULT_OCR_MODEL_DIR = Path(rapidocr.__file__).parent / "models"
        except Exception:
            _DEFAULT_OCR_MODEL_DIR = Path(__file__).parent.parent.parent / "resources"
    return _DEFAULT_OCR_MODEL_DIR

class OpenVINOInferSession(InferSession):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.logger = logger

        model_path = cfg.get("model_path", None)
        if model_path is None:
            model_root_dir = cfg.get("model_root_dir", None)
            if model_root_dir is None:
                model_root_dir = str(_get_ocr_model_dir())
            model_info = self.get_model_url(
                FileInfo(
                    engine_type=cfg.engine_type,
                    ocr_version=cfg.ocr_version,
                    task_type=cfg.task_type,
                    lang_type=cfg.lang_type,
                    model_type=cfg.model_type,
                )
            )
            model_path = Path(model_root_dir) / Path(model_info["model_dir"]).name
            download_params = DownloadFileInput(
                file_url=model_info["model_dir"],
                sha256=model_info["SHA256"],
                save_path=model_path,
                logger=self.logger,
            )
            DownloadFile.run(download_params)

        self.logger.info(f"Using {model_path}")
        model_path = Path(model_path)
        self._verify_model(model_path)

        device = cfg.get('device', 'CPU')
        ov_config = self._init_config(cfg)

        self.model_path = Path(model_path)
        self.device = device
        self.core = get_openvino_core()
        # Read once for the input/output names.  Actual inference uses a
        # static-shape model per bucket, compiled by the runner.
        self.model = self.core.read_model(model=str(self.model_path))
        self.input_tensor = self.model.inputs[0]
        self.input_name = self.input_tensor.get_any_name()
        self.output_tensors = self.model.outputs
        task = getattr(cfg.get("task_type"), "value", cfg.get("task_type", "ocr"))
        self.runner = StaticShapeRunner(
            self.core,
            self.model_path,
            device,
            ov_config,
            label=f"ocr-{task}",
        )

    def _init_config(self, cfg: DictConfig) -> Dict[Any, Any]:
        config = {}
        engine_cfg = cfg.get("engine_cfg", {})

        performance_hint = engine_cfg.get("performance_hint", None)
        if performance_hint is not None:
            config["PERFORMANCE_HINT"] = str(performance_hint)

        performance_num_requests = engine_cfg.get("performance_num_requests", -1)
        if performance_num_requests != -1:
            config["PERFORMANCE_HINT_NUM_REQUESTS"] = str(performance_num_requests)

        cache_dir = engine_cfg.get("cache_dir", None)
        if cache_dir is not None:
            config["CACHE_DIR"] = str(cache_dir)

        device = cfg.get('device', 'CPU')
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

        device = cfg.get('device', 'CPU')
        precision = engine_cfg.get("inference_precision") or os.getenv("RAPID_DOC_OCR_PRECISION")
        if precision:
            config["INFERENCE_PRECISION_HINT"] = str(precision)
        if device.upper() == 'GPU':
            # 允许 GPU 分配超过设备默认单块上限（GPU_DEVICE_MAX_ALLOC_MEM_SIZE，
            # 常见 4GB）的 buffer，否则大文档下 batch 输入张量会触发
            # "Exceeded max size of memory object allocation"。
            # 默认开启；如需关闭可设 engine_cfg["enable_large_allocations"] = False。
            enable_large_allocations = engine_cfg.get("enable_large_allocations", False)
            config["GPU_ENABLE_LARGE_ALLOCATIONS"] = bool(enable_large_allocations)

        self.logger.info(f"Using OpenVINO config: {config}")
        return config

    def __call__(self, input_content: np.ndarray) -> Any:
        try:
            return self.runner.run(
                {self.input_name: np.asarray(input_content)}
            )[0]
        except Exception as e:
            error_info = traceback.format_exc()
            raise OpenVIONError(error_info) from e

    def precompile(self, shape) -> None:
        self.runner.precompile(
            {self.input_name: np.zeros(shape, dtype=np.float32)}
        )

    def have_key(self, key: str = "character") -> bool:
        return False

    def get_character_list(self, key: str = "character") -> list:
        return []


class OpenVIONError(Exception):
    pass
