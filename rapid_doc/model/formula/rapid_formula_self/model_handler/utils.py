import os
from pathlib import Path

from ..utils.download_file import DownloadFile, DownloadFileInput
from ..utils.logger import Logger
from ..utils.typings import ModelType, EngineType
from ..utils.utils import mkdir, read_yaml


class ModelProcessor:
    logger = Logger(logger_name=__name__).get_log()

    cur_dir = Path(__file__).resolve().parent
    root_dir = cur_dir.parent
    DEFAULT_MODEL_PATH = root_dir / "configs" / "default_models.yaml"

    DEFAULT_MODEL_DIR = Path(os.getenv('RAPID_MODELS_DIR', root_dir / "models"))
    mkdir(DEFAULT_MODEL_DIR)

    model_map = read_yaml(DEFAULT_MODEL_PATH)

    @classmethod
    def get_model_path(cls, model_type: ModelType, engine_type: EngineType) -> str:
        return cls.get_single_model_path(model_type, engine_type)

    @classmethod
    def get_single_model_path(cls, model_type: ModelType, engine_type: EngineType) -> str:
        # An in-process engine (sycl-infer) uses the ONNX file; it has no
        # entry of its own in the download table.
        key = engine_type.value
        if key not in cls.model_map:
            key = "onnxruntime"
        model_info = cls.model_map[key][model_type.value]
        save_model_path = (
            cls.DEFAULT_MODEL_DIR / Path(model_info["model_dir_or_path"]).name
        )
        download_params = DownloadFileInput(
            file_url=model_info["model_dir_or_path"],
            sha256=model_info["SHA256"],
            save_path=save_model_path,
            logger=cls.logger,
        )
        DownloadFile.run(download_params)

        return str(save_model_path)

    @classmethod
    def get_character_path(cls, model_type: ModelType, engine_type: EngineType) -> str:
        key = engine_type.value
        if key not in cls.model_map:
            key = "onnxruntime"
        model_info = cls.model_map[key][model_type.value]
        dict_download_url = model_info.get("dict_url")
        save_model_path = (
            cls.DEFAULT_MODEL_DIR / Path(dict_download_url).name
        )

        download_params = DownloadFileInput(
            file_url=dict_download_url,
            save_path=save_model_path,
            logger=cls.logger,
        )
        DownloadFile.run(download_params)

        return str(save_model_path)
