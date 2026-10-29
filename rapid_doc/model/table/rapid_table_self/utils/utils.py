# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import hashlib
import importlib
from pathlib import Path
from typing import List, Tuple, Union
from urllib.parse import urlparse

import cv2
import numpy as np
from omegaconf import DictConfig, OmegaConf


def normalize_ocr_result(ocr_results) -> Tuple[list, list, list]:
    """Coerce any supported OCR payload to ``[boxes, texts, scores]``.

    Table crops with no detectable text reach this code as ``[]``.  Callers
    may also pass ``None``, a 2-tuple of boxes/texts, the raw tuple produced by
    RapidOCR, or a RapidOCR result object.  Every form has to normalise to the
    same 3-list contract so an empty table cannot abort a whole inference
    batch (and so ``format_ocr_results`` never indexes a missing element).
    """
    if ocr_results is None:
        return [], [], []

    if isinstance(ocr_results, dict):
        boxes = ocr_results.get("boxes") or ocr_results.get("dt_boxes") or []
        texts = (
            ocr_results.get("txts")
            or ocr_results.get("texts")
            or ocr_results.get("rec_texts")
            or []
        )
        scores = (
            ocr_results.get("scores")
            or ocr_results.get("rec_scores")
            or []
        )
        return _as_list(boxes), _as_list(texts), _as_list(scores)

    # RapidOCR result objects expose boxes/txts/scores attributes.
    if not isinstance(ocr_results, (list, tuple, np.ndarray)) and hasattr(
        ocr_results, "boxes"
    ):
        boxes = getattr(ocr_results, "boxes", None)
        texts = getattr(ocr_results, "txts", None)
        scores = getattr(ocr_results, "scores", None)
        return _as_list(boxes), _as_list(texts), _as_list(scores)

    if isinstance(ocr_results, np.ndarray):
        # A bare array can only carry boxes; text lives in the sibling lists.
        return ocr_results.tolist(), [], []

    if not isinstance(ocr_results, (list, tuple)):
        return [], [], []

    if len(ocr_results) == 0:
        return [], [], []
    if len(ocr_results) == 1:
        value = ocr_results[0]
        try:
            value_array = np.asarray(value)
        except (TypeError, ValueError):
            value_array = None
        # A single box is a (4, 2) array/list.  A list containing boxes is
        # (N, 4, 2); keep that nested form so format_ocr_results can infer the
        # number of boxes correctly.
        if value_array is not None and value_array.ndim == 2 and value_array.shape[-1] == 2:
            return [value], [], []
        return _as_list(value), [], []
    if len(ocr_results) == 2:
        return _as_list(ocr_results[0]), _as_list(ocr_results[1]), []
    return (
        _as_list(ocr_results[0]),
        _as_list(ocr_results[1]),
        _as_list(ocr_results[2]),
    )


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, str):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def format_ocr_results(
    ocr_results: Tuple[np.ndarray, Tuple[str], Tuple[float]], img_h: int, img_w: int
) -> Tuple[np.ndarray, List[Tuple[str, float]]]:
    box_list, text_list, score_list = normalize_ocr_result(ocr_results)

    if score_list:
        rec_res = list(zip(text_list, score_list))
    else:
        # A 2-tuple carries text without scores; the matcher only needs text.
        rec_res = [(text, 1.0) for text in text_list]

    try:
        bboxes = np.array(box_list, dtype=np.float32)
    except (TypeError, ValueError):
        return np.empty((0, 4), dtype=np.float32), []
    if bboxes.size == 0:
        return np.empty((0, 4), dtype=np.float32), []
    # A single box may arrive as (4, 2); add the box axis.
    if bboxes.ndim == 2 and bboxes.shape[-1] == 2:
        bboxes = bboxes[None, ...]

    min_coords = bboxes[..., :2].min(axis=1)
    max_coords = bboxes[..., :2].max(axis=1)

    min_coords = np.maximum(min_coords, 0)
    max_coords = np.minimum(max_coords, [img_w, img_h])
    dt_boxes = np.hstack([min_coords, max_coords])
    return dt_boxes, rec_res


def save_img(save_path: Union[str, Path], img: np.ndarray):
    cv2.imwrite(str(save_path), img)


def save_txt(save_path: Union[str, Path], txt: str):
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(txt)


def import_package(name, package=None):
    try:
        module = importlib.import_module(name, package=package)
        return module
    except ModuleNotFoundError:
        return None


def mkdir(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def read_yaml(file_path: Union[str, Path]) -> DictConfig:
    return OmegaConf.load(file_path)


def get_file_sha256(file_path: Union[str, Path], chunk_size: int = 65536) -> str:
    with open(file_path, "rb") as file:
        sha_signature = hashlib.sha256()
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            sha_signature.update(chunk)

    return sha_signature.hexdigest()


def is_url(url: str) -> bool:
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False
