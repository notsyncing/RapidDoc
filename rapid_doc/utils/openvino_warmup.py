"""Synchronous GPU warm-up for the shape-dynamic GPU models.

OpenVINO's Intel GPU plugin compiles kernels asynchronously the first time a
shape is inferred.  Letting that happen in the middle of a stream of requests
is what races with later inference and crashes the plugin worker.  After the
pipeline has fixed its shapes (see ``openvino_shapes``), running one dummy
request for each shape at model-initialisation time moves compilation into a
single serial phase.
"""

from __future__ import annotations

import time

import numpy as np
from loguru import logger

from .openvino_shapes import (
    OCR_REC_BATCH,
    OCR_REC_HEIGHT,
    OCR_REC_WIDTH,
)


_WARMED: set[tuple[int, tuple[int, ...], str]] = set()


def _session_of(obj, *names):
    for name in names:
        obj = getattr(obj, name, None)
        if obj is None:
            return None
    return obj


def _run(session, tag: str, shape: tuple[int, ...], scale_factor=None):
    if session is None:
        return
    key = (id(session), shape, tag)
    if key in _WARMED:
        return
    started = time.perf_counter()
    try:
        data = np.zeros(shape, dtype=np.float32)
        if scale_factor is not None:
            session(data, np.asarray(scale_factor, dtype=np.float32))
        else:
            session(data)
        _WARMED.add(key)
        logger.info(
            "OpenVINO GPU warm-up {} {} {:.2f}s",
            tag,
            shape,
            time.perf_counter() - started,
        )
    except Exception as exc:
        # Do not silently treat this as success: the shape is not marked
        # warmed, so the first real request will still exercise it.  A failure
        # here usually means the model/device configuration is wrong.
        logger.warning(f"OpenVINO GPU warm-up failed for {tag} {shape}: {exc}")


def warmup_openvino_gpu_pipeline(model) -> None:
    """Warm all fixed GPU shapes used by one MineruPipelineModel."""
    ocr_model = getattr(model, "ocr_model", None)
    det_session = _session_of(ocr_model, "text_detector", "session")
    rec_session = _session_of(ocr_model, "text_recognizer", "session")
    # These are the detection shapes actually used by page/table/refine
    # crops in production.  Larger canvas buckets still have a fixed shape,
    # but are compiled on first use instead of lengthening every restart.
    det_shapes = {
        (1, 3, 512, 512),
        (8, 3, 512, 512),
        (1, 3, 768, 768),
        (8, 3, 768, 768),
        (1, 3, 960, 960),
        (2, 3, 960, 960),
        (4, 3, 960, 960),
        (8, 3, 960, 960),
    }
    for shape in sorted(det_shapes):
        _run(det_session, "ocr-det", shape)
    _run(
        rec_session,
        "ocr-rec",
        (OCR_REC_BATCH, 3, OCR_REC_HEIGHT, OCR_REC_WIDTH),
    )

