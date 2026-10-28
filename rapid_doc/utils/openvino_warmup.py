"""Optional static-shape pre-compilation for OpenVINO GPU.

Static bucket models remove the runtime shape-compile race, so a synchronous
warm-up is no longer required for correctness.  The old warm-up ran real
inference on dynamic shapes and cost 40-50 s on every process start; it is now
opt-in via ``RAPID_DOC_OPENVINO_PRECOMPILE=1`` and only compiles a few
likely buckets.

The first request otherwise compiles the buckets it actually uses (about
2-3 s each for detection and 1-2 s for recognition) while holding the global
GPU lock, then reuses them for the rest of the process lifetime.
"""

from __future__ import annotations

import os
import time

from loguru import logger

from .openvino_shapes import OCR_REC_HEIGHT


def _precompile(session, shapes, tag: str) -> None:
    precompile = getattr(session, "precompile", None)
    if precompile is None:
        return
    for shape in shapes:
        started = time.perf_counter()
        try:
            precompile(shape)
        except Exception as exc:
            # Surface the failure: a precompile error means the model/device
            # configuration is wrong, and the first request would fail too.
            logger.warning(f"OpenVINO GPU precompile {tag} {shape} failed: {exc}")
            continue
        logger.info(
            "OpenVINO GPU precompile {} {} {:.2f}s",
            tag,
            shape,
            time.perf_counter() - started,
        )


def warmup_openvino_gpu_pipeline(model) -> None:
    """Pre-compile a few common OCR buckets (opt-in, not correctness)."""
    if os.getenv("RAPID_DOC_OPENVINO_PRECOMPILE", "0") != "1":
        return
    ocr_model = getattr(model, "ocr_model", None)
    if ocr_model is None:
        return
    det_session = getattr(getattr(ocr_model, "text_detector", None), "session", None)
    rec_session = getattr(getattr(ocr_model, "text_recognizer", None), "session", None)

    # The detector's long side is capped at 960 by RapidOCR.  Rectangular
    # 960x512 covers the common text-block shape; 512x512 covers small crops.
    _precompile(
        det_session,
        [(8, 3, 960, 512), (8, 3, 960, 960), (1, 3, 512, 512)],
        "ocr-det",
    )
    # Recognition widths are bucketed; 512/768 cover ordinary body text and
    # wide table rows.  The batch cap follows the configured bucket ladder.
    _precompile(
        rec_session,
        [
            (16, 3, OCR_REC_HEIGHT, 512),
            (16, 3, OCR_REC_HEIGHT, 768),
        ],
        "ocr-rec",
    )
