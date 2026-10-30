"""Shape buckets for the OpenVINO GPU models.

The Intel GPU plugin compiles oneDNN/OpenCL kernels asynchronously when a
model is executed with a shape it has not seen before.  A production server
sees a new OCR-recognition width or table batch on almost every document, so
background compilation repeatedly races with the next inference and can
crash the plugin worker (SIGSEGV in libopenvino_intel_gpu_plugin.so while
Python is blocked in ``ov::InferRequest::wait()``).

The fix is not to freeze every model at its maximum size.  Instead each
dynamic dimension is snapped to a small set of buckets that are close to the
real input size, and every (model, bucket) pair is compiled once as a static
OpenVINO model (see ``openvino_static``).  Static compiled models never start
a background shape compile, and bucketing keeps padding within one bucket
step instead of always paying for the maximum.

This module is the single source of truth for those buckets.
"""

from __future__ import annotations

import math
import os

import cv2
import numpy as np


def _env_positive_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


# ---------------------------------------------------------------------------
# Batch buckets
# ---------------------------------------------------------------------------
# One shared ladder.  ``batch_bucket`` never returns a value larger than the
# configured cap, so a page with three tables is not padded to 32.
BATCH_BUCKETS = (1, 2, 4, 8, 16, 32, 64)
DEFAULT_MAX_BATCH = 64


def batch_bucket(size: int, max_batch: int | None = None,
                 buckets: tuple[int, ...] = BATCH_BUCKETS) -> int:
    """Smallest bucket >= ``size`` (capped at ``max_batch``)."""
    if size <= 0:
        return 0
    cap = len(buckets)
    if max_batch is not None and max_batch > 0:
        available = [b for b in buckets if b <= max_batch]
        if not available:
            # The configured cap is below the smallest ladder entry.
            return int(max_batch)
        for bucket in available:
            if bucket >= size:
                return int(bucket)
        # A cap such as 12 is not on the power-of-two ladder; use it as the
        # final bucket instead of returning a value smaller than ``size``.
        return int(max_batch)
    for bucket in buckets[:cap]:
        if bucket >= size:
            return int(bucket)
    return int(buckets[cap - 1])


def padded_batches(items, max_batch: int | None = None,
                   buckets: tuple[int, ...] = BATCH_BUCKETS,
                   filler=None):
    """Yield ``(chunk, bucket, real_count)`` with bucket-sized chunks.

    Full chunks use ``max_batch``; the remainder is padded only to the next
    bucket.  A sequence of 84 items with ``max_batch=16`` therefore yields
    five 16-item chunks and one 4-item chunk, never a 16-item chunk padded
    from four real items.
    """
    items = list(items)
    total = len(items)
    if not total:
        return
    cap = max_batch or DEFAULT_MAX_BATCH
    start = 0
    while start < total:
        remaining = total - start
        size = cap if remaining > cap else remaining
        bucket = batch_bucket(size, cap, buckets)
        chunk = items[start:start + size]
        if bucket > len(chunk):
            fill = chunk[-1] if filler is None else filler
            chunk = list(chunk) + [fill] * (bucket - len(chunk))
        yield chunk, bucket, size
        start += size


# ---------------------------------------------------------------------------
# OCR recognition: (batch, 3, 48, width)
# ---------------------------------------------------------------------------
OCR_REC_HEIGHT = 48
# Widths are ~ aspect_ratio * 48.  The list covers the observed 320..2529
# range with <= ~12% padding at each step.  It deliberately starts at 320,
# which is the recognition model's default width.
REC_WIDTH_BUCKETS = (
    320, 384, 448, 512, 576, 640, 704, 768,
    896, 1024, 1152, 1280, 1536, 1792, 2048, 2304, 2560,
)
# RapidOCR's default recognition canvas.  Kept for compatibility with the
# text recogniser's built-in parameter alignment.
OCR_REC_DEFAULT_WIDTH = 320
OCR_REC_MAX_BATCH = _env_positive_int("RAPID_DOC_OCR_REC_BATCH_SIZE", 64)
# The fixed-shape recogniser path (rapid_ocr.py) pads every batch to
# OCR_REC_BATCH and every crop to OCR_REC_MAX_WH_RATIO.  The symbols were
# removed when the bucketed path landed but are still imported by the tuned
# recogniser, so restore them here with the image's measured defaults.
OCR_REC_BATCH = _env_positive_int("RAPID_DOC_OCR_REC_BATCH_SIZE", 128)
try:
    OCR_REC_MAX_WH_RATIO = float(os.getenv("RAPID_DOC_OCR_MAX_WH_RATIO", "8"))
except (TypeError, ValueError):
    OCR_REC_MAX_WH_RATIO = 8.0
# The CTC output is [batch, width/4, vocab] float32 (~18.7k classes for
# PP-OCRv6).  A batch of 64 at width 2560 would materialise ~3 GB, and the
# runner copies it once more for the caller.  Limit the batch by the output
# budget; narrow lines still use large batches, wide lines use smaller ones.
REC_OUTPUT_BUDGET_MB = _env_positive_int(
    "RAPID_DOC_OCR_REC_OUTPUT_BUDGET_MB", 512
)


def rec_width_bucket(width: float) -> int:
    """Smallest recognition width bucket >= ``width``."""
    required = max(float(width), float(OCR_REC_DEFAULT_WIDTH))
    for bucket in REC_WIDTH_BUCKETS:
        if required <= bucket + 1e-6:
            return int(bucket)
    return int(REC_WIDTH_BUCKETS[-1])


def rec_required_width(images) -> int:
    """Recognition width needed by a batch, without any bucket padding."""
    required = OCR_REC_DEFAULT_WIDTH
    for img in images:
        h, w = img.shape[:2]
        if h <= 0:
            continue
        required = max(required, int(math.ceil(w * OCR_REC_HEIGHT / float(h))))
    return required


def rec_batch_limit(width: int, vocab: int = 20000,
                    budget_mb: int | None = None) -> int:
    """Largest batch bucket whose CTC output fits the memory budget."""
    timesteps = max(1, int(width) // 4)
    per_image = timesteps * max(int(vocab), 1) * 4
    budget = (budget_mb or REC_OUTPUT_BUDGET_MB) * 1024 * 1024
    # The OpenVINO output copy roughly doubles the live tensor.
    allowed = max(1, int(budget // max(1, per_image * 2)))
    best = 1
    for bucket in BATCH_BUCKETS:
        if bucket <= allowed:
            best = bucket
    return best


def rec_batch_shape(real_images) -> tuple[int, int]:
    """Return ``(batch_bucket, width_bucket)`` for a recognition batch."""
    count = len(real_images)
    return batch_bucket(count, OCR_REC_MAX_BATCH), rec_width_bucket(
        rec_required_width(real_images)
    )


def pad_rec_batch(images, batch: int, width: int) -> list[np.ndarray]:
    """Pad an image list and each normalised crop to ``(batch, 48, width)``.

    Crops are padded with zeros, exactly like RapidOCR's ``resize_norm_img``
    pads a short line to its batch width, so recognition scores are unchanged.
    """
    images = list(images)
    if not images:
        return images
    if len(images) < batch:
        images = images + [images[-1]] * (batch - len(images))
    return images


# ---------------------------------------------------------------------------
# OCR detection: (batch, 3, H, W)
# ---------------------------------------------------------------------------
# RapidOCR caps the detector's long side at 960 (rounding to 32).  Never
# choose a canvas above that: the detector would immediately downscale it,
# wasting bandwidth and changing the mapping for no benefit.
DET_MAX_SIDE = 960
DET_STRIDE = 128
# Small square canvases used by the single-image table/full-OCR paths.
DET_SQUARE_BUCKETS = (320, 384, 512, 640, 768, 896, 960)
DET_MAX_BATCH = 8


def det_canvas_shape(height: int, width: int) -> tuple[int, int]:
    """Rectangular document canvas covering ``height x width``.

    Content larger than 960 is scaled down first (the detector does the same
    internally), then each side is rounded up to a 128-multiple.  A 960x480
    document therefore runs at 960x512 instead of the old fixed 960x960.
    """
    height = max(int(height), 1)
    width = max(int(width), 1)
    longest = max(height, width)
    scale = min(1.0, float(DET_MAX_SIDE) / float(longest))
    scaled_h = max(1, int(round(height * scale)))
    scaled_w = max(1, int(round(width * scale)))
    canvas_h = min(DET_MAX_SIDE, int(math.ceil(scaled_h / DET_STRIDE) * DET_STRIDE))
    canvas_w = min(DET_MAX_SIDE, int(math.ceil(scaled_w / DET_STRIDE) * DET_STRIDE))
    return max(DET_STRIDE, canvas_h), max(DET_STRIDE, canvas_w)


def det_square_bucket(max_side: int) -> int:
    """Smallest square detector canvas >= ``max_side`` (capped at 960)."""
    required = min(int(max_side), DET_MAX_SIDE)
    for bucket in DET_SQUARE_BUCKETS:
        if required <= bucket:
            return int(bucket)
    return int(DET_SQUARE_BUCKETS[-1])


def pad_det_to_shape(img: np.ndarray, height: int, width: int):
    """Fit ``img`` inside ``height x width`` and pad the rest with white.

    Returns ``(canvas, scale)``.  Boxes must be divided by ``scale`` to map
    back to the original image.
    """
    h, w = img.shape[:2]
    scale = 1.0
    if h > height or w > width:
        scale = min(height / float(h), width / float(w))
        nh = max(1, min(height, int(round(h * scale))))
        nw = max(1, min(width, int(round(w * scale))))
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        h, w = nh, nw
    padded = np.ones((height, width, 3), dtype=np.uint8) * 255
    padded[:h, :w] = img
    return padded, scale


def pad_det_to_square(img: np.ndarray, canvas: int):
    """Compatibility helper: pad to a square detector canvas."""
    return pad_det_to_shape(img, canvas, canvas)


def det_canvas_size(height: int, width: int) -> int:
    """Square detector canvas for the fixed-shape recognition path."""
    return det_square_bucket(max(int(height), int(width)))


def pad_det_to_canvas(img: np.ndarray, canvas: int):
    """Square detection canvas, named for the fixed-shape recogniser path."""
    return pad_det_to_square(img, canvas)


# ---------------------------------------------------------------------------
# Table models
# ---------------------------------------------------------------------------
TABLE_CLS_SIZE = 224
TABLE_CLS_MAX_BATCH = _env_positive_int("RAPID_DOC_TABLE_CLS_BATCH_SIZE", 32)
TABLE_STRUCT_MAX_BATCH = _env_positive_int(
    "RAPID_DOC_TABLE_STRUCTURE_BATCH_SIZE", 16
)
SLANET_SIZE = 488

# UNET keeps its long side at 1024 (the model's trained scale) and buckets
# only the shorter side.  A 1024x585 table runs at 1024x640, not 1024x1024.
UNET_SIZE = 1024
UNET_STRIDE = 128


def unet_bucket_shape(height: int, width: int) -> tuple[int, int]:
    """Smallest UNET shape >= ``height x width`` keeping the long side 1024."""
    height = max(int(height), 1)
    width = max(int(width), 1)
    if width >= height:
        bucket_w = UNET_SIZE
        bucket_h = min(
            UNET_SIZE, int(math.ceil(height / float(UNET_STRIDE)) * UNET_STRIDE)
        )
    else:
        bucket_h = UNET_SIZE
        bucket_w = min(
            UNET_SIZE, int(math.ceil(width / float(UNET_STRIDE)) * UNET_STRIDE)
        )
    return max(UNET_STRIDE, bucket_h), max(UNET_STRIDE, bucket_w)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
LAYOUT_SIZE = 800
