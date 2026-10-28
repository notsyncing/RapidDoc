"""Fixed GPU input shapes and buckets for the OpenVINO pipeline.

The Intel GPU plugin compiles oneDNN/OpenCL kernels asynchronously when it
sees a new input shape.  A production server that receives documents with
different text-line lengths and table sizes therefore feeds the plugin a new
shape on almost every request.  The background compiles race with normal
inference and can crash the plugin process (SIGSEGV in
libopenvino_intel_gpu_plugin.so while Python is blocked in
InferRequest::wait()).

Keep every GPU model on a small, deterministic set of shapes and pre-warm the
set during model initialisation.  This module is the single source of truth
for those shapes; both the pipeline code and the warm-up use it.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

# Layout / table models use fixed spatial sizes already.
LAYOUT_SIZE = 800
LAYOUT_BATCH = int(os.getenv("RAPID_DOC_LAYOUT_BATCH_SIZE", "1") or 1)
TABLE_CLS_SIZE = 224
TABLE_CLS_BATCH = int(os.getenv("RAPID_DOC_TABLE_CLS_BATCH_SIZE", "32") or 32)
TABLE_STRUCT_BATCH = int(os.getenv("RAPID_DOC_TABLE_STRUCTURE_BATCH_SIZE", "16") or 16)
UNET_SIZE = 1024
SLANET_SIZE = 488

# OCR recognition: RapidOCR pads every crop to rec_img_shape (3, 48, W).
# A fixed aspect-ratio cap makes W constant.  The batch is padded to the
# configured size so the batch dimension is constant as well.
OCR_REC_HEIGHT = 48
OCR_REC_MAX_WH_RATIO = float(os.getenv("RAPID_DOC_OCR_MAX_WH_RATIO", "16"))
OCR_REC_WIDTH = int(OCR_REC_HEIGHT * OCR_REC_MAX_WH_RATIO)
OCR_REC_BATCH = int(os.getenv("RAPID_DOC_OCR_REC_BATCH_SIZE", "16"))

# OCR detection accepts dynamic H/W.  Square canvases keep the number of
# distinct shapes small while never shrinking a crop that fits in the canvas.
DET_CANVAS_SIZES = (512, 768, 960, 1280, 1600, 2048, 2560, 3072, 4096)
DET_BATCH_BY_CANVAS = {
    512: 8,
    768: 8,
    960: 8,
    1280: 4,
    1600: 2,
    2048: 2,
    2560: 1,
    3072: 1,
    4096: 1,
}


def det_canvas_size(height: int, width: int) -> int:
    """Return the smallest square canvas that contains ``height x width``."""
    longest = max(int(height), int(width), 1)
    for size in DET_CANVAS_SIZES:
        if longest <= size:
            return size
    # Extremely large crops: keep one deterministic final bucket.
    return DET_CANVAS_SIZES[-1]


def pad_det_to_canvas(img: np.ndarray, canvas: int):
    """Pad/resize a detection image onto a fixed square canvas.

    Returns ``(canvas_image, scale)``.  Boxes from the detector must be
    divided by ``scale`` to map back to the original image.
    """
    h, w = img.shape[:2]
    scale = 1.0
    if h > canvas or w > canvas:
        scale = min(canvas / float(h), canvas / float(w))
        nh = max(1, min(canvas, int(round(h * scale))))
        nw = max(1, min(canvas, int(round(w * scale))))
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        h, w = nh, nw
    padded = np.ones((canvas, canvas, 3), dtype=np.uint8) * 255
    padded[:h, :w] = img
    return padded, scale


def det_model_size(canvas: int) -> int:
    """Shape the DB detector actually receives after RapidOCR preprocess.

    RapidOCR selects ``limit_side_len`` from the input's longest side and
    rounds to a multiple of 32.  Canvas buckets above 2000 collapse to 1984;
    smaller buckets are used as-is (all buckets are multiples of 64).
    """
    if canvas >= 2048:
        return 1984
    return canvas


def det_batch_size(canvas: int) -> int:
    """Deterministic batch size for a detection canvas.

    A fixed value per canvas means the (batch, height, width) tuple is stable
    for every request instead of following the number of crops on the page.
    """
    if canvas <= 512:
        return DET_BATCH_BY_CANVAS[512]
    if canvas <= 768:
        return DET_BATCH_BY_CANVAS[768]
    if canvas <= 960:
        return DET_BATCH_BY_CANVAS[960]
    if canvas <= 1280:
        return DET_BATCH_BY_CANVAS[1280]
    if canvas <= 1600:
        return DET_BATCH_BY_CANVAS[1600]
    if canvas <= 2048:
        return DET_BATCH_BY_CANVAS[2048]
    if canvas <= 2560:
        return DET_BATCH_BY_CANVAS[2560]
    if canvas <= 3072:
        return DET_BATCH_BY_CANVAS[3072]
    return DET_BATCH_BY_CANVAS[4096]


def pad_batch(images: list, batch: int, filler=None) -> tuple[list, int]:
    """Pad ``images`` to a multiple of ``batch`` and return real count."""
    real = len(images)
    if real == 0:
        return images, 0
    if filler is None:
        filler = images[-1]
    missing = (-real) % batch
    if missing:
        images = list(images) + [filler] * missing
    return images, real
