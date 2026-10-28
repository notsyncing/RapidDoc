import numpy as np

from rapid_doc.utils.openvino_shapes import (
    OCR_REC_HEIGHT,
    OCR_REC_MAX_WH_RATIO,
    OCR_REC_WIDTH,
    det_batch_size,
    det_canvas_size,
    pad_batch,
    pad_det_to_canvas,
)


def test_det_canvas_buckets_are_stable():
    assert det_canvas_size(400, 900) == 960
    assert det_canvas_size(100, 100) == 512
    assert det_canvas_size(1300, 200) == 1600
    assert det_canvas_size(5000, 100) == 4096
    assert det_batch_size(512) == 8
    assert det_batch_size(960) == 8
    assert det_batch_size(1600) == 2
    assert det_batch_size(4096) == 1


def test_recognition_width_is_constant():
    assert OCR_REC_WIDTH == int(OCR_REC_HEIGHT * OCR_REC_MAX_WH_RATIO)
    assert OCR_REC_WIDTH == 768


def test_pad_batch_pads_and_reports_real_count():
    images = [1, 2, 3]
    padded, real = pad_batch(images, 8)
    assert real == 3
    assert len(padded) == 8
    assert padded[:3] == images
    padded, real = pad_batch([1, 2], 2)
    assert real == 2
    assert padded == [1, 2]


def test_pad_det_to_canvas_preserves_scale():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    canvas, scale = pad_det_to_canvas(image, 512)
    assert canvas.shape == (512, 512, 3)
    assert scale == 1.0

    large = np.zeros((1200, 600, 3), dtype=np.uint8)
    canvas, scale = pad_det_to_canvas(large, 512)
    assert canvas.shape == (512, 512, 3)
    assert scale == 512 / 1200
