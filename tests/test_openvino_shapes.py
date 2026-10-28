import numpy as np

from rapid_doc.utils.openvino_shapes import (
    OCR_REC_HEIGHT,
    batch_bucket,
    det_canvas_shape,
    det_square_bucket,
    pad_det_to_shape,
    pad_det_to_square,
    padded_batches,
    rec_batch_limit,
    rec_required_width,
    rec_width_bucket,
    unet_bucket_shape,
)


def test_batch_bucket_never_pads_to_the_global_maximum():
    assert batch_bucket(1) == 1
    assert batch_bucket(3) == 4
    assert batch_bucket(16) == 16
    assert batch_bucket(17) == 32
    assert batch_bucket(3, max_batch=12) == 4
    assert batch_bucket(12, max_batch=12) == 12


def test_padded_batches_only_pads_the_remainder():
    batches = list(padded_batches(range(19), max_batch=8))
    assert [(len(c), b, real) for c, b, real in batches] == [
        (8, 8, 8), (8, 8, 8), (4, 4, 3)
    ]
    # A tiny table is not padded to 32.
    batches = list(padded_batches(range(3), max_batch=32))
    assert len(batches) == 1
    chunk, bucket, real = batches[0]
    assert bucket == 4 and len(chunk) == 4 and real == 3


def test_recognition_width_is_bucketed_near_the_real_ratio():
    narrow = np.zeros((20, 120, 3), dtype=np.uint8)  # ratio 6 -> 288 -> 320
    assert rec_required_width([narrow]) == 320
    assert rec_width_bucket(320) == 320
    assert rec_width_bucket(321) == 384
    assert rec_width_bucket(600) == 640
    # Very wide lines are not silently squashed to the old 768 cap.
    assert rec_width_bucket(2529) == 2560
    assert rec_width_bucket(99) == 320
    assert OCR_REC_HEIGHT == 48


def test_recognition_batch_is_bounded_by_output_memory():
    # Narrow lines can use a large batch; very wide lines must use a small
    # one so the [batch, T, vocab] CTC output stays bounded.
    assert rec_batch_limit(320, vocab=18710) >= 16
    assert rec_batch_limit(2560, vocab=18710) <= 8
    assert rec_batch_limit(2560, vocab=18710) < rec_batch_limit(320, vocab=18710)


def test_det_canvas_is_rectangular_and_capped_at_960():
    assert det_canvas_shape(100, 900) == (128, 960)
    assert det_canvas_shape(480, 960) == (512, 960)
    assert det_canvas_shape(100, 100) == (128, 128)
    # Never above the detector's own 960 long-side limit.
    h, w = det_canvas_shape(5000, 100)
    assert max(h, w) <= 960
    assert h == 960


def test_det_square_bucket_uses_small_values():
    assert det_square_bucket(100) == 320
    assert det_square_bucket(400) == 512
    assert det_square_bucket(900) == 960
    assert det_square_bucket(4000) == 960


def test_unet_buckets_only_the_short_side():
    assert unet_bucket_shape(585, 1024) == (640, 1024)
    assert unet_bucket_shape(1024, 585) == (1024, 640)
    assert unet_bucket_shape(1024, 1024) == (1024, 1024)
    assert unet_bucket_shape(20, 1000) == (128, 1024)


def test_pad_det_to_shape_preserves_scale():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    canvas, scale = pad_det_to_shape(image, 128, 256)
    assert canvas.shape == (128, 256, 3)
    assert scale == 1.0

    large = np.zeros((1200, 600, 3), dtype=np.uint8)
    canvas, scale = pad_det_to_shape(large, 512, 512)
    assert canvas.shape == (512, 512, 3)
    assert scale == 512 / 1200

    square, scale = pad_det_to_square(image, 128)
    assert square.shape == (128, 128, 3)
    assert scale == 128 / 200
