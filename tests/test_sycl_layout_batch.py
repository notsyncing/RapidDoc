"""Batch invariance of the sycl-infer layout detector.

The in-process ONNX executor used to peel the output coordinates of an int64
``Transpose`` in column-major order.  That is an identity at batch=1 and
duplicates/interleaves elements at batch>1, so a batched layout run attached
each page's ``orders`` values to the wrong detections - the reading order was
scrambled while the first six box columns stayed correct.  The executor is
fixed; this test is the regression.

It runs the raw detector at batch=1 and at batch=N for the same N pages and
requires every page's detections and reading-order values to match.  A pair of
near-tied detections may be emitted in either order (the convolution's fp
accumulation differs by ~1e-5 between batch shapes, and the model's TopK breaks
the tie differently), so the comparison is a set comparison over the geometry
and the order *values*; genuine page mixing changes both the set and the
values.

    pytest -q tests/test_sycl_layout_batch.py
    RAPID_DOC_LAYOUT_TEST_PDF=... RAPID_DOC_LAYOUT_TEST_BATCH=16 \
        pytest -q tests/test_sycl_layout_batch.py

Skipped when the sycl-infer library or the layout model is unavailable.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

try:
    import pytest
except ImportError:  # plain-script use (the benchmark venv has no pytest)
    pytest = None


def _skip(message):
    if pytest is not None:
        pytest.skip(message)
    raise SystemExit(f"SKIP: {message}")


try:
    import pypdfium2
except ImportError:
    _skip("pypdfium2 is not installed")

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
PDF = Path(os.getenv(
    "RAPID_DOC_LAYOUT_TEST_PDF",
    REPO / "error_pdfs" / "600348_2016_semi_summary_bf25118c.pdf"))
BATCH = int(os.getenv("RAPID_DOC_LAYOUT_TEST_BATCH", "16"))


def _model():
    if not os.getenv("SYCL_INFER_ONNX_LIB"):
        _skip("SYCL_INFER_ONNX_LIB is not set (build sycl_infer_onnx_c)")
    if not PDF.exists():
        _skip(f"no test PDF at {PDF}")
    from rapid_doc.model.layout.rapid_layout import RapidLayoutModel
    from rapid_doc.model.layout.rapid_layout_self import EngineType
    try:
        m = RapidLayoutModel({"engine_type": EngineType.SYCL_INFER,
                              "engine_cfg": {"device": os.getenv("SYCL_INFER_ONNX_DEVICE", "GPU.0")}})
    except Exception as e:  # no library / model / device
        _skip(f"sycl-infer layout session unavailable: {e}")
    return m


def _canonical(boxes, nums, masks):
    pages = []
    off = 0
    for k in nums:
        b = boxes[off:off + k]
        mk = masks[off:off + k]
        off += k
        # Match rows by geometry (not by their position in the batch output)
        # and compare the reading-order values as a multiset: a near-tie may
        # swap two rows, but page mixing changes the boxes themselves.
        geom = b[:, :6]
        order = np.lexsort((geom[:, 4], geom[:, 3], geom[:, 2], geom[:, 1], geom[:, 0]))
        pages.append((geom[order], np.sort(b[:, 6]), mk[order]))
    return pages


def _run(m, images):
    h = m.model.model_handler.model_processors
    sess = m.model.session
    pre = [h.preprocess(im) for im in images]
    x = np.concatenate(pre, 0)
    sf = np.array([[800 / im.shape[0], 800 / im.shape[1]] for im in images], np.float32)
    out = sess.run({"image": x, "scale_factor": sf,
                    "im_shape": np.array([[800, 800]], np.float32)})
    assert out[0].shape[0] == 300 * len(images)
    return _canonical(out[0], out[1], out[2])


def test_layout_batch_matches_single_page_order():
    m = _model()
    pdf = pypdfium2.PdfDocument(str(PDF))
    n = min(BATCH, len(pdf))
    images = [pdf[i].render(scale=200 / 72).to_numpy() for i in range(n)]
    ref = [_run(m, [im])[0] for im in images]
    got = _run(m, images)
    assert len(got) == n
    for i, ((g, go, gm), (rg, ro, rmk)) in enumerate(zip(got, ref)):
        assert g.shape == rg.shape, f"page {i}: {g.shape} != {rg.shape}"
        assert np.abs(g - rg).max() <= 0.5, f"page {i}: geometry changed by batch"
        assert np.array_equal(go, ro), f"page {i}: reading-order values changed by batch"
        assert np.count_nonzero(gm != rmk) <= 8, f"page {i}: masks changed by batch"


def test_layout_batch_predict_returns_each_page_once():
    """The public path: batch_predict must not mix pages or reorder them."""
    m = _model()
    pdf = pypdfium2.PdfDocument(str(PDF))
    n = min(BATCH, len(pdf))
    images = [pdf[i].render(scale=200 / 72).to_numpy() for i in range(n)]
    one = m.batch_predict(images, batch_size=1)
    many = m.batch_predict(images, batch_size=n)
    assert len(many) == len(one) == n
    for i, (a, b) in enumerate(zip(one, many)):
        assert len(a) == len(b), f"page {i}: box count changed"
        for j, (x, y) in enumerate(zip(a, b)):
            assert x["original_label"] == y["original_label"], f"page {i} box {j}: label changed"
            assert x["original_order"] == y["original_order"], f"page {i} box {j}: order changed"
            assert np.abs(np.asarray(x["poly"], np.float64) -
                          np.asarray(y["poly"], np.float64)).max() <= 0.5, \
                f"page {i} box {j}: coordinates changed"


if __name__ == "__main__":
    test_layout_batch_matches_single_page_order()
    test_layout_batch_predict_returns_each_page_once()
    print("layout batch regression OK")
