"""The table models require a [boxes, texts, scores] contract per image.

A table crop whose OCR found nothing used to arrive as ``[]``.  The batched
inference path passed that straight to ``RapidTable`` and
``format_ocr_results`` indexed the missing elements, aborting the whole
batch with ``IndexError`` (production /file_parse 500).  These tests pin the
normalisation instead of relying on a try/except around the batch.
"""
from types import SimpleNamespace

import numpy as np

from rapid_doc.backend.pipeline.analyze_utils import _run_batched_table_inference
from rapid_doc.model.table.rapid_table_self.utils import (
    format_ocr_results,
    normalize_ocr_result,
)

BOXES = [[[10, 10], [200, 10], [200, 40], [10, 40]]]


class _FakeStructureModel:
    cfg = SimpleNamespace(model_type="SLANETPLUS")

    def __init__(self):
        self.calls = []

    def __call__(self, imgs, ocrs, batch_size=1):
        self.calls.append((len(imgs), batch_size, list(ocrs)))
        return SimpleNamespace(pred_htmls=["<table><tr><td>x</td></tr></table>"] * len(imgs))


class _FakeTableModel:
    def __init__(self):
        self.wired_table_model = _FakeStructureModel()
        self.wireless_table_model = _FakeStructureModel()

    def predict(self, **kwargs):
        return None


def test_normalize_ocr_result_accepts_every_producer_shape():
    assert normalize_ocr_result(None) == ([], [], [])
    assert normalize_ocr_result([]) == ([], [], [])
    assert normalize_ocr_result((BOXES, ["a"])) == (BOXES, ["a"], [])
    assert normalize_ocr_result((BOXES, ["a"], [0.5])) == (BOXES, ["a"], [0.5])

    rapidocr_like = SimpleNamespace(boxes=BOXES, txts=["a"], scores=[0.5])
    assert normalize_ocr_result(rapidocr_like) == (BOXES, ["a"], [0.5])

    mapping = {"boxes": BOXES, "texts": ["a"], "scores": [0.5]}
    assert normalize_ocr_result(mapping) == (BOXES, ["a"], [0.5])


def test_format_ocr_results_handles_empty_and_two_element_results():
    boxes, rec = format_ocr_results([], 100, 100)
    assert boxes.shape == (0, 4)
    assert rec == []

    boxes, rec = format_ocr_results(None, 100, 100)
    assert boxes.shape == (0, 4)
    assert rec == []

    boxes, rec = format_ocr_results((BOXES, ["hi"]), 100, 100)
    assert boxes.shape == (1, 4)
    assert rec == [("hi", 1.0)]

    boxes, rec = format_ocr_results((BOXES, ["hi"], [0.9]), 100, 100)
    assert rec == [("hi", 0.9)]


def test_empty_ocr_table_keeps_the_static_slane_batch():
    empty_model = _FakeTableModel()
    entries = [
        ({"table_res": {}}, empty_model, {"image": np.zeros((40, 200, 3), np.uint8),
                                          "ocr_result": (BOXES, ["a"], [0.9])}),
        ({"table_res": {}}, empty_model, {"image": np.zeros((40, 200, 3), np.uint8),
                                          "ocr_result": []}),
    ]

    _run_batched_table_inference({"model_type": "UNET_SLANET_PLUS"}, entries)

    # One padded SLANE batch, not a per-table fallback.
    wired_calls = empty_model.wired_table_model.calls
    assert len(wired_calls) == 1
    batch_imgs, batch_size, batch_ocrs = wired_calls[0]
    assert batch_imgs == 16 and batch_size == 16
    assert all(len(ocr) == 3 for ocr in batch_ocrs)
    assert batch_ocrs[1] == ([], [], [])
    # The normalised contract is written back for fallback/retry consumers.
    assert entries[1][2]["ocr_result"] == ([], [], [])
