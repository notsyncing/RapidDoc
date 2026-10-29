import numpy as np

from rapid_doc.model.table.rapid_table_self.utils import normalize_ocr_result
from rapid_doc.model.table.rapid_table_self.utils.utils import format_ocr_results


def test_empty_ocr_normalizes_to_three_lists():
    assert normalize_ocr_result(None) == ([], [], [])
    assert normalize_ocr_result([]) == ([], [], [])


def test_two_item_ocr_result_gets_scores():
    boxes = np.array([[[0, 0], [10, 0], [10, 5], [0, 5]]], dtype=np.float32)
    normalized = normalize_ocr_result([boxes, ["cell"]])
    assert len(normalized) == 3
    assert normalized[2] == []
    dt_boxes, rec_res = format_ocr_results([boxes, ["cell"]], 100, 100)
    assert dt_boxes.shape == (1, 4)
    assert rec_res == [("cell", 1.0)]


def test_single_box_and_three_item_result():
    single_box = [[0, 0], [10, 0], [10, 5], [0, 5]]
    boxes, texts = format_ocr_results([single_box], 100, 100)
    assert boxes.shape == (1, 4)
    assert texts == []

    boxes, texts = format_ocr_results([[[0, 0], [10, 0], [10, 5], [0, 5]], ["x"], [0.9]], 100, 100)
    assert boxes.shape == (1, 4)
    assert texts == [("x", 0.9)]
