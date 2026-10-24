from rapid_doc.utils.pdf_image_tools import _expand_image_bbox_with_native


def test_native_bbox_containing_model_bbox_is_used():
    model = [230.9, 166.2, 492.9, 346.8]
    native = [113.2, 94.9, 507.7, 425.4]

    assert _expand_image_bbox_with_native(
        model, [{"bbox": native}]
    ) == native


def test_wider_shorter_native_bbox_is_used_without_model_title():
    model = [92.7, 621.1, 528.3, 684.3]
    native = [89.9, 636.5, 532.4, 671.0]

    assert _expand_image_bbox_with_native(
        model, [{"bbox": native}]
    ) == native


def test_narrower_native_bbox_is_unioned_with_model_bbox():
    model = [100.0, 100.0, 300.0, 300.0]
    native = [250.0, 150.0, 350.0, 250.0]

    assert _expand_image_bbox_with_native(
        model, [{"bbox": native}]
    ) == [100.0, 100.0, 350.0, 300.0]


def test_non_overlapping_native_bbox_is_ignored():
    model = [100.0, 100.0, 200.0, 200.0]
    native = [300.0, 300.0, 400.0, 400.0]

    assert _expand_image_bbox_with_native(
        model, [{"bbox": native}]
    ) == model


def test_multiple_native_bboxes_are_merged():
    model = [100.0, 100.0, 200.0, 200.0]
    natives = [
        {"bbox": [90.0, 90.0, 210.0, 150.0]},
        {"bbox": [90.0, 150.0, 210.0, 210.0]},
    ]

    assert _expand_image_bbox_with_native(model, natives) == [
        90.0, 90.0, 210.0, 210.0
    ]
