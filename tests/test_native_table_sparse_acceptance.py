"""A mostly-empty native table can still be the complete parse.

Cross-page continuation tables often have one long cell and many empty cells,
so the blank-cell ratio alone rejects the native result (0.89 on the reported
2014 annual reports) and the table falls through to a structure model that
reads the narrow, vertically-wrapped text as a few fragments.  The native
parse must be judged by the text it preserves, not only by how many cells are
empty.
"""
from rapid_doc.backend.pipeline.analyze_utils import (
    _native_table_lost_signed_number,
    _native_table_text_coverage,
    _native_vertical_text_fragment,
    _table_html_blank_ratio,
)

BBOX = (0.0, 0.0, 500.0, 700.0)
TEXT = "有关上市规则及《公司章程》的规定，遵循等价、有偿、公平交易的原则，履行合法程序"


def _page(*spans):
    return {
        "blocks": [
            {
                "lines": [
                    {"spans": [{"bbox": [x0, y0, x1, y1], "text": text}
                               for x0, y0, x1, y1, text in spans]}
                ]
            }
        ]
    }


def _sparse_html(text):
    return ("<table><tbody><tr>"
            "<td></td><td></td><td></td>"
            f"<td>{text}</td>"
            "<td></td><td></td><td></td><td></td><td></td>"
            "</tr></tbody></table>")


def test_sparse_native_table_keeps_all_region_text():
    page = _page((240.0, 80.0, 290.0, 680.0, TEXT))
    html = _sparse_html(TEXT)

    assert _table_html_blank_ratio(html) > 0.45
    assert _native_table_text_coverage(page, BBOX, html) == 1.0


def test_native_table_that_drops_text_is_not_credited():
    page = _page((240.0, 80.0, 290.0, 680.0, TEXT))
    html = _sparse_html(TEXT[:10])

    assert _native_table_text_coverage(page, BBOX, html) < 0.9


def test_no_native_text_has_no_coverage():
    assert _native_table_text_coverage({"blocks": []}, BBOX, _sparse_html(TEXT)) is None


def test_vertical_text_strip_is_recognised_as_text():
    # One upright character per line at the page edge: a continuation cell,
    # not a table grid.  Each span is roughly square and stacked vertically.
    page = _page(*[(3.3, 92.0 + index * 13.0, 13.0, 101.5 + index * 13.0, char)
                   for index, char in enumerate("有限责任公司")])
    vertical_bbox = (0.0, 88.0, 842.0, 173.0)

    assert _native_vertical_text_fragment(page, vertical_bbox) == "有限责任公司"


def test_horizontal_text_does_not_become_a_vertical_fragment():
    page = _page((240.0, 80.0, 290.0, 680.0, TEXT))

    assert _native_vertical_text_fragment(page, BBOX) is None


def test_native_table_that_drops_a_minus_sign_is_rejected():
    page = _page((240.0, 80.0, 290.0, 680.0, "（二）所有者投入和减少资本-61,301,849.85"))
    html = _sparse_html("（二）所有者投入和减少资本61,301,849.85")

    assert _native_table_lost_signed_number(page, BBOX, html) is True


def test_native_table_that_keeps_the_minus_sign_is_accepted():
    page = _page((240.0, 80.0, 290.0, 680.0, "（二）所有者投入和减少资本－61,301,849.85"))
    html = _sparse_html("（二）所有者投入和减少资本-61,301,849.85")

    assert _native_table_lost_signed_number(page, BBOX, html) is False
