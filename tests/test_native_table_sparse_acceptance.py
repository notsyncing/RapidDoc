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
    _native_text_grid_html,
    _native_vertical_text_column,
    _native_vertical_text_fragment,
    _table_html_blank_ratio,
    _table_html_grid_shape,
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


def test_coverage_normalises_typographic_quotes():
    # The native HTML writes ASCII quotes while the PDF text carries the
    # typographic form; that is not lost content.
    page = _page((240.0, 80.0, 290.0, 680.0, "“追加补偿探矿权”"))
    html = _sparse_html('"追加补偿探矿权"')

    assert _native_table_text_coverage(page, BBOX, html) == 1.0


def test_vertical_text_column_accepts_multi_glyph_runs():
    # Wrapped continuation text arrives as multi-character horizontal runs in
    # a narrow column, not one glyph per line.
    text = ("有关上市规则及公司章程的规定遵循等价有偿公平交易的原则履行合法程序"
            "保证信息披露真实准确完整不得有虚假记载误导性陈述或重大遗漏"
            "公司董事会应当勤勉尽责维护公司及全体股东的合法权益")
    rows = [text[index:index + 4] for index in range(0, len(text), 4)]
    page = _page(*[(240.0, 80.0 + index * 17.0, 280.0, 90.0 + index * 17.0, run)
                   for index, run in enumerate(rows)])

    assert _native_vertical_text_column(page, BBOX) == text


def test_wide_horizontal_paragraph_is_not_a_vertical_column():
    text = "有关上市规则及公司章程的规定遵循等价有偿公平交易的原则履行合法程序"
    rows = [text[index:index + 4] for index in range(0, len(text), 4)]
    page = _page(*[(20.0, 80.0 + index * 14.5, 480.0, 90.0 + index * 14.5, run)
                   for index, run in enumerate(rows)])

    assert _native_vertical_text_column(page, BBOX) is None


def _grid_page(rows):
    chars = []
    for y, cells in rows:
        for x, text in cells:
            for offset, char in enumerate(text):
                left = x + offset * 10.0
                chars.append({"char": char, "bbox": (left, y, left + 9.0, y + 9.0)})
    return {"chars": chars}


def test_native_text_grid_rebuilds_rows_and_columns():
    page = _grid_page([
        (5.0, [(5.0, "第一组")]),
        (40.0, [(5.0, "甲乙"), (105.0, "丙丁"), (205.0, "12")]),
        (70.0, [(5.0, "戊己"), (105.0, "庚辛"), (205.0, "34")]),
    ])
    rules = [(0.0, 0.0, 300.0, 0.0), (0.0, 30.0, 300.0, 30.0),
             (0.0, 60.0, 300.0, 60.0), (0.0, 100.0, 300.0, 100.0)]

    html = _native_text_grid_html(page, (0.0, 0.0, 300.0, 100.0), rules=rules)

    assert html is not None
    assert 'colspan="3"' in html
    assert "第一组" in html
    assert "甲乙" in html and "丙丁" in html and "12" in html
    assert "戊己" in html and "庚辛" in html and "34" in html


def test_native_text_grid_needs_more_than_one_column():
    page = _grid_page([(5.0, [(5.0, "只有一列")]), (40.0, [(5.0, "还是这一列")])])
    rules = [(0.0, 0.0, 300.0, 0.0), (0.0, 30.0, 300.0, 30.0),
             (0.0, 100.0, 300.0, 100.0)]

    assert _native_text_grid_html(page, (0.0, 0.0, 300.0, 100.0), rules=rules) is None


def test_table_html_grid_shape_counts_colspan():
    html = ('<table><tr><td colspan="3">标题</td></tr>'
            '<tr><td>a</td><td>b</td><td>c</td></tr></table>')

    assert _table_html_grid_shape(html) == (2, 3)
