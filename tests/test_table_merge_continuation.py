from rapid_doc.utils.table_merge import (
    _apply_cell_merge,
    _align_rendered_column_grid,
    _infer_sparse_continuation_cell_merge,
    build_table_state_from_html,
    build_visual_col_mapping,
    perform_table_merge,
)


def test_continuation_aligns_columns_hidden_by_previous_rowspans():
    previous = build_table_state_from_html(
        '<table><tr><td rowspan="3">Category</td><td>Index</td><td colspan="11">Task</td>'
        '<td colspan="3">Person</td><td colspan="3">Work</td></tr>'
        '<tr><td rowspan="2">1</td><td colspan="11" rowspan="2">Task1</td>'
        '<td colspan="3">P1</td><td colspan="3">W1</td></tr>'
        '<tr><td colspan="3">P2</td><td colspan="3">W2</td></tr></table>'
    )
    current = build_table_state_from_html(
        '<table><tr><td rowspan="2"></td><td rowspan="2">2</td><td rowspan="2">Task2</td>'
        '<td>P1</td><td>W3</td></tr><tr><td>P2</td><td>W4</td></tr></table>'
    )
    owner = {'blocks': []}
    perform_table_merge(previous, current, owner, [])
    assert len(previous.rows) == 5
    cells = previous.rows[-2].find_all('td')
    assert [int(c.get('colspan', 1)) for c in cells] == [1, 1, 11, 3, 3]
    assert [int(c.get('rowspan', 1)) for c in cells] == [2, 2, 2, 1, 1]
    assert build_visual_col_mapping(previous.rows, -2) == [0, 1, 2, 13, 16]
    assert build_visual_col_mapping(previous.rows, -1) == [13, 16]
    assert previous.total_cols == 19


def test_larger_continuation_grid_expands_previous_inherited_cells():
    previous = build_table_state_from_html(
        '<table><tr><td rowspan="2">A</td><td rowspan="2">B</td><td>C</td></tr>'
        '<tr><td>D</td></tr></table>'
    )
    current = build_table_state_from_html(
        '<table><tr><td colspan="2">X</td><td colspan="4">Y</td><td colspan="3">Z</td></tr></table>'
    )
    perform_table_merge(previous, current, {'blocks': []}, [])
    assert [int(c.get('colspan', 1)) for c in previous.rows[0].find_all('td')] == [2, 4, 3]
    assert build_visual_col_mapping(previous.rows, 1) == [6]
    assert previous.total_cols == 9


def test_unmapped_interior_boundaries_do_not_partially_rewrite_table():
    source = build_table_state_from_html(
        '<table><tr><td colspan="2">A</td><td colspan="2">B</td><td colspan="2">C</td></tr>'
        '<tr>' + '<td>x</td>'*6 + '</tr></table>'
    )
    reference = build_table_state_from_html(
        '<table><tr><td>X</td><td colspan="3">Y</td><td colspan="5">Z</td></tr></table>'
    )
    original = str(source.soup)
    assert not _align_rendered_column_grid(source, reference, 0, 0)
    assert str(source.soup) == original
    assert source.total_cols == 6


def test_sparse_page_break_row_continues_previous_cells():
    previous = build_table_state_from_html(
        '<table><tr><td>9</td><td>海南（取送</td><td>1</td>'
        '<td>山东核电设备制造有</td></tr></table>'
    )
    current = build_table_state_from_html(
        '<table><tr><td></td><td>费）</td><td></td><td>限公司</td></tr>'
        '<tr><td>10</td><td>海南（不含税）</td><td>1</td><td>山东公司</td></tr></table>'
    )

    flags = _infer_sparse_continuation_cell_merge(previous, current, 0)
    assert flags == [0, 1, 0, 1]
    current.owner_block['cell_merge'] = flags
    _apply_cell_merge(previous, current, 0)

    assert len(current.rows) == 1
    assert previous.rows[-1].find_all('td')[1].get_text() == '海南（取送费）'
    assert previous.rows[-1].find_all('td')[3].get_text() == '山东核电设备制造有限公司'


def test_sparse_first_row_with_key_is_not_inferred_as_continuation():
    previous = build_table_state_from_html(
        '<table><tr><td>9</td><td>old</td><td>1</td></tr></table>'
    )
    current = build_table_state_from_html(
        '<table><tr><td>10</td><td>new</td><td></td></tr>'
        '<tr><td>11</td><td>next</td><td>1</td></tr></table>'
    )
    assert _infer_sparse_continuation_cell_merge(previous, current, 0) is None


def test_sharp_density_drop_blocks_cross_page_merge():
    from rapid_doc.utils.table_merge import can_merge_by_structure

    dense_rows = ''.join(
        f'<tr><td>row{i}</td><td>{i}</td></tr>' for i in range(10)
    )
    previous = build_table_state_from_html(
        f'<table><tr><td>name</td><td>value</td></tr>{dense_rows}</table>'
    )
    sparse_rows = ''.join(
        '<tr><td></td><td></td></tr>' for _ in range(20)
    )
    current = build_table_state_from_html(
        f'<table><tr><td>name</td><td>value</td></tr>'
        f'<tr><td>row10</td><td>10</td></tr>{sparse_rows}</table>'
    )

    assert not can_merge_by_structure(current, previous)


def test_dense_continuation_still_merges_by_structure():
    from rapid_doc.utils.table_merge import can_merge_by_structure

    previous = build_table_state_from_html(
        '<table><tr><td>name</td><td>value</td></tr>'
        '<tr><td>row0</td><td>0</td></tr></table>'
    )
    current = build_table_state_from_html(
        '<table><tr><td>name</td><td>value</td></tr>'
        '<tr><td>row1</td><td>1</td></tr></table>'
    )

    assert can_merge_by_structure(current, previous)


def test_sparse_continuation_not_merged_by_pipeline_rule():
    from rapid_doc.utils.table_merge import can_merge_tables

    previous = build_table_state_from_html(
        '<table><tr><td>name</td><td>value</td></tr>'
        '<tr><td>row0</td><td>0</td></tr><tr><td>row1</td><td>1</td></tr></table>'
    )
    current = build_table_state_from_html(
        '<table><tr><td>name</td><td>value</td></tr>'
        + '<tr><td></td><td></td></tr>' * 20
        + '</table>'
    )
    previous.owner_block = {'blocks': [], 'bbox': [0, 0, 100, 100]}
    current.owner_block = {'blocks': [], 'bbox': [0, 0, 100, 100]}

    assert not can_merge_tables(current, previous)
