import pytest

from rapid_doc.backend.pipeline.batch_analyze import BatchAnalyze
from rapid_doc.utils.config_reader import get_env_batch_size

BATCH_ENV_NAMES = (
    'RAPID_DOC_PAGE_BATCH_SIZE',
    'RAPID_DOC_LAYOUT_BATCH_SIZE',
    'RAPID_DOC_FORMULA_BATCH_SIZE',
    'RAPID_DOC_OCR_DET_BATCH_SIZE',
    'RAPID_DOC_OCR_REC_BATCH_SIZE',
    'RAPID_DOC_OCR_CLS_BATCH_SIZE',
    'RAPID_DOC_TABLE_CLS_BATCH_SIZE',
    'RAPID_DOC_TABLE_REC_BATCH_SIZE',
    'RAPID_DOC_TABLE_STRUCTURE_BATCH_SIZE',
)


def _clear_batch_env(monkeypatch):
    for name in BATCH_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_get_env_batch_size_uses_default_when_unset(monkeypatch):
    monkeypatch.delenv('RAPID_DOC_LAYOUT_BATCH_SIZE', raising=False)
    assert get_env_batch_size('RAPID_DOC_LAYOUT_BATCH_SIZE', 5) == 5
    assert get_env_batch_size('RAPID_DOC_LAYOUT_BATCH_SIZE') is None


@pytest.mark.parametrize('value', ['0', '-1', 'abc', ''])
def test_get_env_batch_size_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv('RAPID_DOC_LAYOUT_BATCH_SIZE', value)
    assert get_env_batch_size('RAPID_DOC_LAYOUT_BATCH_SIZE', 5) == 5


def test_batch_analyze_keeps_existing_batch_logic_without_env(monkeypatch):
    _clear_batch_env(monkeypatch)
    analyze = BatchAnalyze(
        model_manager=None,
        batch_ratio=2,
        formula_enable=True,
        table_enable=True,
        layout_config={'batch_num': 3},
        ocr_config={},
        formula_config={'batch_num': 4},
    )
    assert analyze.layout_base_batch_size == 6
    assert analyze.formula_base_batch_size == 8
    assert analyze.ocr_det_base_batch_size == 2
    assert analyze.ocr_config['Det.rec_batch_num'] == 2
    assert analyze.ocr_config['Rec.rec_batch_num'] == 2


def test_batch_analyze_env_overrides_final_batch_sizes(monkeypatch):
    _clear_batch_env(monkeypatch)
    monkeypatch.setenv('RAPID_DOC_LAYOUT_BATCH_SIZE', '7')
    monkeypatch.setenv('RAPID_DOC_FORMULA_BATCH_SIZE', '8')
    monkeypatch.setenv('RAPID_DOC_OCR_DET_BATCH_SIZE', '11')
    monkeypatch.setenv('RAPID_DOC_OCR_REC_BATCH_SIZE', '12')
    monkeypatch.setenv('RAPID_DOC_OCR_CLS_BATCH_SIZE', '13')

    analyze = BatchAnalyze(
        model_manager=None,
        batch_ratio=2,
        formula_enable=True,
        table_enable=True,
        layout_config={'batch_num': 3},
        ocr_config={},
        formula_config={'batch_num': 4},
    )
    assert analyze.layout_base_batch_size == 7
    assert analyze.formula_base_batch_size == 8
    assert analyze.ocr_det_base_batch_size == 11
    assert analyze.ocr_config['Det.rec_batch_num'] == 11
    assert analyze.ocr_config['Rec.rec_batch_num'] == 12
    assert analyze.ocr_config['Cls.cls_batch_num'] == 13
