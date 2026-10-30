import asyncio
import gc
import glob
import io
import json
import os
import re
import shutil
import tempfile
import time
import uuid
import zipfile
from base64 import b64encode
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, Response
from loguru import logger

from file_converter import ensure_pdf, OFFICE_EXTENSIONS
from rapid_doc.cli.common import aio_do_parse, pdf_suffixes, image_suffixes, office_suffixes
from rapid_doc.utils.empty_office import normalize_empty_office_bytes
from rapid_doc.utils.hash_utils import make_hashable
from rapid_doc.utils.pdf_image_tools import images_bytes_to_pdf_bytes
from rapid_doc.model.custom.paddleocr_vl.paddleocr_vl import PaddleOCRVLTableModel, PaddleOCRVLOCRModel, PaddleOCRVLFormulaModel
from rapid_doc.version import __version__

app = FastAPI(
    title="RapidDoc Web API",
    description="Compatible with RapidDoc official API - Parse documents using RapidDoc",
    version=__version__
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, value)


# Request coalescing.  Concurrent /file_parse calls are merged into one
# internal RapidDoc batch; RapidDoc's own document workers (RAPID_DOC_DOC_WORKERS)
# then overlap host work with the single in-order GPU queue.  Only one batch
# executes at a time: several independent pipelines on one A770 contend for the
# same device and lose more to that contention than they gain from concurrency.
_API_CONCURRENCY = _env_int("RAPID_DOC_API_CONCURRENCY", 1, 1)
_BATCH_ENABLED = os.getenv("RAPID_DOC_API_BATCH", "1").lower() not in (
    "0", "false", "off", "no", "disabled")
_BATCH_MAX_FILES = _env_int("RAPID_DOC_API_BATCH_MAX_FILES", 16, 1)
# 0 means "no page limit"; the default fits the 12-document error corpus
# (957 pages) in one batch while still refusing an unbounded upload.
_BATCH_MAX_PAGES = _env_int("RAPID_DOC_API_BATCH_MAX_PAGES", 1200, 0)
_BATCH_WINDOW_S = _env_int("RAPID_DOC_API_BATCH_WINDOW_MS", 100, 0) / 1000.0
_BATCH_MAX_WINDOW_S = max(
    _BATCH_WINDOW_S,
    _env_int("RAPID_DOC_API_BATCH_MAX_WINDOW_MS", 1000, 0) / 1000.0,
)
# A 12-document request is ~700-800 s on one A770.  The timeout is the hard
# upper bound that turns a wedged request into a 504 instead of a hung client.
_REQUEST_TIMEOUT_S = _env_int("RAPID_DOC_API_REQUEST_TIMEOUT_S", 1800, 0)
_QUEUE_MAX = _env_int("RAPID_DOC_API_QUEUE_MAX", 64, 1)

# One physical inference lane.  The batch dispatcher and the fallback
# (clear_output_file=false, office/image, return_model_output, ...) share it.
_GPU_LANE = asyncio.Semaphore(1)
_BATCH_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rapiddoc-batch")
_RESPONSE_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rapiddoc-response")
_FALLBACK_EXECUTOR = ThreadPoolExecutor(
    max_workers=_API_CONCURRENCY, thread_name_prefix="rapiddoc-parse")

_QUEUE: "asyncio.Queue[_ParseJob]" = asyncio.Queue()
_DISPATCHER_TASK: Optional[asyncio.Task] = None
_ACTIVE_UPLOADS = 0


class _RequestError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class _NamedPdfBytes(bytes):
    """bytes that also look like a readable, named stream to RapidDoc.

    RapidDoc derives a document's output directory from ``item.name``; passing
    raw bytes would name every document ``document_N`` and collide across a
    coalesced batch.  It also sorts a batch by input size, so keeping the bytes
    subclass preserves the longest-first scheduling.
    """

    def __new__(cls, data: bytes, name: str):
        obj = super().__new__(cls, data)
        obj.name = name
        return obj

    def read(self, *args, **kwargs) -> bytes:
        return bytes(self)

    def seek(self, *args, **kwargs) -> int:
        return 0


@dataclass
class _FileSpec:
    index: int
    name: str
    content: bytes
    suffix: str
    pages: Optional[int] = None
    internal_name: str = ""


@dataclass
class _RequestSpec:
    job_id: str
    files: List[_FileSpec]
    output_dir: str
    clear_output_file: bool
    backend: str
    parse_method: str
    formula_enable: bool
    table_enable: bool
    layout_config: Dict[str, Any]
    ocr_config: Dict[str, Any]
    formula_config: Dict[str, Any]
    table_config: Dict[str, Any]
    checkbox_config: Dict[str, Any]
    image_config: Dict[str, Any]
    lang_list: List[str]
    return_md: bool
    return_middle_json: bool
    return_model_output: bool
    return_content_list: bool
    return_images: bool
    response_format_zip: bool
    start_page_id: int
    end_page_id: int
    key: tuple = ()


@dataclass
class _ParseJob:
    spec: _RequestSpec
    future: asyncio.Future
    pending: int = 0
    results: Dict[int, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class _Unit:
    job: _ParseJob
    files: List[_FileSpec]


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "version": __version__,
        "api": "RapidDoc Web API",
        "compatible": "Official RapidDoc API",
        "batch": {
            "enabled": _BATCH_ENABLED,
            "max_files": _BATCH_MAX_FILES,
            "max_pages": _BATCH_MAX_PAGES,
            "window_ms": int(_BATCH_WINDOW_S * 1000),
            "doc_workers": os.getenv("RAPID_DOC_DOC_WORKERS", "6"),
        },
    }

# 支持的文件扩展名 - 与官方 API 保持一致
office_extensions = [suffix[1:] for suffix in OFFICE_EXTENSIONS]

def sanitize_filename(filename: str) -> str:
    """
    格式化压缩文件的文件名
    移除路径遍历字符, 保留 Unicode 字母、数字、._-
    禁止隐藏文件
    """
    sanitized = re.sub(r'[/\\\.]{2,}|[/\\]', '', filename)
    sanitized = re.sub(r'[^\w.-]', '_', sanitized, flags=re.UNICODE)
    if sanitized.startswith('.'):
        sanitized = '_' + sanitized[1:]
    return sanitized or 'unnamed'

def cleanup_file(file_path: str) -> None:
    """清理临时 zip 文件"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        logger.warning(f"fail clean file {file_path}: {e}")

def encode_image(image_path: str) -> str:
    """Encode image using base64"""
    with open(image_path, "rb") as f:
        return b64encode(f.read()).decode()


def get_infer_result(file_suffix_identifier: str, pdf_name: str, parse_dir: str) -> Optional[str]:
    """从结果文件中读取推理结果"""
    result_file_path = os.path.join(parse_dir, f"{pdf_name}{file_suffix_identifier}")
    if os.path.exists(result_file_path):
        with open(result_file_path, "r", encoding="utf-8") as fp:
            return fp.read()
    return None


def resolve_parse_dir(output_dir: str, pdf_name: str, backend: str, parse_method: str) -> str:
    """Resolve the actual parse directory written by aio_do_parse."""
    office_dir = os.path.join(output_dir, pdf_name, "office")
    if os.path.exists(office_dir):
        return office_dir

    if backend.startswith("pipeline"):
        return os.path.join(output_dir, pdf_name, parse_method)

    return os.path.join(output_dir, pdf_name, "vlm")

def _convert_value_to_enum(config):
    """
      自动将配置 dict 中的字符串枚举（如 "OCRVersion.PPOCRV5"）转换为实际枚举对象。
      """
    # ⚡ 如果 config 是空字典，直接返回
    if not config:
        return config
    from rapidocr import EngineType as OCREngineType, OCRVersion, ModelType as OCRModelType, LangDet, LangRec
    from rapid_doc.model.layout.rapid_layout_self import ModelType as LayoutModelType
    from rapid_doc.model.formula.rapid_formula_self import ModelType as FormulaModelType, EngineType as FormulaEngineType
    from rapid_doc.model.table.rapid_table_self import ModelType as TableModelType

    # 可识别的枚举类映射表（可扩展）
    enum_map = {
        "OCREngineType": OCREngineType,
        "OCRVersion": OCRVersion,
        "OCRModelType": OCRModelType,
        "LangDet": LangDet,
        "LangRec": LangRec,
        "LayoutModelType": LayoutModelType,
        "FormulaModelType": FormulaModelType,
        "FormulaEngineType": FormulaEngineType,
        "TableModelType": TableModelType,
    }

    def resolve_enum_value(value):
        """递归解析单个值"""
        if isinstance(value, str) and "." in value:
            prefix, member = value.split(".", 1)
            if prefix in enum_map and hasattr(enum_map[prefix], member):
                return getattr(enum_map[prefix], member)
        elif isinstance(value, dict):
            return {k: resolve_enum_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [resolve_enum_value(v) for v in value]
        return value

    return {k: resolve_enum_value(v) for k, v in config.items()}


def has_vl_env():
    return all([
        os.getenv('PADDLEOCRVL_VL_REC_BACKEND'),
        os.getenv('PADDLEOCRVL_VL_VL_REC_SERVER_URL'),
        os.getenv('PADDLEOCRVL_VERSION'),
    ])


# ---------------------------------------------------------------------------
# Request preparation
# ---------------------------------------------------------------------------

def _pdf_page_count(content: bytes) -> Optional[int]:
    """Count a PDF's pages without keeping the document open."""
    try:
        import pypdfium2 as pdfium
        from rapid_doc.utils import PyPDFium2Parser

        with PyPDFium2Parser.lock:
            pdf = pdfium.PdfDocument(content)
            try:
                return len(pdf)
            finally:
                pdf.close()
    except Exception as e:
        logger.warning(f"[api-batch] cannot count PDF pages: {e}")
        return None


def _effective_pages(content: bytes, start_page_id: int, end_page_id: int) -> Optional[int]:
    total = _pdf_page_count(content)
    if total is None:
        return None
    if total <= 0:
        return 0
    start = max(0, int(start_page_id))
    if end_page_id is None or int(end_page_id) < 0 or int(end_page_id) >= total:
        end = total - 1
    else:
        end = int(end_page_id)
    return max(0, end - start + 1)


async def _read_uploaded_files(
    files: List[UploadFile],
    start_page_id: int,
    end_page_id: int,
) -> List[_FileSpec]:
    specs: List[_FileSpec] = []
    for index, file in enumerate(files):
        filename = file.filename or "unnamed"
        file_suffix = Path(filename).suffix[1:].lower()
        if file_suffix not in pdf_suffixes + office_extensions + image_suffixes:
            raise _RequestError(
                400, f"File type {file_suffix} is not supported for {filename}")

        if file_suffix in pdf_suffixes + image_suffixes + office_suffixes:
            content = await file.read()
            content = normalize_empty_office_bytes(content, file_suffix)
            if file_suffix in image_suffixes:
                content = images_bytes_to_pdf_bytes(content)
        else:
            # Other document formats still use the external converter.
            temp_dir = tempfile.mkdtemp(prefix="fastapi_adapter_")
            try:
                temp_file_path = os.path.join(temp_dir, filename)
                with open(temp_file_path, "wb") as tmp_f:
                    tmp_f.write(await file.read())
                convert_file_path, _temp_pdf_to_delete = ensure_pdf(temp_file_path, temp_dir)
                if not convert_file_path:
                    raise _RequestError(
                        500, f"Failed to convert Office/document: {filename}")
                with open(convert_file_path, "rb") as f:
                    content = f.read()
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        pages = None
        if file_suffix in pdf_suffixes:
            pages = await asyncio.to_thread(
                _effective_pages, content, start_page_id, end_page_id)
        specs.append(_FileSpec(
            index=index,
            name=Path(filename).stem,
            content=content,
            suffix=file_suffix,
            pages=pages,
        ))
    return specs


def _config_key(spec: _RequestSpec) -> tuple:
    return (
        spec.backend,
        spec.parse_method,
        bool(spec.formula_enable),
        bool(spec.table_enable),
        make_hashable(spec.layout_config),
        make_hashable(spec.ocr_config),
        make_hashable(spec.formula_config),
        make_hashable(spec.table_config),
        make_hashable(spec.checkbox_config),
        make_hashable(spec.image_config),
        tuple(spec.lang_list),
        int(spec.start_page_id),
        int(spec.end_page_id),
        os.path.abspath(spec.output_dir),
    )


def _coalesce_reason(spec: _RequestSpec) -> str:
    """Return an empty string when the request can join a shared batch."""
    if not _BATCH_ENABLED:
        return "batch disabled"
    if spec.backend != "pipeline":
        return f"backend={spec.backend}"
    if not spec.clear_output_file:
        return "clear_output_file=false"
    if spec.return_model_output:
        return "return_model_output"
    if has_vl_env():
        return "paddleocr-vl env"
    for cfg in (
        spec.layout_config, spec.ocr_config, spec.formula_config,
        spec.table_config, spec.checkbox_config, spec.image_config,
    ):
        if isinstance(cfg, dict) and cfg.get("custom_model") is not None:
            return "custom_model config"
    if not spec.files:
        return "no files"
    for f in spec.files:
        if f.suffix not in pdf_suffixes:
            return f"non-pdf:{f.suffix}"
        if f.pages is None:
            return "unknown page count"
    return ""


def _assign_internal_names(job: _ParseJob) -> None:
    prefix = job.spec.job_id[:8]
    for f in job.spec.files:
        safe = sanitize_filename(f.name)[:80] or "doc"
        f.internal_name = f"b{prefix}_{f.index:03d}_{safe}"


# ---------------------------------------------------------------------------
# Result reading / response building
# ---------------------------------------------------------------------------

def _read_results(
    spec: _RequestSpec,
    root: str,
    files: List[_FileSpec],
) -> Dict[int, Dict[str, Any]]:
    results: Dict[int, Dict[str, Any]] = {}
    for f in files:
        parse_dir = resolve_parse_dir(root, f.internal_name, spec.backend, spec.parse_method)
        raw: Dict[str, Any] = {"found": os.path.isdir(parse_dir)}
        if raw["found"]:
            if spec.return_md:
                raw["md"] = get_infer_result(".md", f.internal_name, parse_dir)
            if spec.return_middle_json:
                raw["middle"] = get_infer_result("_middle.json", f.internal_name, parse_dir)
            if spec.return_model_output:
                raw["model"] = get_infer_result("_model.json", f.internal_name, parse_dir)
            if spec.return_content_list:
                raw["content"] = get_infer_result("_content_list.json", f.internal_name, parse_dir)
            if spec.return_images:
                images_dir = os.path.join(parse_dir, "images")
                image_paths = (
                    glob.glob(os.path.join(glob.escape(images_dir), "*.jpg"))
                    + glob.glob(os.path.join(glob.escape(images_dir), "*.png"))
                )
                raw["images"] = {
                    os.path.basename(path): Path(path).read_bytes()
                    for path in image_paths
                }
        results[f.index] = raw
    return results


def _build_payload_sync(spec: _RequestSpec, results: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    if spec.response_format_zip:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for f in spec.files:
                raw = results.get(f.index) or {}
                if not raw.get("found"):
                    continue
                safe = sanitize_filename(f.name)
                if spec.return_md and raw.get("md") is not None:
                    zf.writestr(f"{safe}/{safe}.md", raw["md"])
                if spec.return_middle_json and raw.get("middle") is not None:
                    zf.writestr(f"{safe}/{safe}_middle.json", raw["middle"])
                if spec.return_model_output and raw.get("model") is not None:
                    zf.writestr(f"{safe}/{safe}_model.json", raw["model"])
                if spec.return_content_list and raw.get("content") is not None:
                    zf.writestr(f"{safe}/{safe}_content_list.json", raw["content"])
                if spec.return_images:
                    for image_name, image_bytes in (raw.get("images") or {}).items():
                        zf.writestr(f"{safe}/images/{image_name}", image_bytes)
        return {
            "kind": "zip",
            "bytes": buffer.getvalue(),
            "filename": "results.zip",
        }

    result_dict: Dict[str, Any] = {}
    for f in spec.files:
        raw = results.get(f.index) or {}
        data: Dict[str, Any] = {}
        if raw.get("found"):
            if spec.return_md:
                data["md_content"] = raw.get("md")
            if spec.return_middle_json:
                data["middle_json"] = (
                    json.loads(raw["middle"]) if raw.get("middle") is not None else None)
            if spec.return_model_output:
                data["model_output"] = (
                    json.loads(raw["model"]) if raw.get("model") is not None else None)
            if spec.return_content_list:
                data["content_list"] = (
                    json.loads(raw["content"]) if raw.get("content") is not None else None)
            if spec.return_images:
                data["images"] = {
                    name: f"data:image/jpeg;base64,{b64encode(image_bytes).decode()}"
                    for name, image_bytes in (raw.get("images") or {}).items()
                }
        result_dict[f.name] = data
    return {
        "kind": "json",
        "content": {
            "backend": spec.backend,
            "version": __version__,
            "results": result_dict,
        },
    }


def _payload_to_response(payload: Dict[str, Any]) -> Response:
    if payload["kind"] == "zip":
        return Response(
            content=payload["bytes"],
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="results.zip"'},
        )
    return JSONResponse(status_code=200, content=payload["content"])


# ---------------------------------------------------------------------------
# Batch execution
# ---------------------------------------------------------------------------

def _run_aio_parse_sync(spec: _RequestSpec) -> Dict[str, Any]:
    """Fallback: one request, one aio_do_parse pipeline."""
    unique_dir = os.path.join(spec.output_dir, str(uuid.uuid4()))
    os.makedirs(unique_dir, exist_ok=True)
    logger.info(f"Created unique output directory: {unique_dir}")
    for f in spec.files:
        if not f.internal_name:
            f.internal_name = f.name
    try:
        asyncio.run(aio_do_parse(
            output_dir=unique_dir,
            pdf_file_names=[f.name for f in spec.files],
            pdf_bytes_list=[f.content for f in spec.files],
            p_lang_list=list(spec.lang_list),
            backend=spec.backend,
            parse_method=spec.parse_method,
            formula_enable=spec.formula_enable,
            table_enable=spec.table_enable,
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
            f_dump_md=spec.return_md,
            f_dump_middle_json=spec.return_middle_json,
            f_dump_model_output=spec.return_model_output,
            f_dump_orig_pdf=False,
            f_dump_content_list=spec.return_content_list,
            start_page_id=spec.start_page_id,
            end_page_id=spec.end_page_id,
            layout_config=spec.layout_config,
            ocr_config=spec.ocr_config,
            formula_config=spec.formula_config,
            table_config=spec.table_config,
            checkbox_config=spec.checkbox_config,
            image_config=spec.image_config,
        ))
        results = _read_results(spec, unique_dir, spec.files)
        return _build_payload_sync(spec, results)
    finally:
        if spec.clear_output_file:
            shutil.rmtree(unique_dir, ignore_errors=True)


def _run_rapid_doc_batch_sync(
    batch: List[_Unit],
    batch_root: str,
) -> List[tuple]:
    """Run one coalesced batch through RapidDoc's document workers.

    ``RapidDoc.__call__`` spawns ``min(RAPID_DOC_DOC_WORKERS, len(inputs))``
    document threads.  Each document keeps its own call sequence and page
    batches; only the model entry points are serialised by RapidDoc's
    per-session locks, which is what lets host render/post overlap the single
    in-order device queue.
    """
    from rapid_doc import RapidDoc
    from rapid_doc.utils.config_reader import get_processing_window_size

    spec = batch[0].job.spec
    inputs = []
    langs = []
    for unit in batch:
        for f in unit.files:
            inputs.append(_NamedPdfBytes(f.content, f.internal_name + ".pdf"))
            langs.append(spec.lang_list[f.index])

    engine = os.getenv("RAPID_DOC_ENGINE")
    if not engine and os.getenv("SYCL_INFER_ONNX_LIB"):
        engine = "sycl-infer"
    parser = RapidDoc(
        layout_config=spec.layout_config,
        ocr_config=spec.ocr_config,
        formula_config=spec.formula_config,
        table_config=spec.table_config,
        checkbox_config=spec.checkbox_config,
        image_config=spec.image_config,
        parse_method=spec.parse_method,
        formula_enable=spec.formula_enable,
        table_enable=spec.table_enable,
        lang=langs[0] if langs else "ch",
        inference_engine=engine or None,
        sycl_device=os.getenv("SYCL_INFER_ONNX_DEVICE", "GPU"),
        pdf_pages_batch=get_processing_window_size(default=64),
    )
    parser(
        inputs,
        output_dir=batch_root,
        parse_method=spec.parse_method,
        formula_enable=spec.formula_enable,
        table_enable=spec.table_enable,
        lang=langs,
        start_page_id=spec.start_page_id,
        end_page_id=spec.end_page_id,
        f_dump_middle_json=any(u.job.spec.return_middle_json for u in batch),
        f_dump_content_list=any(u.job.spec.return_content_list for u in batch),
        f_draw_layout_bbox=False,
        f_draw_span_bbox=False,
    )

    parts: List[tuple] = []
    for unit in batch:
        try:
            results = _read_results(unit.job.spec, batch_root, unit.files)
        except Exception as e:  # one bad document must not hide the others
            parts.append((unit, None, e))
        else:
            parts.append((unit, results, None))
    return parts


def _split_job(job: _ParseJob) -> List[_Unit]:
    max_files = _BATCH_MAX_FILES
    max_pages = _BATCH_MAX_PAGES
    units: List[_Unit] = []
    current: List[_FileSpec] = []
    pages = 0

    def flush():
        nonlocal current, pages
        if current:
            units.append(_Unit(job=job, files=current))
            current = []
            pages = 0

    for f in job.spec.files:
        p = f.pages or 0
        if current and (
            len(current) >= max_files
            or (max_pages and pages + p > max_pages)
        ):
            flush()
        current.append(f)
        pages += p
    flush()
    return units


def _form_batches(units: List[_Unit]) -> List[List[_Unit]]:
    groups: "Dict[tuple, List[_Unit]]" = {}
    order: List[tuple] = []
    for unit in units:
        key = unit.job.spec.key
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(unit)

    batches: List[List[_Unit]] = []
    for key in order:
        current: List[_Unit] = []
        files = 0
        pages = 0
        for unit in groups[key]:
            unit_files = len(unit.files)
            unit_pages = sum(f.pages or 0 for f in unit.files)
            if current and (
                files + unit_files > _BATCH_MAX_FILES
                or (_BATCH_MAX_PAGES and pages + unit_pages > _BATCH_MAX_PAGES)
            ):
                batches.append(current)
                current = []
                files = 0
                pages = 0
            current.append(unit)
            files += unit_files
            pages += unit_pages
        if current:
            batches.append(current)
    return batches


def _fail_job(job: _ParseJob, exc: BaseException) -> None:
    if job.future.done():
        return
    logger.error(f"[api-batch] job {job.spec.job_id} failed: {exc}")
    job.future.set_exception(exc)


async def _finalize_job(job: _ParseJob) -> None:
    if job.future.done() or job.pending != 0:
        return
    loop = asyncio.get_running_loop()
    try:
        payload = await loop.run_in_executor(
            _RESPONSE_EXECUTOR, _build_payload_sync, job.spec, job.results)
    except Exception as e:
        logger.exception(e)
        _fail_job(job, e)
        return
    if not job.future.done():
        job.future.set_result(payload)
    job.results.clear()


async def _execute_batch(batch: List[_Unit]) -> None:
    batch = [u for u in batch if not u.job.future.done()]
    if not batch:
        return

    spec = batch[0].job.spec
    batch_root = os.path.join(spec.output_dir, f"_batch_{uuid.uuid4().hex}")
    loop = asyncio.get_running_loop()
    started = time.monotonic()
    logger.info(
        "[api-batch] batch start: requests={} files={} pages={} dir={}",
        len({id(u.job) for u in batch}),
        sum(len(u.files) for u in batch),
        sum(f.pages or 0 for u in batch for f in u.files),
        batch_root,
    )

    async with _GPU_LANE:
        parts = None
        batch_error: Optional[BaseException] = None
        try:
            os.makedirs(batch_root, exist_ok=True)
            parts = await loop.run_in_executor(
                _BATCH_EXECUTOR, _run_rapid_doc_batch_sync, batch, batch_root)
        except Exception as e:
            batch_error = e
            logger.exception(e)
        finally:
            shutil.rmtree(batch_root, ignore_errors=True)

    logger.info(
        "[api-batch] batch done: requests={} files={} {:.1f}s",
        len({id(u.job) for u in batch}),
        sum(len(u.files) for u in batch),
        time.monotonic() - started,
    )

    if parts is None:
        error = batch_error or RuntimeError("batch parse failed")
        for unit in batch:
            _fail_job(unit.job, error)
        return

    touched: Dict[int, _ParseJob] = {}
    for unit, results, error in parts:
        touched[id(unit.job)] = unit.job
        if error is not None:
            _fail_job(unit.job, error)
        elif not unit.job.future.done():
            unit.job.results.update(results)
            unit.job.pending -= 1

    for job in touched.values():
        if job.pending == 0 and not job.future.done():
            await _finalize_job(job)


async def _execute_fallback(spec: _RequestSpec) -> Dict[str, Any]:
    loop = asyncio.get_running_loop()
    async with _GPU_LANE:
        return await loop.run_in_executor(
            _FALLBACK_EXECUTOR, _run_aio_parse_sync, spec)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

async def _collect_jobs(first: _ParseJob) -> List[_ParseJob]:
    loop = asyncio.get_running_loop()
    jobs = [first]
    start = loop.time()
    deadline = start + _BATCH_WINDOW_S
    max_deadline = start + _BATCH_MAX_WINDOW_S

    while True:
        now = loop.time()
        if _ACTIVE_UPLOADS > 0:
            deadline = min(max(deadline, now + _BATCH_WINDOW_S), max_deadline)
        remaining = deadline - now
        if remaining <= 0:
            break
        try:
            # Poll so another handler that started reading after this one
            # can extend the coalescing window.
            job = await asyncio.wait_for(
                _QUEUE.get(), timeout=min(remaining, 0.05))
        except asyncio.TimeoutError:
            continue
        jobs.append(job)
    return jobs


async def _dispatcher_loop() -> None:
    while True:
        try:
            first = await _QUEUE.get()
            jobs = await _collect_jobs(first)
            jobs = [j for j in jobs if not j.future.done()]
            if not jobs:
                continue

            units: List[_Unit] = []
            for job in jobs:
                _assign_internal_names(job)
                job_units = _split_job(job)
                job.pending = len(job_units)
                units.extend(job_units)
            if not units:
                for job in jobs:
                    _fail_job(job, RuntimeError("no files to parse"))
                continue

            for batch in _form_batches(units):
                await _execute_batch(batch)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception(e)
            await asyncio.sleep(0.05)


def _ensure_dispatcher() -> None:
    global _DISPATCHER_TASK
    if _DISPATCHER_TASK is None or _DISPATCHER_TASK.done():
        _DISPATCHER_TASK = asyncio.create_task(
            _dispatcher_loop(), name="rapiddoc-batch-dispatcher")


async def _submit_job(spec: _RequestSpec) -> Dict[str, Any]:
    if _QUEUE.qsize() >= _QUEUE_MAX:
        raise _RequestError(503, "server busy: parse queue is full")
    loop = asyncio.get_running_loop()
    job = _ParseJob(spec=spec, future=loop.create_future())
    _ensure_dispatcher()
    await _QUEUE.put(job)
    logger.info(
        "[api-batch] queued job {}: files={} pages={}",
        spec.job_id,
        len(spec.files),
        sum(f.pages or 0 for f in spec.files),
    )
    try:
        if _REQUEST_TIMEOUT_S:
            return await asyncio.wait_for(job.future, timeout=_REQUEST_TIMEOUT_S)
        return await job.future
    except asyncio.TimeoutError:
        if not job.future.done():
            job.future.cancel()
        raise _RequestError(
            504, f"parse timed out after {_REQUEST_TIMEOUT_S}s")


# ---------------------------------------------------------------------------
# HTTP endpoint
# ---------------------------------------------------------------------------

@app.post(
    "/file_parse",
    tags=["projects"],
    summary="Parse files using RapidDoc - Compatible with official API",
)
async def file_parse(
    request: Request,
    files: List[UploadFile] = File(...),
    output_dir: str = Form("./output"),
    clear_output_file: bool = Form(True),
    lang_list: List[str] = Form(["ch"]),
    backend: str = Form("pipeline"),
    parse_method: str = Form("auto"),
    formula_enable: bool = Form(True),
    table_enable: bool = Form(True),
    layout_config: Optional[str] = Form("{}"),
    ocr_config: Optional[str] = Form("{}"),
    formula_config: Optional[str] = Form("{}"),
    table_config: Optional[str] = Form("{}"),
    checkbox_config: Optional[str] = Form("{}"),
    image_config: Optional[str] = Form("{}"),
    return_md: bool = Form(True),
    return_middle_json: bool = Form(False),
    return_model_output: bool = Form(False),
    return_content_list: bool = Form(False),
    return_images: bool = Form(True),
    response_format_zip: bool = Form(False),
    start_page_id: int = Form(0),
    end_page_id: int = Form(99999),
):
    """
    Parse files using RapidDoc - Compatible with official API

    Concurrent requests with the same parsing configuration are coalesced into
    one internal RapidDoc batch.  The document workers inside RapidDoc then run
    the batch, so several HTTP clients share one GPU pipeline instead of each
    starting its own.
    """
    global _ACTIVE_UPLOADS
    try:
        # Newer FastAPI treats omitted bool Form fields as unchecked checkboxes
        # (False), including fields declared with Form(True). This API uses
        # omission to mean the documented default; explicit false still wins.
        form_data = await request.form()
        clear_output_file = clear_output_file if 'clear_output_file' in form_data else True
        formula_enable = formula_enable if 'formula_enable' in form_data else True
        table_enable = table_enable if 'table_enable' in form_data else True
        return_md = return_md if 'return_md' in form_data else True
        return_images = return_images if 'return_images' in form_data else True

        layout_config = _convert_value_to_enum(json.loads(layout_config or "{}"))
        ocr_config = _convert_value_to_enum(json.loads(ocr_config or "{}"))
        formula_config = _convert_value_to_enum(json.loads(formula_config or "{}"))
        table_config = _convert_value_to_enum(json.loads(table_config or "{}"))
        checkbox_config = json.loads(checkbox_config or "{}")
        image_config = json.loads(image_config or "{}")
        if has_vl_env():
            ocr_config["custom_model"] = PaddleOCRVLOCRModel()
            formula_config["custom_model"] = PaddleOCRVLFormulaModel()
            table_config["custom_model"] = PaddleOCRVLTableModel()

        _ACTIVE_UPLOADS += 1
        try:
            file_specs = await _read_uploaded_files(
                files, start_page_id, end_page_id)
        finally:
            _ACTIVE_UPLOADS -= 1

        actual_lang_list = list(lang_list)
        if len(actual_lang_list) != len(file_specs):
            actual_lang_list = [
                actual_lang_list[0] if actual_lang_list else "ch"
            ] * len(file_specs)

        spec = _RequestSpec(
            job_id=uuid.uuid4().hex,
            files=file_specs,
            output_dir=output_dir,
            clear_output_file=clear_output_file,
            backend=backend,
            parse_method=parse_method,
            formula_enable=formula_enable,
            table_enable=table_enable,
            layout_config=layout_config,
            ocr_config=ocr_config,
            formula_config=formula_config,
            table_config=table_config,
            checkbox_config=checkbox_config,
            image_config=image_config,
            lang_list=actual_lang_list,
            return_md=return_md,
            return_middle_json=return_middle_json,
            return_model_output=return_model_output,
            return_content_list=return_content_list,
            return_images=return_images,
            response_format_zip=response_format_zip,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
        )
        reason = _coalesce_reason(spec)
        if not reason:
            try:
                spec.key = _config_key(spec)
            except Exception as e:
                reason = f"unhashable config: {e}"

        if reason:
            logger.info(
                "[api-batch] job {} runs independently: {}", spec.job_id, reason)
            payload = await _execute_fallback(spec)
        else:
            payload = await _submit_job(spec)
        return _payload_to_response(payload)

    except _RequestError as e:
        logger.warning(f"[api-batch] request rejected: {e.message}")
        return JSONResponse(
            status_code=e.status_code, content={"error": e.message})
    except Exception as e:
        logger.exception(e)
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to process file: {str(e)}"}
        )
    finally:
        gc.collect()


if __name__ == "__main__":
    load_dotenv()
    # os.environ['MINERU_MODEL_SOURCE'] = "modelscope"
    uvicorn.run(app, host="0.0.0.0", port=8888)
