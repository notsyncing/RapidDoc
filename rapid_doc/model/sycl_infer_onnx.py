"""In-process sycl-infer ONNX sessions.

The document pipeline's layout, OCR, formula and table models run in this
process through ``libsycl_infer_onnx``.  There is no HTTP server and no
subprocess: the tensors are passed to the SYCL executor directly and the
device is the one named by ``engine_cfg["device"]`` (GPU by default).

The C ABI is ``include/sycl_infer/onnx.h`` in the sycl-infer tree.  The
session surface matches the local ONNXRuntime/OpenVINO sessions, so the
handlers do not need to know where the inference runs.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
import struct
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from rapid_doc.utils.gpu_lock import doc_workers, serialized_per_instance

__all__ = [
    "SyclInferError",
    "SyclInferSession",
    "SyclOcrSession",
    "attach_sycl_sessions",
]

_MAGIC_REQ = b"SION"
_MAGIC_RESP = b"SIOR"
_VERSION = 1
_F32, _F16, _I64, _I32, _I8, _U8, _BOOL = range(7)
_DTYPE_TO_CODE = {
    np.dtype(np.float32): _F32,
    np.dtype(np.float16): _F16,
    np.dtype(np.int64): _I64,
    np.dtype(np.int32): _I32,
    np.dtype(np.int8): _I8,
    np.dtype(np.uint8): _U8,
    np.dtype(np.bool_): _BOOL,
}
_CODE_TO_DTYPE = {v: k for k, v in _DTYPE_TO_CODE.items()}


class _DirectInput(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("dtype", ctypes.c_int32),
        ("ndim", ctypes.c_int32),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("data", ctypes.c_void_p),
        ("nbytes", ctypes.c_size_t),
    ]


class _DirectOutput(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("dtype", ctypes.c_int32),
        ("ndim", ctypes.c_int32),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("data", ctypes.c_void_p),
        ("nbytes", ctypes.c_size_t),
    ]


class SyclInferError(RuntimeError):
    pass


_lib = None
_lib_path: Optional[str] = None


def _candidates() -> List[str]:
    out: List[str] = []
    env = os.environ.get("SYCL_INFER_ONNX_LIB")
    if env:
        out.append(env)
    build = os.environ.get("SYCL_INFER_BUILD")
    if build:
        out.append(str(Path(build) / "libsycl_infer_onnx.so.1.0.0"))
    # Sibling checkout used by this workspace; a deployment sets the env var.
    here = Path(__file__).resolve()
    for parent in here.parents:
        for name in ("libsycl_infer_onnx.so.1.0.0", "libsycl_infer_onnx.so.1",
                     "libsycl_infer_onnx.so"):
            for build in (parent / "build-rel2", parent / "sycl-infer" / "build-rel2"):
                p = build / name
                if p.exists():
                    out.append(str(p))
    found = ctypes.util.find_library("sycl_infer_onnx")
    if found:
        out.append(found)
    return out


def load() -> ctypes.CDLL:
    global _lib, _lib_path
    if _lib is not None:
        return _lib
    last: Optional[Exception] = None
    for path in _candidates():
        try:
            _lib = ctypes.CDLL(path)
            _lib_path = path
            break
        except OSError as e:
            last = e
    if _lib is None:
        raise SyclInferError(
            "cannot load libsycl_infer_onnx; build sycl-infer's "
            "sycl_infer_onnx_c target and set SYCL_INFER_ONNX_LIB "
            f"({last})")
    lib = _lib
    lib.syi_onnx_abi_version.restype = ctypes.c_int
    lib.syi_onnx_last_error.restype = ctypes.c_char_p
    lib.syi_onnx_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
                                    ctypes.c_char_p, ctypes.c_int,
                                    ctypes.POINTER(ctypes.c_void_p)]
    lib.syi_onnx_create.restype = ctypes.c_int
    lib.syi_onnx_destroy.argtypes = [ctypes.c_void_p]
    lib.syi_onnx_info_json.argtypes = [ctypes.c_void_p,
                                       ctypes.POINTER(ctypes.c_void_p),
                                       ctypes.POINTER(ctypes.c_size_t)]
    lib.syi_onnx_info_json.restype = ctypes.c_int
    lib.syi_onnx_run_wire.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8),
                                      ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p),
                                      ctypes.POINTER(ctypes.c_size_t)]
    lib.syi_onnx_run_wire.restype = ctypes.c_int
    lib.syi_onnx_run_direct.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(_DirectInput), ctypes.c_size_t,
        ctypes.POINTER(_DirectOutput), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t)]
    lib.syi_onnx_run_direct.restype = ctypes.c_int
    lib.syi_onnx_free.argtypes = [ctypes.c_void_p]
    if hasattr(lib, "syi_onnx_set_output_mode"):
        lib.syi_onnx_set_output_mode.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                                 ctypes.c_char_p]
        lib.syi_onnx_set_output_mode.restype = ctypes.c_int
    return lib


def _last_error() -> str:
    p = load().syi_onnx_last_error()
    return p.decode("utf-8", "replace") if p else ""


def _pack_request(model: str, inputs: Dict[str, np.ndarray]) -> bytes:
    parts: List[bytes] = [_MAGIC_REQ, struct.pack("<I", _VERSION)]
    name = model.encode("utf-8")
    parts.append(struct.pack("<H", len(name)))
    parts.append(name)
    parts.append(struct.pack("<I", len(inputs)))
    for iname, arr in inputs.items():
        arr = np.ascontiguousarray(arr)
        code = _DTYPE_TO_CODE.get(arr.dtype)
        if code is None:
            raise SyclInferError(f"input {iname!r}: unsupported dtype {arr.dtype}")
        bname = iname.encode("utf-8")
        parts.append(struct.pack("<H", len(bname)))
        parts.append(bname)
        parts.append(struct.pack("<BB", code, arr.ndim))
        parts.append(struct.pack("<%dI" % arr.ndim, *arr.shape))
        parts.append(struct.pack("<Q", arr.nbytes))
        parts.append(arr.tobytes())
    return b"".join(parts)


def _unpack_response(raw: bytes):
    if len(raw) < 4 or raw[:4] != _MAGIC_RESP:
        raise SyclInferError(f"not an ONNX reply: {raw[:16]!r}")
    off = 4
    (status,) = struct.unpack_from("<i", raw, off)
    off += 4
    (mlen,) = struct.unpack_from("<I", raw, off)
    off += 4
    msg = raw[off:off + mlen].decode("utf-8", "replace")
    off += mlen
    (n_out,) = struct.unpack_from("<I", raw, off)
    off += 4
    outs = []
    for _ in range(n_out):
        (nlen,) = struct.unpack_from("<H", raw, off)
        off += 2
        name = raw[off:off + nlen].decode("utf-8", "replace")
        off += nlen
        code, ndim = struct.unpack_from("<BB", raw, off)
        off += 2
        dims = struct.unpack_from("<%dI" % ndim, raw, off)
        off += 4 * ndim
        (nbytes,) = struct.unpack_from("<Q", raw, off)
        off += 8
        data = raw[off:off + nbytes]
        off += nbytes
        dtype = _CODE_TO_DTYPE.get(code)
        if dtype is None:
            raise SyclInferError(f"unknown dtype code {code}")
        # See the HTTP client: the decoders modify the arrays in place, and
        # np.frombuffer over an immutable bytes object is read-only.
        arr = np.frombuffer(data, dtype=dtype).copy()
        arr = arr.reshape(dims) if ndim else arr.reshape(())
        outs.append((name, arr))
    return status, msg, outs


def _config_parts(cfg: Any):
    if isinstance(cfg, dict):
        path = cfg.get("model_dir_or_path")
        engine_cfg = dict(cfg.get("engine_cfg") or {})
        model = cfg.get("model") or engine_cfg.get("model") or ""
    else:
        path = getattr(cfg, "model_dir_or_path", None)
        engine_cfg = dict(getattr(cfg, "engine_cfg", None) or {})
        model = engine_cfg.get("model") or ""
    if isinstance(path, dict):
        # Paddle Q-cls builds two sessions; each caller already split the map.
        path = next(iter(path.values()), None)
    return path, engine_cfg, model


class SyclInferSession:
    """One ONNX model, run in this process on the SYCL device."""

    def __init__(self, cfg: Any, *, characters: Optional[List[str]] = None,
                 device: Optional[str] = None) -> None:
        path, engine_cfg, model = _config_parts(cfg)
        if not path:
            raise SyclInferError(
                "the sycl-infer engine needs model_dir_or_path; the local "
                "model path is resolved by the caller")
        self.path = str(path)
        self.model = model or Path(self.path).stem
        self.device = str(device or engine_cfg.get("device") or
                          os.getenv("SYCL_INFER_ONNX_DEVICE") or "GPU").upper()
        drops = engine_cfg.get("drop_outputs") or []
        if isinstance(drops, str):
            drops = [x.strip() for x in drops.split(",") if x.strip()]
        self._drop_outputs = ",".join(str(x) for x in drops)
        self._dnnl = 1 if engine_cfg.get("dnnl", True) else 0
        self._characters = list(characters or [])
        # A zero-copy view is valid until this model runs again.  Every
        # consumer in the pipeline (the CTC decoder, the DB post-process, the
        # layout's mask walk) runs immediately after the call, so the copies
        # - which are 115 MB for a recogniser output and 48 MB/page for a
        # layout mask - are only paid when a caller asks to keep the array.
        # A view is valid only until this model's next run.  With concurrent
        # documents another thread can start that run before the caller has
        # read the view, so the view becomes a use-after-free (it crashed the
        # 12-document run near its end).  Copy when more than one document is
        # in flight; the outputs the pipeline reads are small (the rec result
        # is already the device argmax and the layout mask is reduced to u8).
        self._copy_outputs = bool(engine_cfg.get("copy_outputs", False)) or doc_workers() > 1
        # One call at a time per session; different sessions may overlap.
        self._run_lock = threading.RLock()
        self._pending_modes: List[tuple] = []
        self._last_views: List[Any] = []
        self._handle: Optional[int] = None
        self._info: Dict[str, Any] = {}
        self._create()
        for name, mode in (engine_cfg.get("output_modes") or {}).items():
            self.set_output_mode(name, mode)

    def _create(self) -> None:
        lib = load()
        h = ctypes.c_void_p()
        st = lib.syi_onnx_create(
            self.path.encode(), self.model.encode(), self.device.encode(),
            self._drop_outputs.encode() if self._drop_outputs else None,
            self._dnnl, ctypes.byref(h))
        if st != 0:
            raise SyclInferError(f"cannot load {self.path} on {self.device}: {_last_error()}")
        self._handle = h
        self._info = self._read_info()

    def _read_info(self) -> Dict[str, Any]:
        lib = load()
        p = ctypes.c_void_p()
        n = ctypes.c_size_t()
        if lib.syi_onnx_info_json(self._handle, ctypes.byref(p), ctypes.byref(n)) != 0:
            raise SyclInferError(f"cannot read model info: {_last_error()}")
        try:
            return json.loads(ctypes.string_at(p, n.value).decode("utf-8"))
        finally:
            lib.syi_onnx_free(p)

    def set_output_mode(self, output: str, mode: str) -> None:
        """Reduce an output on the device (see sycl_infer/onnx.h).

        ``output`` may be a graph output's name or its index into
        ``get_output_names()`` (as a string or an int); the pipeline knows
        the layout mask and the recogniser's probability tensor by index, and
        a name that changes with the exporter is a fragile thing to hardcode.
        """
        if isinstance(output, int) or (isinstance(output, str) and output.isdigit()):
            names = self.get_output_names()
            idx = int(output)
            if idx < 0 or idx >= len(names):
                raise SyclInferError(f"output index {idx} out of range for {self.model!r}")
            output = names[idx]
        lib = load()
        if not hasattr(lib, "syi_onnx_set_output_mode"):
            raise SyclInferError("this libsycl_infer_onnx predates output modes")
        st = lib.syi_onnx_set_output_mode(self._handle, str(output).encode(), str(mode).encode())
        if st != 0:
            raise SyclInferError(f"cannot set the output mode of {output!r}: {_last_error()}")

    @serialized_per_instance
    def run(self, inputs: Dict[str, np.ndarray], copy: Optional[bool] = None) -> List[np.ndarray]:
        lib = load()
        # Direct tensor path: no wire pack/decode and no response encode/copy.
        # The output views are model-owned until the next run, so copy them
        # before returning (the decoders want writable arrays).
        if hasattr(lib, "syi_onnx_run_direct"):
            descs = (_DirectInput * len(inputs))()
            names = []
            keep = []
            for i, (name, arr) in enumerate(inputs.items()):
                arr = np.ascontiguousarray(arr)
                code = _DTYPE_TO_CODE.get(arr.dtype)
                if code is None:
                    raise SyclInferError(f"input {name!r}: unsupported dtype {arr.dtype}")
                bname = name.encode()
                names.append(bname)
                shape = (ctypes.c_int64 * arr.ndim)(*arr.shape)
                d = descs[i]
                d.name = bname
                d.dtype = code
                d.ndim = arr.ndim
                d.shape = shape
                d.data = arr.ctypes.data
                d.nbytes = arr.nbytes
                # keep shape/array alive until the call returns
                keep.append((shape, arr))
            outs = (_DirectOutput * 8)()
            n = ctypes.c_size_t(0)
            st = lib.syi_onnx_run_direct(self._handle, descs, len(inputs), outs, 8,
                                         ctypes.byref(n))
            if st != 0:
                raise SyclInferError(_last_error() or "the ONNX run failed")
            result = []
            for i in range(n.value):
                o = outs[i]
                dtype = _CODE_TO_DTYPE.get(o.dtype)
                if dtype is None:
                    raise SyclInferError(f"unknown output dtype code {o.dtype}")
                shape = tuple(int(o.shape[j]) for j in range(o.ndim))
                want_copy = self._copy_outputs if copy is None else copy
                if want_copy or o.nbytes == 0:
                    raw = ctypes.string_at(o.data, o.nbytes)
                    arr = np.frombuffer(bytearray(raw), dtype=dtype).reshape(shape)
                else:
                    # A view of the executor's host buffer: no copy, valid
                    # until this model's next run.  `string_at` + `bytearray`
                    # was two more copies of every output (230 MB per
                    # recogniser call).
                    buf = (ctypes.c_uint8 * o.nbytes).from_address(o.data)
                    arr = np.ctypeslib.as_array(buf).view(dtype).reshape(shape)
                    self._last_views = [buf]
                result.append(arr)
            return result
        body = _pack_request(self.model, inputs)
        req = (ctypes.c_uint8 * len(body)).from_buffer_copy(body)
        out = ctypes.c_void_p()
        out_n = ctypes.c_size_t()
        st = lib.syi_onnx_run_wire(self._handle, req, len(body), ctypes.byref(out),
                                   ctypes.byref(out_n))
        if st != 0:
            raise SyclInferError(_last_error() or "the ONNX run failed")
        try:
            raw = ctypes.string_at(out, out_n.value)
        finally:
            lib.syi_onnx_free(out)
        status, msg, outs2 = _unpack_response(raw)
        if status != 0:
            raise SyclInferError(f"model {self.model!r} failed: {msg}")
        return [a for _, a in outs2]

    def run_one(self, arr: np.ndarray) -> List[np.ndarray]:
        names = self.get_input_names()
        return self.run({names[0] if names else "x": arr})

    def __call__(self, input_content, scale_factor=None, *args, **kwargs):
        if isinstance(input_content, dict):
            return self.run(input_content)
        # The layout detector has three ports (image, scale_factor, im_shape);
        # the other model families take one array.  This is the same assembly
        # the local and HTTP layout sessions use.
        if scale_factor is not None:
            h, w = input_content.shape[-2:]
            # The model's im_shape port is a single [h, w] row (OpenVINO's
            # session passes the same shape); broadcasting it over the batch
            # is what the graph expects.  Passing N rows changes the detector
            # output layout/order for N > 1.
            return self.run({
                "image": input_content,
                "scale_factor": scale_factor,
                "im_shape": np.array([[h, w]], dtype=np.float32),
            })
        return self.run_one(input_content)

    def get_input_names(self) -> List[str]:
        return [x.split(":")[0].strip() for x in self._info.get("inputs", [])]

    def get_output_names(self) -> List[str]:
        return [x.split(":")[0].strip() for x in self._info.get("outputs", [])]

    @property
    def character(self) -> str:
        return self._info.get("character", "") or ""

    @property
    def characters(self):
        """The model's character metadata, in its family's shape.

        The OCR/table metadata is a newline-separated list; the formula
        recogniser stores a JSON object (the tokenizer + token list).  The
        local formula sessions return the parsed object, so do the same when
        the metadata is JSON.
        """
        if self._characters:
            return self._characters
        raw = self.character
        if not raw:
            return []
        if raw.lstrip().startswith(("{", "[")):
            try:
                return json.loads(raw)
            except (TypeError, ValueError):
                pass
        return raw.splitlines()

    def get_character_list(self, key: str = "character"):
        return self.characters

    def have_key(self, key: str = "character") -> bool:
        return bool(self.characters)

    def close(self) -> None:
        if self._handle:
            load().syi_onnx_destroy(self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class SyclOcrSession:
    """rapidocr's session surface: one array in, the first output out."""

    def __init__(self, *args, **kwargs):
        self._session = SyclInferSession(*args, **kwargs)
        self.device = self._session.device

    def __call__(self, input_content: np.ndarray) -> np.ndarray:
        outs = self._session.run_one(input_content)
        return outs[0] if outs else None

    def set_output_mode(self, output, mode: str) -> None:
        self._session.set_output_mode(output, mode)

    def run_argmax(self, input_content: np.ndarray):
        """The recognition tensor's per-timestep argmax and value.

        The graph's output is [B, T, V] of float32 (115 MB for a batch of
        32); the CTC decoder needs ``argmax`` and ``max`` along the last axis
        and nothing else.  Both are computed on the device at readback, so
        only the two [B, T] arrays cross the bus.
        """
        outs = self._session.run_one(input_content)
        if len(outs) != 2:
            raise SyclInferError(
                "the recogniser session is not in argmax output mode "
                f"(got {len(outs)} outputs)")
        return outs[0], outs[1]

    def get_input_names(self) -> List[str]:
        return self._session.get_input_names()

    def get_output_names(self) -> List[str]:
        return self._session.get_output_names()

    def get_character_list(self, key: str = "character") -> List[str]:
        return self._session.get_character_list(key)

    def have_key(self, key: str = "character") -> bool:
        return self._session.have_key(key)

    def close(self) -> None:
        self._session.close()


def attach_sycl_sessions(ocr_engine, det_path: str, rec_path: str, *,
                         device: str = "GPU",
                         det_drops: Optional[List[str]] = None,
                         rec_drops: Optional[List[str]] = None,
                         dnnl: bool = True) -> None:
    """Replace rapidocr's det/rec sessions with the in-process engine.

    The character dictionary is read from the built recogniser first (it has
    already resolved it from the local model), so the ONNX file does not need
    to carry it for the pipeline's decoder.
    """
    characters: List[str] = []
    try:
        characters = list(ocr_engine.text_rec.session.get_character_list())
    except Exception:
        characters = []

    def cfg(path: str, drops: Optional[List[str]], modes=None):
        return {
            "model_dir_or_path": path,
            "engine_cfg": {"device": device, "drop_outputs": drops or [], "dnnl": dnnl,
                           "output_modes": modes or {}},
        }

    ocr_engine.text_det.session = SyclOcrSession(cfg(det_path, det_drops))
    # The recogniser output is [B, T, 18710] float32 and the CTC decoder uses
    # only its per-timestep argmax and max.  Reduce both on the device; see
    # syi_onnx_set_output_mode.  (The output has a single port, so index 0 is
    # the whole answer - the tensor's name is the exporter's.)
    rec_modes = ({"0": "argmax"}
                 if os.getenv("RAPID_DOC_REC_ARGMAX", "1") != "0" else {})
    ocr_engine.text_rec.session = SyclOcrSession(
        cfg(rec_path, rec_drops, rec_modes), characters=characters)

    # Keep the recogniser's padded width constant.  RapidOCR asks for a
    # per-batch max_wh_ratio; when a table cell is extremely wide the width,
    # and therefore the activation, grows with it and a large batch can lose
    # the device.  Normal text is still resized to the same aspect budget.
    try:
        import os as _os
        cap = float(_os.getenv("RAPID_DOC_OCR_MAX_WH_RATIO", "16"))
        rec = ocr_engine.text_rec
        orig_resize = rec.resize_norm_img

        def fixed_resize(img, max_wh_ratio, *a, **kw):
            return orig_resize(img, cap, *a, **kw)

        rec.resize_norm_img = fixed_resize
    except Exception:
        pass
