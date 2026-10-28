import json
from typing import List, Dict, Any

from rapid_doc.model.ocr.ocr_patch import apply_ocr_patch
from rapid_doc.model.ocr.seal_crop import SortPolyBoxes, CropByPolys
from rapid_doc.model.ocr.onnx_ops_session import (
    ops_settings, install_onnx_ops, configure_cuda_onnx,
)

# 应用所有 OCR 相关补丁
apply_ocr_patch()

import os
import cv2
import copy
import time
import warnings
import numpy as np
from pathlib import Path
from loguru import logger
from rapidocr import RapidOCR, EngineType, OCRVersion
from rapidocr.ch_ppocr_rec import TextRecInput, TextRecOutput
from tqdm import tqdm

from rapid_doc.utils.config_reader import get_device, get_env_batch_size
from rapid_doc.utils.model_utils import check_openvino
from rapid_doc.utils.openvino_shapes import (
    BATCH_BUCKETS,
    OCR_REC_HEIGHT,
    det_square_bucket,
    padded_batches,
    pad_det_to_square,
    rec_batch_limit,
    rec_required_width,
    rec_width_bucket,
)
from rapid_doc.utils.ocr_utils import check_img, preprocess_image, sorted_boxes, merge_det_boxes, update_det_boxes, get_rotate_crop_image
from rapidocr.inference_engine.base import InferSession
from importlib.metadata import version
from packaging.version import Version
rapidocr_version = Version(version("rapidocr"))
models_dir = os.getenv('RAPID_MODELS_DIR', None)
models_root_path = Path(models_dir) if models_dir else None
if models_dir:
    # 从指定的文件夹内寻找模型文件
    InferSession.DEFAULT_MODEL_PATH = models_root_path # <v3.8.0
    from rapidocr.ch_ppocr_rec import main as rec_main
    rec_main.DEFAULT_MODEL_PATH = models_root_path

root_dir = os.path.join(Path(__file__).resolve().parent.parent, 'utils')
DEFAULT_SEAL_DEBUG_DIR = os.path.join(
    Path(__file__).resolve().parents[3],
    'output_images',
    'seal_ocr_debug',
)

class RapidOcrModel(object):
    def __init__(self, det_db_box_thresh=0.5, lang=None, ocr_config=None, use_dilation=True, det_db_unclip_ratio=1.8, enable_merge_det_boxes=True, is_seal=False):
        ocr_config = dict(ocr_config or {})
        onnx_ops_settings = ops_settings(ocr_config.pop("onnx_ops", True))
        self.drop_score = 0.5
        self.enable_merge_det_boxes = enable_merge_det_boxes
        self.is_seal = is_seal
        device = get_device()
        self._openvino_gpu = device == 'openvino_gpu'
        # 默认配置
        default_params = {
            "Global.use_cls": False,
            "engine_type": EngineType.ONNXRUNTIME,
            "Det.engine_type": EngineType.ONNXRUNTIME,
            "Rec.engine_type": EngineType.ONNXRUNTIME,
            "Det.ocr_version": OCRVersion.PPOCRV5,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            # "Det.model_type": ModelType.SERVER,
            # "Rec.model_type": ModelType.SERVER,
            "Det.limit_side_len": 960,
            "Det.limit_type": 'max',
            "Det.std": [0.229, 0.224, 0.225],
            "Det.mean": [0.485, 0.456, 0.406],
            "Det.box_thresh": det_db_box_thresh,
            "Det.use_dilation": use_dilation,
            "Det.unclip_ratio": det_db_unclip_ratio,
        }
        if models_root_path and rapidocr_version >= Version("3.8.0"):
            default_params["Global.model_root_dir"] = models_root_path

        # 获取用户传入的 engine_type
        engine_type = ocr_config.get('engine_type') if ocr_config else None
        rapid_doc_dir = Path(os.path.abspath(__file__)).parent.parent.parent
        # CPU 上优先使用 OpenVINO（如果可用且用户未指定 engine_type）
        if device.startswith('cpu') and check_openvino() and not engine_type:
            default_params["Det.engine_type"] = EngineType.OPENVINO
            default_params["Rec.engine_type"] = EngineType.OPENVINO
        # OpenVINO GPU 自动路径（当 MINERU_DEVICE_MODE=openvino_gpu 时）
        if device == 'openvino_gpu' and check_openvino(target_device="GPU") and not engine_type:
            default_params["Det.engine_type"] = EngineType.OPENVINO
            default_params["Rec.engine_type"] = EngineType.OPENVINO
            default_params["Det.device"] = "GPU"
            default_params["Rec.device"] = "GPU"
            default_params["Cls.engine_type"] = EngineType.OPENVINO
            default_params["Cls.device"] = "GPU"
        if engine_type == EngineType.TORCH:
            default_params["Det.engine_type"] = EngineType.TORCH
            default_params["Rec.engine_type"] = EngineType.TORCH

        # 如果传入了 ocr_config，则覆盖参数
        if ocr_config is not None:
            for key, value in ocr_config.items():
                default_params[key] = value

        if device.startswith('cuda'):
            if not engine_type:
                # The plugin opt-in selects ONNX; otherwise keep the Torch default.
                cuda_engine = EngineType.ONNXRUNTIME if onnx_ops_settings else EngineType.TORCH
                for task in ("Det", "Rec"):
                    if not onnx_ops_settings or f"{task}.engine_type" not in ocr_config:
                        default_params[f"{task}.engine_type"] = cuda_engine
            gpu_id = int(device.split(':')[1]) if ':' in device else 0 # GPU 编号
            if default_params.get('Det.engine_type') == EngineType.TORCH:
                default_params['EngineConfig.torch.use_cuda'] = True
                if rapidocr_version <= Version("3.4.2"):
                    default_params['EngineConfig.torch.gpu_id'] = gpu_id
                else:
                    default_params['EngineConfig.torch.cuda_ep_cfg.device_id'] = gpu_id
        elif device.startswith('npu'):
            if not engine_type:
                # npu 环境默认使用 torch
                default_params["Det.engine_type"] = EngineType.TORCH
                default_params["Rec.engine_type"] = EngineType.TORCH
            npu_id = int(device.split(':')[1]) if ':' in device else 0  # npu 编号
            if default_params.get('Det.engine_type') == EngineType.TORCH:
                default_params['EngineConfig.torch.use_npu'] = True
                if rapidocr_version <= Version("3.4.2"):
                    default_params['EngineConfig.torch.npu_id'] = npu_id
                else:
                    default_params['EngineConfig.torch.npu_ep_cfg.device_id'] = npu_id
        default_params.pop('engine_type', None)
        default_params.pop('use_det_mode', None)
        default_params.pop('custom_model', None)
        default_params.pop('seal_enable', None)

        configure_cuda_onnx(default_params, device, rapidocr_version)

        if default_params["Det.engine_type"] in [EngineType.ONNXRUNTIME, EngineType.OPENVINO]:
            if not default_params.get("Det.model_path"):
                v6_det_small_path = os.path.join(rapid_doc_dir, 'resources', 'PP-OCRv6_det_small.onnx')
                default_params["Det.model_path"] = v6_det_small_path
            if not default_params.get("Rec.model_path"):
                v6_rec_small_path = os.path.join(rapid_doc_dir, 'resources', 'PP-OCRv6_rec_small.onnx')
                default_params["Rec.model_path"] = v6_rec_small_path
        elif default_params["Det.engine_type"] in [EngineType.TORCH]:
            if not default_params.get("Det.model_path"):
                v6_det_small_path = os.path.join(rapid_doc_dir, 'resources', 'PP-OCRv6_det_small.safetensors')
                default_params["Det.model_path"] = v6_det_small_path
            if not default_params.get("Rec.model_path"):
                v6_rec_small_path = os.path.join(rapid_doc_dir, 'resources', 'PP-OCRv6_rec_small.safetensors')
                default_params["Rec.model_path"] = v6_rec_small_path
        if not default_params.get("Rec.rec_keys_path"):
            v6_dict_path = os.path.join(rapid_doc_dir, 'resources', 'ppocrv6_dict.txt')
            default_params["Rec.rec_keys_path"] = v6_dict_path

        if self.is_seal:
            # 印章识别参数
            default_params['Det.limit_side_len'] = 736
            default_params['Det.limit_type'] = 'min'
            default_params['Det.max_side_limit'] = 4000
            default_params['Det.thresh'] = 0.2
            default_params['Det.box_thresh'] = 0.6
            default_params['Det.unclip_ratio'] = 0.5
            default_params['Det.box_type'] = 'poly'
            default_params['Det.use_dilation'] = False
            default_params['Det.ocr_version'] = OCRVersion.PPOCRV4
            # default_params['Rec.ocr_version'] = OCRVersion.PPOCRV4 #印章使用v4的rec模型效果更好
            if device == 'openvino_gpu' and check_openvino(target_device="GPU") and not engine_type:
                default_params["Det.engine_type"] = EngineType.OPENVINO
                default_params["Rec.engine_type"] = EngineType.OPENVINO
                default_params["Det.device"] = "GPU"
                default_params["Rec.device"] = "GPU"
                default_params["Cls.engine_type"] = EngineType.OPENVINO
                default_params["Cls.device"] = "GPU"
            elif device.startswith('cpu') and check_openvino() and not engine_type:
                default_params["Det.engine_type"] = EngineType.OPENVINO
                default_params["Rec.engine_type"] = EngineType.OPENVINO
            else:
                default_params["Det.engine_type"] = EngineType.ONNXRUNTIME
                default_params["Rec.engine_type"] = EngineType.ONNXRUNTIME

            seal_det_model_path = os.path.join(rapid_doc_dir, 'resources', 'pp-ocrv4_mobile_seal_det.onnx')
            default_params['Det.model_path'] = seal_det_model_path
            if default_params["Rec.engine_type"] in [EngineType.ONNXRUNTIME, EngineType.OPENVINO]:
                v6_rec_small_path = os.path.join(rapid_doc_dir, 'resources', 'PP-OCRv6_rec_small.onnx')
                default_params["Rec.model_path"] = v6_rec_small_path
            elif default_params["Rec.engine_type"] == EngineType.TORCH:
                v6_rec_small_path = os.path.join(rapid_doc_dir, 'resources', 'PP-OCRv6_rec_small.safetensors')
                default_params["Rec.model_path"] = v6_rec_small_path
                v6_dict_path = os.path.join(rapid_doc_dir, 'resources', 'ppocrv6_dict.txt')
                default_params["Rec.rec_keys_path"] = v6_dict_path
            self.enable_merge_det_boxes = False

        self.ocr_engine = RapidOCR(params=default_params)
        # RapidOCR 3.10+ loads adapters lazily. This wrapper calls Det/Rec
        # directly, so load both before replacing sessions or reading metadata.
        for task in ("det", "rec"):
            load_model = getattr(self.ocr_engine, f"_load_{task}_model", None)
            if callable(load_model) and getattr(self.ocr_engine, f"text_{task}", None) is None:
                load_model()
        self.onnx_ops_reports = install_onnx_ops(self.ocr_engine, onnx_ops_settings)
        if self.onnx_ops_reports:
            for task, report in self.onnx_ops_reports.items():
                logger.info("OCR {} ONNX ops: {}", task, report)
        self.text_detector = self.ocr_engine.text_det
        self.text_recognizer = self.ocr_engine.text_rec
        self.rec_batch_num = self.text_recognizer.rec_batch_num

        if self.is_seal:
            self._seal_sort_boxes = SortPolyBoxes()
            self._seal_crop_by_polys = CropByPolys(det_box_type='poly')
            self._seal_debug_counter = 0
            self._seal_debug_dir = self._resolve_seal_debug_dir()

    def _resolve_seal_debug_dir(self):
        if not self.is_seal:
            return None

        debug_dir = os.getenv("MINERU_SEAL_OCR_DEBUG_DIR")
        if debug_dir:
            return debug_dir

        debug_enable = os.getenv("MINERU_SEAL_OCR_DEBUG", "").lower()
        if debug_enable in {"1", "true", "yes", "on"}:
            return DEFAULT_SEAL_DEBUG_DIR

        return None

    def _dump_seal_debug_artifacts(self, input_image, dt_boxes, img_crop_list, rec_res=None):
        if not self._seal_debug_dir:
            return

        sample_dir = os.path.join(
            self._seal_debug_dir,
            f"sample_{self._seal_debug_counter:04d}",
        )
        self._seal_debug_counter += 1
        os.makedirs(sample_dir, exist_ok=True)

        cv2.imwrite(os.path.join(sample_dir, "input.png"), input_image)

        det_vis = input_image.copy()
        for index, box in enumerate(dt_boxes or []):
            points = np.asarray(box, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(det_vis, [points], isClosed=True, color=(0, 0, 255), thickness=2)
            anchor = tuple(np.asarray(box[0], dtype=np.int32).tolist())
            cv2.putText(
                det_vis,
                str(index),
                anchor,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
        cv2.imwrite(os.path.join(sample_dir, "det_vis.png"), det_vis)

        records = []
        for index, crop_img in enumerate(img_crop_list or []):
            crop_name = f"crop_{index:02d}.png"
            cv2.imwrite(os.path.join(sample_dir, crop_name), crop_img)
            record = {
                "index": index,
                "crop_path": crop_name,
            }
            if rec_res is not None and index < len(rec_res):
                text, score = rec_res[index]
                record["text"] = text
                record["score"] = float(score)
            records.append(record)

        with open(os.path.join(sample_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

    def ocr(self,
            img,
            det=True,
            rec=True,
            mfd_res=None,
            tqdm_enable=False,
            tqdm_desc="OCR-rec Predict",
            return_word_box=False,
            ori_img=None,
            dt_boxes=None,
            ):
        assert isinstance(img, (np.ndarray, list, str, bytes))
        if isinstance(img, list) and det == True:
            logger.error('When input a list of images, det must be false')
            exit(0)
        img = check_img(img)
        imgs = [img]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if det and rec:
                ocr_res = []
                for img in imgs:
                    img = preprocess_image(img)
                    dt_boxes, rec_res = self.__call__(img, mfd_res=mfd_res)
                    if not dt_boxes and not rec_res:
                        ocr_res.append(None)
                        continue
                    tmp_res = [[box.tolist(), res] for box, res in zip(dt_boxes, rec_res)]
                    ocr_res.append(tmp_res)
                # print(f"ocr===运行时间1: {time.time() - start_time}秒")
                return ocr_res
            elif det and not rec:
                ocr_res = []
                for img in imgs:
                    img = preprocess_image(img)
                    det_res = self.text_detector(img)
                    dt_boxes, elapse = det_res.boxes, det_res.elapse
                    # logger.debug("dt_boxes num : {}, elapse : {}".format(len(dt_boxes), elapse))
                    if dt_boxes is None:
                        ocr_res.append(None)
                        continue
                    if self.is_seal:
                        dt_boxes = self._seal_sort_boxes(dt_boxes)
                        img_crop_list = self._seal_crop_by_polys(img, dt_boxes)
                        self._dump_seal_debug_artifacts(img, dt_boxes, img_crop_list)
                    else:
                        dt_boxes = np.array(dt_boxes) # 转为np
                        dt_boxes = sorted_boxes(dt_boxes)
                        # merge_det_boxes 和 update_det_boxes 都会把poly转成bbox再转回poly，因此需要过滤所有倾斜程度较大的文本框
                        if self.enable_merge_det_boxes:
                            dt_boxes = merge_det_boxes(dt_boxes)
                        if mfd_res:
                            dt_boxes = update_det_boxes(dt_boxes, mfd_res)
                    tmp_res = [box.tolist() for box in dt_boxes]
                    ocr_res.append(tmp_res)
                # print(f"ocr===运行时间222: {time.time() - start_time}秒")
                return ocr_res
            elif not det and rec:
                ocr_res = []
                for img in imgs:
                    if not isinstance(img, list):
                        img = preprocess_image(img)
                        img = [img]
                    rec_input = TextRecInput(img=img, return_word_box=return_word_box)
                    rec_result = self.text_recognizer_call(rec_input, tqdm_enable=tqdm_enable, tqdm_desc=tqdm_desc)
                    if return_word_box and ori_img is not None and dt_boxes:
                        op_record = {'padding_1': {'left': 0, 'top': 0}, 'preprocess': {'ratio_h': 1.0, 'ratio_w': 1.0}}
                        raw_h, raw_w = ori_img.shape[:2]
                        dt_boxes_np = [np.array(box, dtype=np.float32) for box in dt_boxes]
                        word_results = self.calc_word_boxes(img, dt_boxes_np, rec_result, op_record, raw_h, raw_w)
                        rec_res = list(zip(rec_result.txts, rec_result.scores, word_results))
                    else:
                        rec_res = list(zip(rec_result.txts, rec_result.scores))
                    ocr_res.append(rec_res)
                return ocr_res

    def calc_word_boxes(
        self,
        img: List[np.ndarray],
        dt_boxes: np.ndarray,
        rec_res: TextRecOutput,
        op_record: Dict[str, Any],
        raw_h: int,
        raw_w: int,
    ) -> Any:
        rec_res = self.ocr_engine.cal_rec_boxes(
            img, dt_boxes, rec_res, self.ocr_engine.return_single_char_box
        )

        origin_words = []
        for word_line in rec_res.word_results:
            origin_words_item = []
            for txt, score, bbox in word_line:
                if bbox is None:
                    continue

                origin_words_points = self.map_boxes_to_original(
                    np.array([bbox]).astype(np.float64), op_record, raw_h, raw_w
                )
                origin_words_points = origin_words_points.astype(np.int32).tolist()[0]
                origin_words_item.append((txt, score, origin_words_points))

            if origin_words_item:
                origin_words.append(tuple(origin_words_item))
        return tuple(origin_words)

    def map_boxes_to_original(
            self, dt_boxes: np.ndarray, op_record: Dict[str, Any], ori_h: int, ori_w: int
    ) -> np.ndarray:
        for op in reversed(list(op_record.keys())):
            v = op_record[op]
            if "padding" in op:
                top, left = v.get("top"), v.get("left")
                dt_boxes[:, :, 0] -= left
                dt_boxes[:, :, 1] -= top
            elif "preprocess" in op:
                ratio_h = v.get("ratio_h")
                ratio_w = v.get("ratio_w")
                dt_boxes[:, :, 0] *= ratio_w
                dt_boxes[:, :, 1] *= ratio_h

        dt_boxes = np.where(dt_boxes < 0, 0, dt_boxes)
        dt_boxes[..., 0] = np.where(dt_boxes[..., 0] > ori_w, ori_w, dt_boxes[..., 0])
        dt_boxes[..., 1] = np.where(dt_boxes[..., 1] > ori_h, ori_h, dt_boxes[..., 1])
        return dt_boxes

    def __call__(self, img, mfd_res=None):

        if img is None:
            logger.debug("no valid image provided")
            return None, None

        ori_im = img.copy()
        # Full-OCR paths (numeric-cell repair, fallback table OCR, seal OCR)
        # must see the same fixed detector canvas as the main pipeline.
        # Otherwise this hidden path feeds the GPU plugin new H/W shapes and
        # bypasses the recognition shape fix below.
        det_img = img
        det_scale = 1.0
        if self._openvino_gpu:
            canvas = det_square_bucket(max(img.shape[0], img.shape[1]))
            det_img, det_scale = pad_det_to_square(img, canvas)
        det_res = self.text_detector(det_img)
        dt_boxes, elapse = det_res.boxes, det_res.elapse
        if dt_boxes is not None and len(dt_boxes) > 0 and det_scale != 1.0:
            dt_boxes = (np.asarray(dt_boxes, dtype=np.float32) / det_scale)

        if dt_boxes is None:
            logger.debug("no dt_boxes found, elapsed : {}".format(elapse))
            return None, None
        else:
            pass
            # logger.debug("dt_boxes num : {}, elapsed : {}".format(len(dt_boxes), elapse))
        if self.is_seal:
            dt_boxes = self._seal_sort_boxes(dt_boxes)
            img_crop_list = self._seal_crop_by_polys(ori_im, dt_boxes)
        else:
            img_crop_list = []

            dt_boxes = sorted_boxes(dt_boxes)

            # merge_det_boxes 和 update_det_boxes 都会把poly转成bbox再转回poly，因此需要过滤所有倾斜程度较大的文本框
            if self.enable_merge_det_boxes:
                dt_boxes = merge_det_boxes(dt_boxes)

            if mfd_res:
                dt_boxes = update_det_boxes(dt_boxes, mfd_res)

            for bno in range(len(dt_boxes)):
                tmp_box = copy.deepcopy(dt_boxes[bno])
                img_crop = get_rotate_crop_image(ori_im, tmp_box)
                img_crop_list.append(img_crop)

        rec_result = self.text_recognizer_call(TextRecInput(img_crop_list))
        rec_res = list(zip(rec_result.txts, rec_result.scores))
        elapse = rec_result.elapse
        # logger.debug("rec_res num  : {}, elapsed : {}".format(len(rec_res), elapse))
        if self.is_seal:
            self._dump_seal_debug_artifacts(ori_im, dt_boxes, img_crop_list, rec_res)

        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)

        return filter_boxes, filter_rec_res


    def text_recognizer_call(self, args: TextRecInput, tqdm_enable=False, tqdm_desc="OCR-rec Predict") -> TextRecOutput:
        """Recognise crops on bucketed static shapes.

        Crops are sorted by aspect ratio, so every batch contains similar
        lines.  The batch size is the next bucket (1/2/4/8/...), and the
        recognition width is the smallest bucket that fits the batch's widest
        line.  Nothing is padded to a global maximum.
        """
        img_list = [args.img] if isinstance(args.img, np.ndarray) else list(args.img)
        return_word_box = args.return_word_box
        real_img_num = len(img_list)
        if real_img_num == 0:
            return TextRecOutput([], [], [], [], 0)

        # Required width before bucketing, used both for sorting and for
        # grouping crops of similar aspect ratio.
        required_widths = np.asarray(
            [rec_required_width([img]) for img in img_list], dtype=np.int32
        )
        indices = np.argsort(required_widths, kind="stable")  # narrow -> wide

        max_batch = get_env_batch_size(
            "RAPID_DOC_OCR_REC_BATCH_SIZE", 64
        ) or 64
        try:
            vocab = len(self.text_recognizer.postprocess_op.character)
        except AttributeError:
            vocab = 20000
        rec_res = [("", 0.0)] * real_img_num
        elapse = 0.0
        session = self.text_recognizer.session

        # Split the sorted crops into width-bucket regions.  Narrow lines use
        # a large batch; very wide lines use a smaller batch so the
        # [batch, T, vocab] output stays bounded.
        groups = []
        pos = 0
        while pos < real_img_num:
            width_bucket = rec_width_bucket(int(required_widths[indices[pos]]))
            end = pos + 1
            while (
                end < real_img_num
                and rec_width_bucket(int(required_widths[indices[end]]))
                == width_bucket
            ):
                end += 1
            groups.append((indices[pos:end], width_bucket))
            pos = end

        with tqdm(total=real_img_num, desc=tqdm_desc, disable=not tqdm_enable) as pbar:
            for group_indices, width_bucket in groups:
                batch_limit = min(
                    max_batch, rec_batch_limit(width_bucket, vocab)
                )
                for chunk, bucket, real_count in padded_batches(
                    list(group_indices), batch_limit, BATCH_BUCKETS
                ):
                    real_indices = chunk[:real_count]
                    real_images = [img_list[i] for i in real_indices]
                    # The CTC word-box mapping multiplies token positions by
                    # wh_ratio / max_wh_ratio.  Passing the bucket ratio keeps
                    # the mapping correct even though the tensor is wider than
                    # the widest real crop.
                    max_wh_ratio = width_bucket / float(OCR_REC_HEIGHT)

                    padded_images = [img_list[i] for i in chunk]
                    wh_ratio_list = [
                        img.shape[1] * 1.0 / float(img.shape[0])
                        for img in padded_images
                    ]
                    norm_img_batch = []
                    for img in padded_images:
                        norm_img = self.text_recognizer.resize_norm_img(
                            img, max_wh_ratio
                        )
                        if norm_img.shape[2] < width_bucket:
                            norm_img = np.pad(
                                norm_img,
                                ((0, 0), (0, 0),
                                 (0, width_bucket - norm_img.shape[2])),
                                mode="constant",
                            )
                        elif norm_img.shape[2] > width_bucket:
                            norm_img = norm_img[:, :, :width_bucket]
                        norm_img_batch.append(norm_img[np.newaxis, :])
                    norm_img_batch = np.concatenate(norm_img_batch).astype(
                        np.float32, copy=False
                    )

                    started = time.perf_counter()
                    if hasattr(session, "run_argmax"):
                        # sycl-infer can reduce the probability tensor on device.
                        preds_idx, preds_prob = session.run_argmax(norm_img_batch)
                        line_results, word_results = (
                            self.text_recognizer.postprocess_op.decode(
                                preds_idx,
                                preds_prob,
                                return_word_box,
                                wh_ratio_list=wh_ratio_list,
                                max_wh_ratio=max_wh_ratio,
                                remove_duplicate=True,
                            )
                        )
                    else:
                        preds = session(norm_img_batch)
                        line_results, word_results = (
                            self.text_recognizer.postprocess_op(
                                preds,
                                return_word_box,
                                wh_ratio_list=wh_ratio_list,
                                max_wh_ratio=max_wh_ratio,
                            )
                        )

                    for rno in range(min(real_count, len(line_results))):
                        one_res = line_results[rno]
                        if return_word_box:
                            rec_res[real_indices[rno]] = (one_res, word_results[rno])
                        else:
                            rec_res[real_indices[rno]] = (one_res, None)
                    elapse += time.perf_counter() - started
                    pbar.update(real_count)

        all_line_results, all_word_results = list(zip(*rec_res))
        txts, scores = list(zip(*all_line_results))
        return TextRecOutput(
            img_list,
            txts,
            scores,
            all_word_results,
            elapse,
        )

    def det_batch_predict(self, img_list, max_batch_size=8):
        """
        批处理预测方法，支持多张图像同时检测

        Args:
            img_list: 图像列表
            max_batch_size: 最大批处理大小

        Returns:
            batch_results: 批处理结果列表，每个元素为(dt_boxes, elapse)
        """
        if not img_list:
            return []

        batch_results = []

        # 分批处理
        for i in range(0, len(img_list), max_batch_size):
            batch_imgs = img_list[i:i + max_batch_size]
            # assert尺寸一致
            batch_dt_boxes, batch_elapse = self._batch_process_same_size(batch_imgs)
            batch_results.extend(batch_dt_boxes)

        return batch_results


    def _batch_process_same_size(self, img_list):
        """
            对相同尺寸的图像进行批处理

            Args:
                img_list: 相同尺寸的图像列表

            Returns:
                batch_results: 批处理结果列表
                total_elapse: 总耗时
            """
        starttime = time.time()
        # 1、前置处理：预分配 batch buffer 直接填充，
        # 避免 prepro_img_list + concatenate 两份 f32 副本同时驻留内存
        # （大 batch 下可达数 GB，iGPU 共享内存环境易触发系统 OOM）。
        if not img_list:
            return [], 0

        # 同一批图像尺寸一致，preprocess 的 resize 目标与 limit_side_len 相同，
        # 只需创建一次 DetPreProcess 并统一完成 resize + 归一化 + 转置。
        img_inputs = None
        prepro_shape = None
        preprocess_op = None

        for i, img in enumerate(img_list):
            ori_img_shape = img.shape[0], img.shape[1]
            if preprocess_op is None:
                preprocess_op = self.text_detector.get_preprocess(max(img.shape[0], img.shape[1]))
            resized_img = preprocess_op.resize(img)
            if resized_img is None:
                return [(None, 0) for _ in img_list], 0
            if img_inputs is None:
                prepro_shape = (3,) + resized_img.shape[:2]
                img_inputs = np.empty((len(img_list),) + prepro_shape, dtype=np.float32)
            elif resized_img.shape[:2] != img_inputs.shape[2:]:
                raise ValueError(
                    f'inconsistent preprocessed shapes: {img_inputs.shape[2:]} '
                    f'vs {resized_img.shape[:2]}'
                )

            # 融合归一化 + 转置，直接写入预分配 buffer：
            # 与原实现 `(resized.astype(f32) * scale - mean) / std` 运算顺序一致，
            # 但用 in-place 运算省去 3 份临时数组与一次 extra copy。
            out = resized_img.astype(np.float32)
            out *= preprocess_op.scale
            out -= preprocess_op.mean
            out /= preprocess_op.std
            img_inputs[i] = out.transpose(2, 0, 1)

        # 2、批处理推理
        batch_preds = self.text_detector.session(img_inputs)

        # 3、后处理每个图像的结果
        batch_results = []
        total_elapse = time.time() - starttime

        for i in range(len(img_list)):
            # 提取单个图像的预测结果
            single_preds = batch_preds[i:i+1]
            boxes, scores = self.text_detector.postprocess_op(single_preds, ori_img_shape)
            boxes = self.text_detector.sorted_boxes(boxes)
            batch_results.append((boxes, total_elapse / len(img_list)))
        return batch_results, total_elapse


if __name__ == '__main__':
    from rapidocr import EngineType as OCREngineType, OCRVersion, ModelType as OCRModelType
    ocr_config = {
        # 新增的自定义参数
        "engine_type": OCREngineType.ONNXRUNTIME, # 统一设置推理引擎 总运行时间: 5.039876222610474秒
        # "engine_type": OCREngineType.OPENVINO,  # 统一设置推理引擎 总运行时间: 4.724567413330078秒
        # "engine_type": OCREngineType.TORCH,  # 统一设置推理引擎 总运行时间: 4.724567413330078秒
    }
    rapid_ocr = RapidOcrModel(ocr_config=ocr_config)
    # rapid_ocr = RapidOcrModel()
    img = cv2.imread("reader_order_01.png")
    start_time1 = time.time()
    for i in range(1):
        start_time = time.time()
        ocr_res = rapid_ocr.ocr(img=img, det=True, rec=True)
        print(f"运行时间: {time.time() - start_time}秒")
    print(f"总运行时间: {time.time() - start_time1}秒")

    # rapid_ocr = RapidOcrModel()
    # img = cv2.imread("C:\\ocr\\img\\aaa.png")
    # start_time = time.time()
    # # rec 慢
    # ocr_res = rapid_ocr.ocr(img=img, det=True, rec=True)
    # print(ocr_res)
    # print(f"运行时间: {time.time() - start_time}秒")
