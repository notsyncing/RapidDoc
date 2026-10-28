# Copyright (c) RapidAI. All rights reserved.
"""
批量分析模块
"""
import copy
import os
import re
from typing import List, Dict, Tuple
from collections import defaultdict
from functools import lru_cache

import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm

from .model_init import AtomModelSingleton
from .model_list import AtomicModel
from ...model.table.utils import normalize_table_ocr_text
from ...model.table.rule_table import build_rule_table, build_track_table
from ...model.docvortex_adapter import recover_docvortex_table, use_docvortex_table_rules
from ...model.table.utils import normalize_table_html_cell_text
from ...utils.bbox_utils import normalize_to_int_bbox
from ...utils.boxbase import rotate_table_image
from ...utils.config_reader import get_env_batch_size
from ...utils.enum_class import CategoryId
from ...utils.model_utils import crop_img
from ...utils.openvino_shapes import (
    DET_MAX_BATCH,
    DET_MAX_SIDE,
    batch_bucket,
    det_canvas_shape,
    padded_batches,
    pad_det_to_shape,
    pad_det_to_square,
)
from ...utils.ocr_utils import (
    merge_det_boxes, update_det_boxes, sorted_boxes, get_rotate_crop_image,
    get_adjusted_mfdetrec_res, get_ocr_result_list, OcrConfidence, get_ocr_result_list_table
)
from ...utils.span_pre_proc import (
    txt_spans_extract, txt_spans_bbox_extract,
    txt_most_angle_extract_table, extract_table_fill_image
)


# =================================== OCR-det ===================================
def _extract_text_from_pdf(
        ocr_res_all_page: List[Dict],
        pdf_dict_list: List[Dict],
        scale_list: List[float]
):
    """从 PDF 中提取文本框"""
    ocr_res_grouped = {}
    for x in ocr_res_all_page:
        ocr_res_grouped.setdefault(x["page_idx"], []).append(x)

    total_texts = sum(len(texts) for texts in ocr_res_grouped.values())

    with tqdm(total=total_texts, desc="PDF-det Predict") as pbar:
        for page_idx, text_list in ocr_res_grouped.items():
            page_dict = pdf_dict_list[page_idx] if text_list else {}
            scale = scale_list[page_idx] if text_list else 1.0

            for ocr_res_dict in text_list:
                if ocr_res_dict['ocr_enable']:
                    continue
                if page_dict.get("rotate_label") in ["90", "180", "270"]:
                    ocr_res_dict['ocr_enable'] = True
                    continue

                for res in ocr_res_dict['ocr_res_list']:
                    if res.get('native_text'):
                        ocr_res_dict['layout_res'].append({
                            'poly': res['poly'],
                            'category_id': CategoryId.OcrText,
                            'score': 1.0,
                            'text': res['native_text'],
                            'docvortex_native_text': True,
                            'original_label': res.get('original_label'),
                            'original_order': res.get('original_order'),
                        })
                        continue
                    new_image, useful_list = crop_img(
                        res, ocr_res_dict['np_img'], crop_paste_x=50, crop_paste_y=50
                    )

                    adjusted_mfdetrec_res = get_adjusted_mfdetrec_res(
                        ocr_res_dict['single_page_mfdetrec_res'] + ocr_res_dict['checkbox_res'],
                        useful_list
                    )

                    bgr_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
                    ocr_res = txt_spans_bbox_extract(
                        page_dict, res, mfd_res=adjusted_mfdetrec_res,
                        scale=scale, useful_list=useful_list
                    )

                    if ocr_res:
                        ocr_result_list = get_ocr_result_list(
                            ocr_res, useful_list, ocr_res_dict['ocr_enable'],
                            bgr_image, ocr_res_dict['lang'],
                            res['original_label'], res['original_order']
                        )
                        ocr_res_dict['layout_res'].extend(ocr_result_list)

                pbar.update(1)

def _apply_mask_boxes_to_image(
    bgr_image: np.ndarray,
    mask_boxes: list[dict] | None,
) -> np.ndarray:
    if not mask_boxes:
        return bgr_image

    masked_image = bgr_image.copy()
    image_h, image_w = masked_image.shape[:2]
    for mask_box in mask_boxes:
        bbox = mask_box.get("bbox")
        if bbox is None:
            continue

        int_bbox = normalize_to_int_bbox(bbox, image_size=(image_h, image_w))
        if int_bbox is None:
            continue

        x0, y0, x1, y1 = int_bbox
        masked_image[y0:y1, x0:x1] = 255

    return masked_image


_DET_MEM_BUDGET_MIN = 256 * 1024 ** 2  # 256 MB
_DET_MEM_BUDGET_MAX = 512 * 1024 ** 2  # 512 MB：单批 f32 输入预算


@lru_cache(maxsize=1)
def _gpu_det_mem_budget() -> int:
    """OCR-det 单次推理输入张量的显存预算（字节）。

    Intel GPU 即使总显存很大，单块 buffer 分配也存在上限
    （GPU_DEVICE_MAX_ALLOC_MEM_SIZE，常见 4GB）；同时 det 输入只是整条
    GPU 管线（布局/表格/公式/rec + THROUGHPUT 多请求）的一部分，预算
    需同时受单块上限和总显存比例约束，避免吃满设备显存导致进程 OOM。
    iGPU 共享内存下，f32 输入还会在系统内存里再占一份（预分配 buffer）
    且 GPU 拷贝同样落在系统内存，加上首批 JIT 编译时的激活池峰值，
    预算上限保持 512MB 保守。可用环境变量 RAPID_DOC_DET_MEM_BUDGET_MB
    覆盖（单位 MB）。

    GPU 属性进程内不变，用 lru_cache 保证只查询一次。
    """
    hard_max = _DET_MEM_BUDGET_MAX
    env_mb = os.getenv("RAPID_DOC_DET_MEM_BUDGET_MB")
    if env_mb:
        try:
            hard_max = max(int(env_mb) * 1024 ** 2, _DET_MEM_BUDGET_MIN)
        except ValueError:
            pass
    try:
        from rapid_doc.utils.openvino_core import get_openvino_core
        core = get_openvino_core()
        if 'GPU' in core.available_devices:
            total_mem = int(core.get_property('GPU', 'GPU_DEVICE_TOTAL_MEM_SIZE'))
            budget = int(total_mem * 0.15)
            try:
                max_alloc = int(core.get_property('GPU', 'GPU_DEVICE_MAX_ALLOC_MEM_SIZE'))
                budget = min(budget, int(max_alloc * 0.6))
            except Exception:
                # 该属性在部分 OpenVINO 版本/驱动上不存在，降级为总显存比例预算
                logger.warning(
                    'Failed to query GPU_DEVICE_MAX_ALLOC_MEM_SIZE, '
                    'fall back to total-memory ratio budget'
                )
            return min(max(budget, _DET_MEM_BUDGET_MIN), hard_max)
    except Exception:
        # openvino 未安装 / GPU 不可用 / 查询失败，降级为默认预算
        logger.warning(
            f'Failed to query GPU memory, use default det budget '
            f'{hard_max / (1024 ** 3):.1f}GB'
        )
    return hard_max


def _gpu_safe_det_batch_size(max_h: int, max_w: int, requested: int) -> int:
    """按显存预算限制 OCR-det 的 batch 大小，防止输入张量
    （NCHW f32：batch * 3 * H * W * 4 字节）超过设备单次分配上限。"""
    if requested <= 1:
        return requested
    bytes_per_img = max_h * max_w * 3 * 4
    budget = _gpu_det_mem_budget()
    safe = max(1, budget // max(bytes_per_img, 1))
    return min(requested, safe)

def _run_ocr_det_batch(
        ocr_res_all_page: List[Dict],
        atom_model_manager: AtomModelSingleton,
        ocr_config
):
    """批量 OCR 检测"""
    # OCR 配置
    use_det_mode = ocr_config.get("use_det_mode", "auto")
    ocr_det_base_batch_size = get_env_batch_size(
        'RAPID_DOC_OCR_DET_BATCH_SIZE', ocr_config.get("Det.rec_batch_num", 1)
    )

    # 按语言直接分组，避免额外的全局列表持有 bgr_image 引用，
    # 大文档下全量 crop 可达数 GB，处理完一批即释放。
    lang_groups = defaultdict(list)

    for ocr_res_dict in ocr_res_all_page:
        for res in ocr_res_dict['ocr_res_list']:
            ocr_enable = ocr_res_dict['ocr_enable']

            if not ocr_res_dict['ocr_enable']:
                if res.get('need_ocr_det'):
                    ocr_enable = True
                elif use_det_mode == 'txt' or (use_det_mode != 'ocr' and not res.get('need_ocr_det')):
                    continue

            res.pop('need_ocr_det', None)

            new_image, useful_list = crop_img(
                res, ocr_res_dict['np_img'], crop_paste_x=50, crop_paste_y=50
            )

            adjusted_mfdetrec_res = get_adjusted_mfdetrec_res(
                ocr_res_dict['single_page_mfdetrec_res'] + ocr_res_dict['checkbox_res'],
                useful_list
            )

            bgr_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)

            # NOTE: 不在此生成 det_image（mask 拷贝），改为在推理时按批
            # 即时生成，避免大文档下全量 crop 的额外拷贝驻留系统内存。
            lang_groups[ocr_res_dict['lang']].append((
                bgr_image, None, useful_list, ocr_res_dict,
                adjusted_mfdetrec_res, ocr_res_dict['lang'], res, ocr_enable
            ))

    if not lang_groups:
        return

    RESOLUTION_GROUP_STRIDE = 64

    for lang, lang_crop_list in lang_groups.items():
        if not lang_crop_list:
            continue

        ocr_model = atom_model_manager.get_atom_model(
            atom_model_name=AtomicModel.OCR,
            lang=lang,
            ocr_config=ocr_config,
        )

        # GPU 模式：按本文档实际裁剪尺寸选择最近的矩形 canvas（128 对齐，
        # 长边不超过检测器自身的 960 上限），batch 也取最近的桶。这样既不会
        # 让 (batch, H, W) 在请求之间漂移，也不会像固定 960x960 / batch 8
        # 那样为每个 crop 支付最大尺寸的计算量。
        device = ocr_config.get("Det.device", "cpu")
        if ocr_config.get("sycl_infer") or "gpu" in device.lower() or "openvino" in device.lower():
            max_h = max_w = 0
            for info in lang_crop_list:
                bgr_image = info[0]
                max_h = max(max_h, bgr_image.shape[0])
                max_w = max(max_w, bgr_image.shape[1])
            max_side = max(max_h, max_w)
            doc_scale = min(1.0, float(DET_MAX_SIDE) / max_side) if max_side else 1.0
            canvas_h, canvas_w = det_canvas_shape(
                max(1, int(round(max_h * doc_scale))),
                max(1, int(round(max_w * doc_scale))),
            )
            max_batch = min(
                int(ocr_det_base_batch_size or DET_MAX_BATCH), DET_MAX_BATCH
            )
            fixed_batch = batch_bucket(
                min(len(lang_crop_list), max_batch), max_batch
            )

            for start in tqdm(range(0, len(lang_crop_list), fixed_batch), desc=f"OCR-det {lang}"):
                real_batch = lang_crop_list[start:start + fixed_batch]
                real_count = len(real_batch)
                batch_info = list(real_batch)
                if len(batch_info) < fixed_batch:
                    batch_info.extend([batch_info[-1]] * (fixed_batch - len(batch_info)))
                batch_images = []
                batch_scales = []
                for info in batch_info:
                    bgr_image, _, _, _, adjusted_mfdetrec_res, _, _, _ = info
                    det_image = _apply_mask_boxes_to_image(bgr_image, adjusted_mfdetrec_res)
                    if doc_scale < 1.0:
                        h, w = det_image.shape[:2]
                        det_image = cv2.resize(
                            det_image,
                            (max(1, int(round(w * doc_scale))),
                             max(1, int(round(h * doc_scale)))),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    padded, extra_scale = pad_det_to_shape(
                        det_image, canvas_h, canvas_w
                    )
                    batch_images.append(padded)
                    batch_scales.append(doc_scale * extra_scale)

                batch_results = ocr_model.det_batch_predict(batch_images, fixed_batch)

                for info, scale, (dt_boxes, _) in zip(
                    real_batch, batch_scales[:real_count],
                    batch_results[:real_count],
                ):
                    bgr_image, _det_image, useful_list, ocr_res_dict, adjusted_mfdetrec_res, _lang, res, ocr_enable = info
                    if dt_boxes is not None and len(dt_boxes) > 0 and scale != 1.0:
                        dt_boxes = (np.asarray(dt_boxes, dtype=np.float32) / scale).tolist()

                    if dt_boxes is not None and len(dt_boxes) > 0:
                        dt_boxes_sorted = sorted_boxes(dt_boxes)
                        dt_boxes_merged = merge_det_boxes(dt_boxes_sorted) if dt_boxes_sorted else []

                        dt_boxes_final = (
                            update_det_boxes(dt_boxes_merged, adjusted_mfdetrec_res)
                            if dt_boxes_merged and adjusted_mfdetrec_res
                            else dt_boxes_merged
                        )

                        if dt_boxes_final:
                            ocr_res = [box.tolist() if hasattr(box, 'tolist') else box for box in dt_boxes_final]
                            ocr_result_list = get_ocr_result_list(
                                ocr_res, useful_list, ocr_enable, bgr_image,
                                _lang, res['original_label'], res['original_order']
                            )
                            ocr_res_dict['layout_res'].extend(ocr_result_list)

                for i in range(start, start + real_count):
                    lang_crop_list[i] = None
        else:
            # CPU 模式：按分辨率分组，避免无用 padding 增大计算量
            resolution_groups = defaultdict(list)
            for info in lang_crop_list:
                bgr_image, _, _, _, adjusted_mfdetrec_res, _, _, _ = info
                cropped_img = _apply_mask_boxes_to_image(bgr_image, adjusted_mfdetrec_res)
                h, w = cropped_img.shape[:2]
                target_h = ((h + RESOLUTION_GROUP_STRIDE - 1) // RESOLUTION_GROUP_STRIDE) * RESOLUTION_GROUP_STRIDE
                target_w = ((w + RESOLUTION_GROUP_STRIDE - 1) // RESOLUTION_GROUP_STRIDE) * RESOLUTION_GROUP_STRIDE
                resolution_groups[(target_h, target_w)].append(info)

            for (target_h, target_w), group_crops in tqdm(resolution_groups.items(), desc=f"OCR-det {lang}"):
                batch_images = []
                for info in group_crops:
                    bgr_image, _, _, _, adjusted_mfdetrec_res, _, _, _ = info
                    img = _apply_mask_boxes_to_image(bgr_image, adjusted_mfdetrec_res)
                    h, w = img.shape[:2]
                    padded_img = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
                    padded_img[:h, :w] = img
                    batch_images.append(padded_img)

                det_batch_size = min(len(batch_images), ocr_det_base_batch_size)
                batch_results = ocr_model.det_batch_predict(batch_images, det_batch_size)

                for info, (dt_boxes, _) in zip(group_crops, batch_results):
                    bgr_image, _det_image, useful_list, ocr_res_dict, adjusted_mfdetrec_res, _lang, res, ocr_enable = info

                    if dt_boxes is not None and len(dt_boxes) > 0:
                        dt_boxes_sorted = sorted_boxes(dt_boxes)
                        dt_boxes_merged = merge_det_boxes(dt_boxes_sorted) if dt_boxes_sorted else []

                        dt_boxes_final = (
                            update_det_boxes(dt_boxes_merged, adjusted_mfdetrec_res)
                            if dt_boxes_merged and adjusted_mfdetrec_res
                            else dt_boxes_merged
                        )

                        if dt_boxes_final:
                            ocr_res = [box.tolist() if hasattr(box, 'tolist') else box for box in dt_boxes_final]
                            ocr_result_list = get_ocr_result_list(
                                ocr_res, useful_list, ocr_enable, bgr_image,
                                _lang, res['original_label'], res['original_order']
                            )
                            ocr_res_dict['layout_res'].extend(ocr_result_list)


# =================================== OCR-rec ===================================
def _run_ocr_rec_postprocess(images_layout_res: List[List[Dict]], ocr_config):
    """OCR rec 后处理"""
    atom_model_manager = AtomModelSingleton()

    need_ocr_by_lang = {}
    img_crop_by_lang = {}

    for layout_res in images_layout_res:
        for item in layout_res:
            if item['category_id'] == CategoryId.OcrText:
                if 'np_img' in item and 'lang' in item:
                    lang = item['lang']

                    if lang not in need_ocr_by_lang:
                        need_ocr_by_lang[lang] = []
                        img_crop_by_lang[lang] = []

                    need_ocr_by_lang[lang].append(item)
                    img_crop_by_lang[lang].append(item['np_img'])

                    item.pop('np_img')
                    item.pop('lang')

    if not img_crop_by_lang:
        return

    for lang, img_crop_list in img_crop_by_lang.items():
        if not img_crop_list:
            continue

        ocr_model = atom_model_manager.get_atom_model(
            atom_model_name=AtomicModel.OCR,
            lang=lang,
            ocr_config=ocr_config,
        )

        # GPU 模式：uniform padding + 固定 batch 消除 JIT 重编译
        device = ocr_config.get("Rec.device", "cpu")
        _orig_n = len(img_crop_list)
        ocr_res_list = []
        if "gpu" in device.lower() or "openvino" in device.lower():
            # ``text_recognizer_call`` sorts all crops by width and runs
            # bucket-sized, bucket-width batches.  Splitting here would only
            # add partial batches and extra static shapes.
            rec_model = ocr_model.ocr_engine.text_rec
            if hasattr(rec_model, 'cfg') and hasattr(rec_model.cfg, 'rec_batch_num'):
                rec_model.cfg.rec_batch_num = 64
            if hasattr(rec_model, 'rec_batch_num'):
                rec_model.rec_batch_num = 64

            ocr_res_list = ocr_model.ocr(
                img_crop_list, det=False, tqdm_enable=True
            )[0]
            if len(ocr_res_list) != _orig_n:
                raise RuntimeError(
                    f'OCR-rec returned {len(ocr_res_list)}/{_orig_n} results'
                )
        else:
            try:
                ocr_res_list = ocr_model.ocr(img_crop_list, det=False, tqdm_enable=True)[0]
            except Exception as exc:
                logger.warning(f'OCR-rec batch failed, retry one by one: {exc}')
                ocr_res_list = []
                for img_crop in img_crop_list:
                    try:
                        one_res = ocr_model.ocr([img_crop], det=False, tqdm_enable=False)[0]
                    except Exception as one_exc:
                        logger.warning(f'skip failed OCR-rec crop: {one_exc}')
                        one_res = []
                    ocr_res_list.append(one_res[0] if one_res else ("", 0.0))

        assert len(ocr_res_list) == len(need_ocr_by_lang[lang])

        for item, (ocr_text, ocr_score) in zip(need_ocr_by_lang[lang], ocr_res_list):
            item['text'] = ocr_text
            item['score'] = float(f"{ocr_score:.3f}")

            if ocr_score < OcrConfidence.min_confidence:
                item['category_id'] = CategoryId.LowScoreText
            else:
                # 特殊字符过滤
                bbox = [item['poly'][0], item['poly'][1], item['poly'][4], item['poly'][5]]
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]

                special_texts = ['（204号', '（20', '（2', '（2号', '（20号', '号', '（204']
                if ocr_text in special_texts and ocr_score < 0.8 and width < height:
                    item['category_id'] = CategoryId.LowScoreText

# =================================== OCR-rec ===================================
def _detect_table_text(
        ocr_model,
        det_image: np.ndarray,
        mfd_res,
        retry_padding: int = 8,
        retry_short_side: int = 32,
):
    """Detect table text, retrying boundary-clipped thin crops with white padding."""
    det_res = ocr_model.ocr(det_image, mfd_res=mfd_res, rec=False)[0]
    if det_res is not None and len(det_res) > 0:
        return det_res

    height, width = det_image.shape[:2]
    padding = max(0, int(retry_padding))
    if padding == 0 or min(height, width) >= int(retry_short_side):
        return det_res

    padded_image = cv2.copyMakeBorder(
        det_image,
        padding, padding, padding, padding,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    # Formula regions are already masked in det_image. Omitting mfd_res here
    # avoids mixing original-image coordinates with the padded retry image.
    padded_res = ocr_model.ocr(padded_image, mfd_res=None, rec=False)[0]
    if padded_res is None or len(padded_res) == 0:
        return det_res

    restored = np.asarray(padded_res, dtype=np.float32).copy()
    restored[..., 0] = np.clip(restored[..., 0] - padding, 0, max(0, width - 1))
    restored[..., 1] = np.clip(restored[..., 1] - padding, 0, max(0, height - 1))
    logger.debug(
        f'table OCR-det recovered {len(restored)} box(es) with {padding}px edge padding '
        f'for image shape {det_image.shape}'
    )
    return restored.tolist()


def _table_html_blank_ratio(html_code: str) -> float:
    """Fraction of physical cells with no visible text."""
    if not html_code:
        return 1.0
    cells = re.findall(r"<(?:td|th)\b[^>]*>(.*?)</(?:td|th)>", html_code, flags=re.S | re.I)
    if not cells:
        return 1.0
    blank = sum(
        1 for cell in cells
        if not re.sub(r"<[^>]+>|&nbsp;|\s", "", cell)
    )
    return blank / len(cells)


def _count_uncovered_text_boxes(det_res, ocr_result, min_overlap: float = 0.3):
    """Count detected text boxes that native PDF text extraction did not cover.

    Native extraction can miss low-contrast text (for example gray signature
    titles that are still visible in the table crop). When a material share of
    detected boxes has no native text, OCR recognition is more complete.
    """
    if not det_res:
        return 0, 0
    native_boxes = []
    if ocr_result and len(ocr_result) > 0:
        for item in ocr_result[0]:
            pts = np.asarray(item, dtype=np.float32).reshape(-1, 2)
            if pts.size == 0:
                continue
            native_boxes.append((
                float(pts[:, 0].min()), float(pts[:, 1].min()),
                float(pts[:, 0].max()), float(pts[:, 1].max()),
            ))
    missing = 0
    for box in det_res:
        pts = np.asarray(box, dtype=np.float32).reshape(-1, 2)
        if pts.size == 0:
            continue
        x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
        x1, y1 = float(pts[:, 0].max()), float(pts[:, 1].max())
        area = max(1.0, (x1 - x0) * (y1 - y0))
        covered = False
        for nx0, ny0, nx1, ny1 in native_boxes:
            iw = max(0.0, min(x1, nx1) - max(x0, nx0))
            ih = max(0.0, min(y1, ny1) - max(y0, ny0))
            if iw * ih / area >= min_overlap:
                covered = True
                break
            cx, cy = (nx0 + nx1) / 2, (ny0 + ny1) / 2
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                covered = True
                break
        if not covered:
            missing += 1
    return missing, len(det_res)


def _prepare_table_data(
        table_res_dict: Dict,
        page_dict: Dict,
        scale: float,
        atom_model_manager: AtomModelSingleton,
        table_config,
        ocr_config,
        skip_table_rec: bool = False,
        uniform_det_size: Tuple[int, int] = None,
        precomputed_table_class: tuple = None,
):
    """表格预处理：rule check → OCR → rotation → fill_image，返回 predict 所需参数。

    Args:
        skip_table_rec: 若为 True，推迟表格内 OCR-rec，将原始 crop 信息存入
            predict_kwargs['_deferred_rec']，供调用方跨表格 batch 处理。
    """
    (table_force_ocr, skip_text_in_image, table_use_word_box,
     table_formula_enable, table_image_enable, table_extract_original_image,
     use_rule_table, rule_table_engine, rule_table_score_threshold,
     ocr_det_retry_padding, ocr_det_retry_short_side,
     ) = _parse_table_config(table_config)

    _lang = table_res_dict['lang']
    useful_list = table_res_dict['useful_list']

    adjusted_mfdetrec_res = None
    if table_formula_enable:
        adjusted_mfdetrec_res = get_adjusted_mfdetrec_res(
            table_res_dict['single_page_mfdetrec_res'] + table_res_dict['checkbox_res'],
            useful_list, return_text=True
        )

    table_class = None
    table_class_score = 1.0
    pdf_not_rotate = page_dict.get("rotate_label") not in ["90", "180", "270"]
    if (
        use_docvortex_table_rules(table_config) and not table_res_dict['ocr_enable']
        and pdf_not_rotate and '_source_pdf_bytes' in page_dict
    ):
        poly = table_res_dict['table_res']['poly']
        table_bbox = (poly[0] / scale, poly[1] / scale, poly[4] / scale, poly[5] / scale)
        table_bbox_scaled = (poly[0], poly[1], poly[4], poly[5])
        has_embedded_formula = any(
            item.get('bbox') and _box_center_in(table_bbox_scaled, item['bbox'])
            for item in table_res_dict['single_page_mfdetrec_res'] + table_res_dict['checkbox_res']
        )
        has_native_image = any(
            img.get('bbox') and _boxes_overlap(table_bbox, img['bbox'])
            for img in page_dict.get('ori_image_list', [])
        )
        if not has_embedded_formula and not has_native_image:
            try:
                native_result = recover_docvortex_table(page_dict, table_bbox)
                if (native_result is not None and native_result.html.strip()
                        and native_result.confidence >= rule_table_score_threshold):
                    blank_ratio = _table_html_blank_ratio(native_result.html)
                    if blank_ratio <= 0.45:
                        table_res_dict['table_res'].pop('layout_image_list', None)
                        table_res_dict['table_res']['html'] = native_result.html
                        table_res_dict['table_res']['rule_table_score'] = native_result.confidence
                        table_res_dict['table_res']['table_parse_method'] = 'docvortex_native'
                        return
                    logger.info(
                        f'DocVortex native table too sparse ({blank_ratio:.2f}), fallback to model'
                    )
            except ImportError:
                raise
            except Exception as exc:
                logger.warning(f'DocVortex table failed, fallback to model: {exc}')

    # The model is initialized only after the native route declines this table.
    table_model = atom_model_manager.get_atom_model(
        atom_model_name='table', lang=_lang, ocr_config=ocr_config,
        table_config=table_config,
    )
    native_rule_candidate = (
        use_rule_table and rule_table_engine == "legacy"
        and not table_force_ocr
        and not table_res_dict['ocr_enable']
        and pdf_not_rotate
        and hasattr(table_model, 'rule_table_class')
    )
    if precomputed_table_class is not None:
        table_class, table_class_score = precomputed_table_class
    elif native_rule_candidate:
        try:
            table_class, table_class_score = table_model.rule_table_class(table_res_dict['table_img'])
            if table_class == "wired":
                poly = table_res_dict['table_res']['poly']
                table_bbox = [poly[0] / scale, poly[1] / scale, poly[4] / scale, poly[5] / scale]
                table_bbox_scaled = [poly[0], poly[1], poly[4], poly[5]]
                has_embedded_formula = any(
                    item.get('bbox') and _box_center_in(table_bbox_scaled, item['bbox'])
                    for item in table_res_dict['single_page_mfdetrec_res'] + table_res_dict['checkbox_res']
                )
                # Native images make cell semantics ambiguous. Skip even when
                # table_image_enable is disabled, since this is a rule-parser
                # safety decision rather than an output feature switch.
                has_native_image = any(
                    img.get('bbox') and _boxes_overlap(table_bbox, img['bbox'])
                    for img in page_dict.get('ori_image_list', [])
                )
                if not has_native_image and not has_embedded_formula:
                    rule_result = (
                        build_rule_table(page_dict, table_bbox)
                        or build_track_table(page_dict, table_bbox)
                    )
                    if rule_result and rule_result.score >= rule_table_score_threshold:
                        blank_ratio = _table_html_blank_ratio(rule_result.html)
                        if blank_ratio <= 0.45:
                            table_res_dict['table_res'].pop('layout_image_list', None)
                            table_res_dict['table_res']['html'] = rule_result.html
                            table_res_dict['table_res']['rule_table_score'] = rule_result.score
                            table_res_dict['table_res']['table_parse_method'] = 'pdfium_rule'
                            return None  # rule-based success
                        logger.info(
                            f'PDFium rule table too sparse ({blank_ratio:.2f}), fallback to model'
                        )
        except Exception as exc:
            logger.warning(f'PDFium rule table failed, fallback to model: {exc}')

    ocr_config_clean = None
    if ocr_config is not None:
        ocr_config_clean = copy.deepcopy(ocr_config)
        ocr_config_clean.pop("custom_model", None)
    ocr_model = atom_model_manager.get_atom_model(
        atom_model_name=AtomicModel.OCR,
        det_db_box_thresh=0.5,
        det_db_unclip_ratio=1.6,
        lang=_lang,
        ocr_config=ocr_config_clean,
        enable_merge_det_boxes=False,
    )

    # 获取表格文本框
    bgr_image = cv2.cvtColor(table_res_dict["table_img"], cv2.COLOR_RGB2BGR)
    det_image = (
        _apply_mask_boxes_to_image(bgr_image, adjusted_mfdetrec_res)
        if adjusted_mfdetrec_res
        else bgr_image
    )
    # GPU 模式：表格检测图 pad 到最近的方形桶（320/384/512/.../960），
    # 而不是所有表格共用最大画布。
    device = ocr_config.get("Det.device", "cpu")
    det_scale = 1.0
    if uniform_det_size is not None and ("gpu" in str(device).lower()):
        canvas = max(uniform_det_size)
        det_image, det_scale = pad_det_to_square(det_image, canvas)
    det_res = _detect_table_text(
            ocr_model, det_image, adjusted_mfdetrec_res,
            retry_padding=ocr_det_retry_padding, retry_short_side=ocr_det_retry_short_side,
        )
    if det_res is not None and len(det_res) > 0 and det_scale != 1.0:
        det_res = (np.asarray(det_res, dtype=np.float32) / det_scale).tolist()

    angles = []
    rotate_label = "0"
    if pdf_not_rotate:
        # 检测文字旋转
        rotate_label, angles = txt_most_angle_extract_table(page_dict, table_res_dict, scale=scale)
    if not angles:
        # 如果没有文本的角度，使用模型判断是否旋转
        img_orientation_cls_model = atom_model_manager.get_atom_model(
            atom_model_name=AtomicModel.ImgOrientationCls,
        )
        rotate_label = img_orientation_cls_model.predict(bgr_image, det_res)
    if rotate_label in ["90", "270"]:
        rotate_table_image(table_res_dict, rotate_label)
        # 旋转后的表格需要重新获取文本框
        bgr_image = cv2.cvtColor(table_res_dict["table_img"], cv2.COLOR_RGB2BGR)
        det_image = (
            _apply_mask_boxes_to_image(bgr_image, adjusted_mfdetrec_res)
            if adjusted_mfdetrec_res
            else bgr_image
        )
        det_scale = 1.0
        if uniform_det_size is not None and ("gpu" in str(device).lower()):
            det_image, det_scale = pad_det_to_square(
                det_image, max(uniform_det_size)
            )
        det_res = _detect_table_text(
            ocr_model, det_image, adjusted_mfdetrec_res,
            retry_padding=ocr_det_retry_padding, retry_short_side=ocr_det_retry_short_side,
        )
        if det_res is not None and len(det_res) > 0 and det_scale != 1.0:
            det_res = (np.asarray(det_res, dtype=np.float32) / det_scale).tolist()

    ocr_result = []

    # 尝试从 PDF 提取文本
    if (not table_force_ocr and not table_res_dict['ocr_enable'] and rotate_label == "0" and pdf_not_rotate):
        ocr_result = _extract_table_text_from_pdf(
            table_res_dict, page_dict, scale, det_res, useful_list, table_use_word_box
        )

    if ocr_result and det_res:
        missing, total = _count_uncovered_text_boxes(det_res, ocr_result)
        if missing and missing / max(1, total) >= 0.25:
            logger.info(
                f'table native text incomplete ({missing}/{total} detected boxes '
                f'uncovered), fallback to OCR recognition'
            )
            ocr_result = []
    needs_rec = not ocr_result and det_res
    if skip_table_rec and needs_rec:
        ocr_result = []
    elif needs_rec:
        ocr_result = _run_table_ocr(ocr_model, bgr_image, det_res, table_use_word_box)

    fill_image_res = []
    if table_image_enable:
        if not pdf_not_rotate:
            table_extract_original_image = False
        fill_image_res = extract_table_fill_image(
            page_dict, table_res_dict, scale, table_extract_original_image
        )

    table_res_dict['table_res'].pop('layout_image_list', None)

    predict_kwargs = dict(
        image=table_res_dict['table_img'],
        ocr_result=ocr_result,
        fill_image_res=fill_image_res,
        mfd_res=adjusted_mfdetrec_res,
        skip_text_in_image=skip_text_in_image,
        skip_table_orientation=True,
    )
    if hasattr(table_model, 'rule_table_class'):
        predict_kwargs.update(table_class=table_class, table_class_score=table_class_score)

    if skip_table_rec and needs_rec:
        predict_kwargs['_deferred_rec'] = (ocr_model, bgr_image, det_res, table_use_word_box, ocr_config)

    return (table_res_dict, table_model, predict_kwargs)


def _parse_table_config(table_config):
    """提取并返回表格配置参数元组。"""
    table_force_ocr = table_config.get("force_ocr", False)
    skip_text_in_image = table_config.get("skip_text_in_image", True)
    table_use_word_box = table_config.get("use_word_box", False)
    table_formula_enable = table_config.get("table_formula_enable", True)
    table_image_enable = table_config.get("table_image_enable", True)
    table_extract_original_image = table_config.get("extract_original_image", False)
    use_rule_table = table_config.get("use_rule_table", True)
    rule_table_engine = table_config.get("rule_table_engine", "docvortex")
    rule_table_score_threshold = float(table_config.get("rule_table_score_threshold", 0.90))
    ocr_det_retry_padding = int(table_config.get("ocr_det_retry_padding", 8))
    ocr_det_retry_short_side = int(table_config.get("ocr_det_retry_short_side", 32))
    return (table_force_ocr, skip_text_in_image, table_use_word_box,
            table_formula_enable, table_image_enable, table_extract_original_image,
            use_rule_table, rule_table_engine, rule_table_score_threshold,
            ocr_det_retry_padding, ocr_det_retry_short_side)


def _run_batched_table_inference(table_config, batch_data: List[tuple]):
    """对一批预处理后的表格执行分组 batch 推理。"""
    from ...model.table.rapid_table_self import ModelType
    from ...model.table.utils import select_best_table_model

    if not batch_data:
        return

    model_type = table_config.get("model_type", "UNET_SLANET_PLUS")
    is_combined = str(model_type) in ["UNET_SLANET_PLUS", "UNET_SLANET1M", "UNET_UNITABLE"]

    def _batch_htmls(entries, attr_name):
        if not entries:
            return []
        model = getattr(entries[0][1], attr_name, None)
        if model is None:
            return [e[1].predict(**e[2]) for e in entries]
        # RapidTable receives raw ndarrays without color conversion, while
        # the single-table path (RapidTableModel.predict) converts RGB to
        # BGR. The pipeline stores table crops as RGB, so batch inference
        # must apply the same conversion or SLANET decodes a wrong grid
        # (e.g. blank leading cells on long financial tables).
        imgs = []
        for e in entries:
            img = np.asarray(e[2]['image'])
            if img.ndim == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            imgs.append(img)
        ocrs = [e[2]['ocr_result'] for e in entries]
        n_real = len(imgs)
        model_kind = str(getattr(getattr(model, 'cfg', None), 'model_type', '')).lower()
        if 'unet' in model_kind:
            # UNET runs one table at a time internally (``_batch_call``
            # loops over the list).  Padding the batch would therefore run
            # the model again for every dummy image, which is exactly the
            # table-throughput regression this change removes.
            return [
                model([img], [ocr], batch_size=1).pred_htmls[0]
                for img, ocr in zip(imgs, ocrs)
            ]

        max_batch = max(
            1, get_env_batch_size('RAPID_DOC_TABLE_STRUCTURE_BATCH_SIZE', 16) or 16
        )
        # SLANET is ~3x faster at batch 16 than at batch 1/4/8 on the Arc GPU
        # (benchmarked: 0.7 s vs 2.2-2.5 s per call), so padding a small group
        # to 16 is faster than running the smaller batch.  Other structure
        # models (UNET) never pad.
        if 'slanet' in model_kind:
            batches = padded_batches(
                list(zip(imgs, ocrs)), max_batch, buckets=(max_batch,)
            )
        else:
            batches = padded_batches(list(zip(imgs, ocrs)), max_batch)
        htmls = []
        for chunk, bucket, real_count in batches:
            chunk_imgs = [item[0] for item in chunk]
            chunk_ocrs = [item[1] for item in chunk]
            # ``real_count`` entries are real; the rest pad the last bucket.
            htmls.extend(
                model(chunk_imgs, chunk_ocrs, batch_size=bucket).pred_htmls[
                    :real_count
                ]
            )
        return htmls

    # SLANET and UNET are both batch-safe. Run both models for normal tables
    # and pick the denser result; this avoids trusting a wrong wired/wireless
    # classifier while still using high-batch table-structure inference.
    if str(model_type) == "UNET_SLANET_PLUS" and os.getenv("RAPID_DOC_TABLE_SINGLE_MODEL") != "1":
        # Tables with fill images/formulas used to take a batch-1 "special"
        # path while ordinary tables used batch 16.  On the GPU plugin that
        # mixes two batch shapes for the same compiled SLANET model.  The OCR
        # results are already materialised by _prepare_table_data, so every
        # table can go through the same fixed-batch structure model.
        batch_entries = list(batch_data)

        if os.getenv("RAPID_DOC_TABLE_PER_TABLE") == "1":
            for entry in batch_entries:
                html_code = entry[1].predict(**entry[2])
                normalized = normalize_table_html_cell_text(html_code) if html_code else None
                if not _write_table_result(entry[0], normalized):
                    _fallback_table_to_text(entry[0], entry[2])
            return
        wired_htmls = _batch_htmls(batch_entries, 'wired_table_model')
        wireless_htmls = _batch_htmls(batch_entries, 'wireless_table_model')
        for entry, wired, wireless in zip(batch_entries, wired_htmls, wireless_htmls):
            ocr_result = entry[2].get('ocr_result') or [[], [], []]
            html_code = select_best_table_model(ocr_result, wired, wireless)
            if html_code and _table_html_blank_ratio(html_code) > 0.4:
                # A batched result can still be the wrong model for very long
                # scanned tables. Re-run both models on just this table and pick
                # the denser grid. Keep the fixed batch shape even for this
                # single-table retry.
                try:
                    w_single = _batch_htmls([entry], 'wired_table_model')[0]
                    l_single = _batch_htmls([entry], 'wireless_table_model')[0]
                    html_code = select_best_table_model(ocr_result or [[], [], []], w_single, l_single)
                except Exception as exc:
                    logger.warning(
                        f'table structure retry failed, keeping batched result: {exc}'
                    )
            normalized = normalize_table_html_cell_text(html_code) if html_code else None
            if not _write_table_result(entry[0], normalized):
                _fallback_table_to_text(entry[0], entry[2])

        return

    # Other combined models (UNITABLE) and single-model configurations keep
    # their existing class-based batching path.
    wired_entries, wireless_entries = [], []
    for entry in batch_data:
        table_res_dict, table_model, pk = entry
        needs_compare = (
            getattr(table_model, "use_compare_table", False)
            or pk.get("table_class_score", 1.0)
            < getattr(table_model, "cls_conf_threshold", 0.9)
        )
        if (
            not is_combined
            or needs_compare
            or pk.get('fill_image_res')
            or pk.get('mfd_res')
        ):
            html_code = table_model.predict(**pk)
            if not _write_table_result(table_res_dict, html_code):
                _fallback_table_to_text(table_res_dict, pk)
        elif pk.get('table_class') == "wired":
            wired_entries.append(entry)
        elif pk.get('table_class') == "wireless":
            wireless_entries.append(entry)
        else:
            html_code = table_model.predict(**pk)
            if not _write_table_result(table_res_dict, html_code):
                _fallback_table_to_text(table_res_dict, pk)

    def _batch_group(entries, attr_name):
        if not entries:
            return
        htmls = _batch_htmls(entries, attr_name)
        for entry, html_code in zip(entries, htmls):
            normalized = normalize_table_html_cell_text(html_code) if html_code else None
            if not _write_table_result(entry[0], normalized):
                _fallback_table_to_text(entry[0], entry[2])

    _batch_group(wired_entries, 'wired_table_model')
    _batch_group(wireless_entries, 'wireless_table_model')


def _write_table_result(table_res_dict, html_code):
    if html_code and '<table>' in html_code and '</table>' in html_code:
        start = html_code.find('<table>')
        end = html_code.rfind('</table>') + len('</table>')
        table_res_dict['table_res']['html'] = html_code[start:end]
        return True
    logger.warning('table recognition processing fails')
    return False


def _fallback_table_to_text(table_res_dict, predict_kwargs):
    """表格结构识别为空时，把该区域还原成普通文本，避免整段丢失。"""
    ocr_result = predict_kwargs.get('ocr_result')
    if not ocr_result or len(ocr_result) < 2:
        return False
    texts = [text for text in ocr_result[1] if text and str(text).strip()]
    if not texts:
        return False
    table_res = table_res_dict.get('table_res', {})
    table_res['category_id'] = CategoryId.OcrText
    table_res['text'] = ' '.join(str(text) for text in texts)
    table_res.pop('html', None)
    return True


def _process_single_table(
        table_res_dict: Dict,
        page_dict: Dict,
        scale: float,
        atom_model_manager: AtomModelSingleton,
        table_config,
        ocr_config,
):
    """处理单个表格"""
    result = _prepare_table_data(table_res_dict, page_dict, scale,
                                 atom_model_manager, table_config, ocr_config)
    if result is None:
        return  # rule-based success
    table_res_dict, table_model, predict_kwargs = result
    html_code = table_model.predict(**predict_kwargs)
    if not _write_table_result(table_res_dict, html_code):
        _fallback_table_to_text(table_res_dict, predict_kwargs)


def _boxes_overlap(a, b) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _box_center_in(container, item) -> bool:
    cx, cy = (item[0] + item[2]) / 2, (item[1] + item[3]) / 2
    return container[0] <= cx <= container[2] and container[1] <= cy <= container[3]

def _extract_table_text_from_pdf(
        table_res_dict: Dict,
        page_dict: Dict,
        scale: float,
        det_res: List,
        useful_list: List,
        table_use_word_box
) -> List:
    """从 PDF 中提取表格文本"""
    if not det_res:
        return []

    try:
        ocr_spans = get_ocr_result_list_table(det_res, useful_list, scale)
        poly = table_res_dict['table_res']['poly']
        table_bboxes = [[
            int(poly[0] / scale), int(poly[1] / scale),
            int(poly[4] / scale), int(poly[5] / scale),
            None, None, None, 'text', None, None, None, None, 1
        ]]

        txt_spans_extract(
            page_dict, ocr_spans, table_res_dict['table_img'], scale,
            table_bboxes, [], return_word_box=table_use_word_box,
            useful_list=table_res_dict['useful_list']
        )

        if table_use_word_box:
            filtered = [
                (w[2], normalize_table_ocr_text(w[0]), w[1])
                for item in ocr_spans
                for group in [item.get('word_result')]
                if group
                for w in group
                if w and w[2] != ""
            ]
        else:
            filtered = [
                [item['ori_bbox'], normalize_table_ocr_text(item['content']), item['score']]
                for item in ocr_spans if item.get('content')
            ]

        return [list(x) for x in zip(*filtered)] if filtered else []
    except Exception:
        logger.warning('table ocr_result get from pdf error')
        return []

def _run_table_ocr(
        ocr_model,
        bgr_image: np.ndarray,
        det_res: List,
        table_use_word_box,
) -> List:
    """执行表格 OCR"""
    rec_img_list = []
    for dt_box in det_res:
        rec_img_list.append({
            "cropped_img": get_rotate_crop_image(bgr_image, np.asarray(dt_box, dtype=np.float32)),
            "dt_box": np.asarray(dt_box, dtype=np.float32),
        })

    cropped_img_list = [item["cropped_img"] for item in rec_img_list]
    # ``text_recognizer_call`` selects bucket batch/width per sorted batch.

    ocr_res_list = None
    try:
        ocr_res_list = ocr_model.ocr(
            cropped_img_list, det=False, tqdm_enable=False,
            return_word_box=table_use_word_box, ori_img=bgr_image, dt_boxes=det_res
        )[0]
    except Exception as exc:
        if table_use_word_box:
            logger.warning(f'table OCR word-box recognition failed, retry line-level OCR: {exc}')
            table_use_word_box = False
            try:
                ocr_res_list = ocr_model.ocr(
                    cropped_img_list, det=False, tqdm_enable=False,
                    return_word_box=False, ori_img=bgr_image, dt_boxes=det_res
                )[0]
            except Exception as retry_exc:
                logger.warning(f'table OCR batch recognition failed, retry one by one: {retry_exc}')
        else:
            logger.warning(f'table OCR batch recognition failed, retry one by one: {exc}')

    if ocr_res_list is None:
        ocr_res_list = []
        rec_img_list_safe = []
        for img_dict in rec_img_list:
            try:
                one_res = ocr_model.ocr(
                    [img_dict["cropped_img"]], det=False, tqdm_enable=False,
                    return_word_box=False, ori_img=bgr_image, dt_boxes=[img_dict["dt_box"]]
                )[0]
            except Exception as exc:
                logger.warning(f'skip failed table OCR crop: {exc}')
                continue
            if not one_res:
                continue
            ocr_res_list.append(one_res[0])
            rec_img_list_safe.append(img_dict)
        rec_img_list = rec_img_list_safe

    ocr_result = []
    for img_dict, ocr_res in zip(rec_img_list, ocr_res_list):
        if table_use_word_box:
            ocr_result.extend([
                [word_result[2], normalize_table_ocr_text(word_result[0]), word_result[1]]
                for word_result in ocr_res[2]
            ])
        else:
            ocr_result.append([img_dict["dt_box"], normalize_table_ocr_text(ocr_res[0]), ocr_res[1]])

    return [list(x) for x in zip(*ocr_result)] if ocr_result else []
