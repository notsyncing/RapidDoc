# Copyright (c) RapidAI. All rights reserved.
"""
批量分析模块
"""
import copy
import os
from typing import List, Dict
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
from ...model.table.utils import normalize_table_html_cell_text
from ...utils.bbox_utils import normalize_to_int_bbox
from ...utils.boxbase import rotate_table_image
from ...utils.enum_class import CategoryId
from ...utils.model_utils import crop_img
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
        import openvino as ov
        core = ov.Core()
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
    ocr_det_base_batch_size = ocr_config.get("Det.rec_batch_num", 1)

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

        # GPU 模式：所有 crop 统一 pad 到全局最大尺寸 → 单一 GPU shape → 无需重编译
        device = ocr_config.get("Det.device", "cpu")
        if "gpu" in device.lower() or "openvino" in device.lower():
            max_h = max_w = 0
            for info in lang_crop_list:
                bgr_image, _, _, _, adjusted_mfdetrec_res, _, _, _ = info
                det_image = _apply_mask_boxes_to_image(bgr_image, adjusted_mfdetrec_res)
                h, w = det_image.shape[:2]
                max_h = max(max_h, ((h + RESOLUTION_GROUP_STRIDE - 1) // RESOLUTION_GROUP_STRIDE) * RESOLUTION_GROUP_STRIDE)
                max_w = max(max_w, ((w + RESOLUTION_GROUP_STRIDE - 1) // RESOLUTION_GROUP_STRIDE) * RESOLUTION_GROUP_STRIDE)

            det_batch_size = min(len(lang_crop_list), ocr_det_base_batch_size)
            det_batch_size = _gpu_safe_det_batch_size(max_h, max_w, det_batch_size)

            # 按批 pad + 推理：避免全量 crop 的 pad 副本驻留系统内存
            # （大文档下可达数 GB），每批推理完立即处理并释放。
            for start in tqdm(range(0, len(lang_crop_list), det_batch_size), desc=f"OCR-det {lang}"):
                batch_info = lang_crop_list[start:start + det_batch_size]
                batch_images = []
                for info in batch_info:
                    bgr_image, _, _, _, adjusted_mfdetrec_res, _, _, _ = info
                    det_image = _apply_mask_boxes_to_image(bgr_image, adjusted_mfdetrec_res)
                    h, w = det_image.shape[:2]
                    padded = np.ones((max_h, max_w, 3), dtype=np.uint8) * 255
                    padded[:h, :w] = det_image
                    batch_images.append(padded)

                batch_results = ocr_model.det_batch_predict(batch_images, det_batch_size)

                for info, (dt_boxes, _) in zip(batch_info, batch_results):
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

                # 处理完该批立即释放 bgr_image 引用，避免全量驻留
                for i in range(start, start + len(batch_info)):
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
            max_h = max(img.shape[0] for img in img_crop_list)
            max_w = max(img.shape[1] for img in img_crop_list)
            # GPU batch: 固定大小 power-of-2 → 所有 batch 同 shape → 只 JIT 一次
            GPU_BATCH = 128
            rec_model = ocr_model.ocr_engine.text_rec
            if hasattr(rec_model, 'cfg') and hasattr(rec_model.cfg, 'rec_batch_num'):
                rec_model.cfg.rec_batch_num = min(GPU_BATCH, _orig_n)

            # 按批 pad + 推理：避免全量 crop 的 pad 副本驻留系统内存
            # （大文档下可达数 GB），每批推理完立即释放。
            GPU_BATCH = 128
            for start in range(0, _orig_n, GPU_BATCH):
                batch = img_crop_list[start:start + GPU_BATCH]
                batch_padded = []
                for img in batch:
                    h, w = img.shape[:2]
                    if h < max_h or w < max_w:
                        padded = np.ones((max_h, max_w, 3), dtype=np.uint8) * 255
                        padded[:h, :w] = img
                        batch_padded.append(padded)
                    else:
                        batch_padded.append(img)
                try:
                    batch_res = ocr_model.ocr(batch_padded, det=False, tqdm_enable=True)[0]
                except Exception as exc:
                    logger.warning(f'OCR-rec batch failed, retry one by one: {exc}')
                    batch_res = []
                    for img_crop in batch:
                        try:
                            one_res = ocr_model.ocr([img_crop], det=False, tqdm_enable=False)[0]
                        except Exception as one_exc:
                            logger.warning(f'skip failed OCR-rec crop: {one_exc}')
                            one_res = []
                        batch_res.append(one_res[0] if one_res else ("", 0.0))
                if len(batch_res) != len(batch):
                    logger.warning(
                        f'OCR-rec batch returned {len(batch_res)}/{len(batch)} results, padding'
                    )
                    batch_res = batch_res[:len(batch)]
                    batch_res += [("", 0.0)] * (len(batch) - len(batch_res))
                ocr_res_list.extend(batch_res)
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
    (table_force_ocr, skip_text_in_image, use_img2table, table_use_word_box,
     table_formula_enable, table_image_enable, table_extract_original_image,
     use_rule_table, rule_table_score_threshold,
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

    # Resolve the configured model route before OCR. For combined models this
    # classification is cached and reused if the native rule result falls
    # back, avoiding the former second classification.
    table_model = atom_model_manager.get_atom_model(
        atom_model_name='table', lang=_lang, ocr_config=ocr_config,
        table_config=table_config,
    )
    table_class = None
    table_class_score = 1.0
    pdf_not_rotate = page_dict.get("rotate_label") not in ["90", "180", "270"]
    native_rule_candidate = (
        use_rule_table
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
                        table_res_dict['table_res'].pop('layout_image_list', None)
                        table_res_dict['table_res']['html'] = rule_result.html
                        table_res_dict['table_res']['rule_table_score'] = rule_result.score
                        table_res_dict['table_res']['table_parse_method'] = 'pdfium_rule'
                        return None  # rule-based success
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
    # GPU 模式：表格检测图统一 pad 到全局最大尺寸 → 单一输入 shape → 一次 JIT
    device = ocr_config.get("Det.device", "cpu")
    if uniform_det_size is not None and ("gpu" in str(device).lower()):
        tgt_h, tgt_w = uniform_det_size
        h, w = det_image.shape[:2]
        if h < tgt_h or w < tgt_w:
            padded = np.ones((tgt_h, tgt_w, 3), dtype=np.uint8) * 255
            padded[:h, :w] = det_image
            det_image = padded
    det_res = _detect_table_text(
            ocr_model, det_image, adjusted_mfdetrec_res,
            retry_padding=ocr_det_retry_padding, retry_short_side=ocr_det_retry_short_side,
        )

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
        det_res = _detect_table_text(
            ocr_model, det_image, adjusted_mfdetrec_res,
            retry_padding=ocr_det_retry_padding, retry_short_side=ocr_det_retry_short_side,
        )

    ocr_result = []

    # 尝试从 PDF 提取文本
    if (not table_force_ocr and not table_res_dict['ocr_enable'] and rotate_label == "0" and pdf_not_rotate):
        ocr_result = _extract_table_text_from_pdf(
            table_res_dict, page_dict, scale, det_res, useful_list, table_use_word_box
        )

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
        use_img2table=use_img2table,
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
    use_img2table = table_config.get("use_img2table", False)
    table_use_word_box = table_config.get("use_word_box", False)
    table_formula_enable = table_config.get("table_formula_enable", True)
    table_image_enable = table_config.get("table_image_enable", True)
    table_extract_original_image = table_config.get("extract_original_image", False)
    use_rule_table = table_config.get("use_rule_table", True)
    rule_table_score_threshold = float(table_config.get("rule_table_score_threshold", 0.90))
    ocr_det_retry_padding = int(table_config.get("ocr_det_retry_padding", 8))
    ocr_det_retry_short_side = int(table_config.get("ocr_det_retry_short_side", 32))
    return (table_force_ocr, skip_text_in_image, use_img2table, table_use_word_box,
            table_formula_enable, table_image_enable, table_extract_original_image,
            use_rule_table, rule_table_score_threshold,
            ocr_det_retry_padding, ocr_det_retry_short_side)


def _run_batched_table_inference(table_config, batch_data: List[tuple]):
    """对一批预处理后的表格执行分组 batch 推理。"""
    from ...model.table.rapid_table_self import ModelType

    if not batch_data:
        return

    model_type = table_config.get("model_type", "UNET_SLANET_PLUS")
    is_combined = str(model_type) in ["UNET_SLANET_PLUS", "UNET_SLANET1M", "UNET_UNITABLE"]

    # Collect per-table result holders while grouping
    wired_entries, wireless_entries = [], []
    for entry in batch_data:
        table_res_dict, table_model, pk = entry
        if is_combined and pk.get('table_class') == "wired":
            wired_entries.append(entry)
        elif is_combined and pk.get('table_class') == "wireless":
            wireless_entries.append(entry)
        else:
            # Single-model or unclassified → run individually
            html_code = table_model.predict(**pk)
            _write_table_result(table_res_dict, html_code)

    def _batch_group(entries, attr_name):
        if not entries:
            return
        imgs = [e[2]['image'] for e in entries]
        ocrs = [e[2]['ocr_result'] for e in entries]
        model = getattr(entries[0][1], attr_name, None)
        if model is None:
            for e in entries:
                html_code = e[1].predict(**e[2])
                _write_table_result(e[0], html_code)
            return
        bs = min(len(imgs), 16)
        try:
            batch_htmls = model(imgs, ocrs, batch_size=bs).pred_htmls
        except Exception:
            for e in entries:
                html_code = e[1].predict(**e[2])
                _write_table_result(e[0], html_code)
            return
        for entry, html_code in zip(entries, batch_htmls):
            _write_table_result(entry[0], normalize_table_html_cell_text(html_code) if html_code else None)

    _batch_group(wired_entries, 'wired_table_model')
    _batch_group(wireless_entries, 'wireless_table_model')


def _write_table_result(table_res_dict, html_code):
    if html_code and '<table>' in html_code and '</table>' in html_code:
        start = html_code.find('<table>')
        end = html_code.rfind('</table>') + len('</table>')
        table_res_dict['table_res']['html'] = html_code[start:end]
    else:
        logger.warning('table recognition processing fails')


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
    _write_table_result(table_res_dict, html_code)


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

    # 统一尺寸消除 GPU shape 重编译
    max_h = max(img.shape[0] for img in cropped_img_list)
    max_w = max(img.shape[1] for img in cropped_img_list)
    uniform_list = []
    for img in cropped_img_list:
        h, w = img.shape[:2]
        if h < max_h or w < max_w:
            padded = np.ones((max_h, max_w, 3), dtype=np.uint8) * 255
            padded[:h, :w] = img
            uniform_list.append(padded)
        else:
            uniform_list.append(img)
    cropped_img_list = uniform_list
    # GPU batch: 固定大小 power-of-2，余数 padd → 所有 batch 同 shape → 一次 JIT
    rec_model = getattr(getattr(ocr_model, 'ocr_engine', None), 'text_rec', None)
    _tbl_n = len(cropped_img_list)
    if rec_model is not None and hasattr(rec_model, 'cfg') and hasattr(rec_model.cfg, 'rec_batch_num'):
        GPU_BATCH = 128
        if _tbl_n > GPU_BATCH:
            pad = (GPU_BATCH - _tbl_n % GPU_BATCH) % GPU_BATCH
            if pad:
                dummy = np.zeros_like(cropped_img_list[0])
                cropped_img_list = cropped_img_list + [dummy] * pad
            rec_model.cfg.rec_batch_num = GPU_BATCH
        else:
            rec_model.cfg.rec_batch_num = _tbl_n

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
