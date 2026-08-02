# Copyright (c) RapidAI. All rights reserved.
"""
批量分析模块
"""
import os
import inspect
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
from PIL import Image
from loguru import logger
from tqdm import tqdm

from .analyze_utils import _extract_text_from_pdf, _run_ocr_det_batch, _process_single_table, _run_ocr_rec_postprocess, _prepare_table_data, _run_batched_table_inference
from .model_init import AtomModelSingleton
from .model_list import AtomicModel
from ..utils.utils import remove_layout_in_ori_images, filter_overlap_boxes, _expand_formula_crop_res
from ...model.custom import CustomBaseModel
from ...utils.bbox_utils import normalize_to_int_bbox
from ...utils.boxbase import get_rotate_image, restore_poly
from ...utils.ocr_utils import get_rotate_crop_image
from ...utils.checkbox_det_cls import checkbox_predict
from ...utils.config_reader import get_formula_enable, get_table_enable
from ...utils.enum_class import CategoryId
from ...utils.model_utils import crop_img, get_res_list_from_layout_res, clean_vram
from ...utils.pdf_image_tools import get_crop_np_img
from ...utils.span_pre_proc import extract_table_fill_image


class BatchAnalyze:
    """批量分析处理器"""
    
    def __init__(
        self,
        model_manager,
        batch_ratio: int,
        formula_enable: bool,
        table_enable: bool,
        layout_config: Optional[Dict] = None,
        ocr_config: Optional[Dict] = None,
        formula_config: Optional[Dict] = None,
        table_config: Optional[Dict] = None,
        checkbox_config: Optional[Dict] = None,
    ):
        self.model_manager = model_manager
        self.batch_ratio = batch_ratio
        
        # 功能开关
        self.formula_enable = get_formula_enable(formula_enable)
        self.table_enable = get_table_enable(table_enable)
        self.checkbox_enable = checkbox_config.get("checkbox_enable", False) if checkbox_config else False
        
        # 配置项
        self.layout_config = layout_config or {}
        self.ocr_config = ocr_config or {}
        self.formula_config = formula_config or {}
        self.table_config = table_config or {}
        
        # OCR 配置
        self.use_det_mode = self.ocr_config.get("use_det_mode", "auto")
        base_ocr = self.ocr_config.get("Det.rec_batch_num", 1)
        self.ocr_det_base_batch_size = base_ocr * self.batch_ratio
        # 写回 config 使 _run_ocr_det_batch 等函数读到放大后的值
        if self.batch_ratio > 1:
            self.ocr_config["Det.rec_batch_num"] = self.ocr_det_base_batch_size
            base_rec = self.ocr_config.get("Rec.rec_batch_num", 1)
            self.ocr_config["Rec.rec_batch_num"] = base_rec * self.batch_ratio
        self.seal_enable = self.ocr_config.get("seal_enable", True)
        self.use_custom_ocr = False
        
        # 版面配置
        base_layout = self.layout_config.get("batch_num", 1)
        self.layout_base_batch_size = base_layout * self.batch_ratio
        self.use_doc_orientation_classify = (str(os.getenv("USE_DOC_ORIENTATION_CLASSIFY", "false"))
                                             .strip().lower() in ("true", "1", "yes", "on"))
        # 公式配置
        self.formula_level = self.formula_config.get("formula_level", 0)
        base_formula = self.formula_config.get("batch_num", 1)
        self.formula_base_batch_size = base_formula * self.batch_ratio
        self.formula_bbox_expand_px = int(self.formula_config.get("bbox_expand_px", 2))
        
        # 表格配置
        self.table_image_enable = self.table_config.get("table_image_enable", True)
        self.table_extract_original_image = self.table_config.get("extract_original_image", False)
    
    def __call__(
        self,
        images_with_extra_info: List[Tuple[Image.Image, float, bool, str, dict]]
    ) -> List[List[Dict]]:
        """
        执行批量分析
        
        Args:
            images_with_extra_info: [(PIL图像, 缩放比例, ocr_enable, 语言, pdf_dict), ...]
            
        Returns:
            每页的版面检测结果列表
        """
        if not images_with_extra_info:
            return []
        
        # 初始化模型
        self.model = self.model_manager.get_model(
            lang=None,
            formula_enable=self.formula_enable,
            table_enable=self.table_enable,
            layout_config=self.layout_config,
            ocr_config=self.ocr_config,
            formula_config=self.formula_config,
            table_config=self.table_config,
        )
        atom_model_manager = AtomModelSingleton()
        self.use_custom_ocr = isinstance(self.model.ocr_model, CustomBaseModel)
        
        # 预处理数据
        pdf_dict_list = [pdf_dict for _, _, _, _, pdf_dict in images_with_extra_info]
        # np_images = [cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) for image, _, _, _, _ in images_with_extra_info]
        np_images = [np.array(image) for image, _, _, _, _ in images_with_extra_info]
        scale_list = [scale for _, scale, _, _, _ in images_with_extra_info]

        # 图片方向矫正
        img_ori_orientation_list = []
        if self.use_doc_orientation_classify:
            img_orientation_cls_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.ImgOrientationCls,
            )
            for i, np_img in enumerate(np_images):
                h, w = np_img.shape[:2]
                rotate_label = img_orientation_cls_model.predict(np_img)
                if rotate_label in ["90", "270"]:
                    np_images[i] = get_rotate_image(np_img, rotate_label)
                pdf_dict_list[i]['rotate_label'] = rotate_label
                img_ori_orientation_list.append((h, w, rotate_label))

        # 1. 版面识别
        images_layout_res = self._run_layout_detection(np_images, pdf_dict_list, scale_list)
        # 2. 收集各类型检测区域
        ocr_res_all_page, table_res_all_page, formula_res_all_page = self._collect_detection_regions(
            images_layout_res, np_images, images_with_extra_info
        )
        # 3. 公式识别
        if self.formula_enable:
            self._run_formula_recognition(formula_res_all_page)
        # 4. OCR 识别 (根据模式选择不同的处理方式)
        if isinstance(self.model.ocr_model, CustomBaseModel):
            self._run_custom_ocr(ocr_res_all_page)
        else:
            self._run_traditional_ocr(
                atom_model_manager, ocr_res_all_page, pdf_dict_list, scale_list
            )
        # 5. 表格识别
        if self.table_enable:
            self._run_table_recognition(atom_model_manager, table_res_all_page, pdf_dict_list, scale_list)
        # 6. 后处理 OCR rec 结果
        _run_ocr_rec_postprocess(images_layout_res, self.ocr_config)

        # 7. 印章识别
        if self.seal_enable:
            self._run_seal_ocr(atom_model_manager, np_images, images_layout_res)

        if img_ori_orientation_list:
            # 把旋转后图片上的矩形框，还原到原图坐标
            for index, np_img in enumerate(np_images):
                h, w, rotate_label = img_ori_orientation_list[index]
                layout_res = images_layout_res[index]
                for layout_re in layout_res:
                    layout_re["rotate_label"] = rotate_label
                    if rotate_label in ["90", "270"]:
                        layout_re["poly"] = restore_poly(layout_re["poly"], rotate_label, w, h)

        clean_vram(self.model.device, vram_threshold=8)
        return images_layout_res
    
    def _run_layout_detection(
        self,
        np_images: List[np.ndarray],
        pdf_dict_list: List[Dict],
        scale_list: List[float]
    ) -> List[List[Dict]]:
        """执行版面检测"""
        images_layout_res = self.model.layout_model.batch_predict(
            np_images, self.layout_base_batch_size
        )
        images_layout_res = [filter_overlap_boxes(item, self.use_custom_ocr) for item in images_layout_res]
        # 如果是 txt 模式，移除原始图片中的版面元素
        if self.use_det_mode == 'txt':
            images_layout_res = remove_layout_in_ori_images(images_layout_res, pdf_dict_list, scale_list)
        
        # 公式等级过滤（不识别行间公式，直接作为文本识别）
        if not self.formula_enable or self.formula_level == 1:
            images_layout_res = [
                [item for item in page if item["category_id"] != CategoryId.InlineEquation]
                for page in images_layout_res
            ]
        
        return images_layout_res
    
    def _collect_detection_regions(
        self,
        images_layout_res: List[List[Dict]],
        np_images: List[np.ndarray],
        images_with_extra_info: List[Tuple]
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """收集各类型检测区域"""
        ocr_res_all_page = []
        table_res_all_page = []
        formula_res_all_page = []
        
        for index, np_img in enumerate(np_images):
            _, _, ocr_enable, _lang, _ = images_with_extra_info[index]
            layout_res = images_layout_res[index]
            
            ocr_res_list, table_res_list, formula_res_list = get_res_list_from_layout_res(layout_res, np_img)
            
            # 复选框检测
            checkbox_res = []
            if self.checkbox_enable:
                checkbox_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
                checkbox_res = checkbox_predict(checkbox_img)
                for res in checkbox_res:
                    poly = [res['bbox'][0], res['bbox'][1], res['bbox'][2], res['bbox'][1],
                            res['bbox'][2], res['bbox'][3], res['bbox'][0], res['bbox'][3]]
                    layout_res.append({
                        'bbox': res['bbox'], 'poly': poly,
                        'category_id': CategoryId.CheckBox,
                        'checkbox': res['text'], 'score': 0.9
                    })

            # OCR 区域
            ocr_res_all_page.append({
                'ocr_res_list': ocr_res_list,
                'lang': _lang,
                'ocr_enable': ocr_enable,
                'np_img': np_img,
                'single_page_mfdetrec_res': formula_res_list,
                'checkbox_res': checkbox_res,
                'layout_res': layout_res,
                'page_idx': index,
            })

            # 表格区域
            for table_res in table_res_list:
                def get_crop_table_img(scale):
                    bbox = [table_res["poly"][0], table_res["poly"][1], table_res["poly"][4], table_res["poly"][5]]
                    bbox = normalize_to_int_bbox(
                        [float(v) / float(scale) for v in bbox]
                    )
                    if bbox is None:
                        return np_img[0:0, 0:0]
                    return get_crop_np_img(bbox, np_img, scale=scale, return_list=True)
                table_img, useful_list = get_crop_table_img(scale=5)
                # table_img, useful_list = crop_img(table_res, np_img)
                # rect_table_img, _ = crop_img(table_res, np_img, layout_shape_mode="rect")
                table_res_all_page.append({
                    'table_res': table_res,
                    'lang': _lang,
                    'table_img': table_img, #矩形框/异型框的表格
                    'rect_table_img': table_img, #矩形框的表格
                    'single_page_mfdetrec_res': formula_res_list,
                    'checkbox_res': checkbox_res,
                    'useful_list': useful_list,
                    'ocr_enable': ocr_enable,
                    'page_idx': index,
                })

            # 公式区域
            for formula_res in formula_res_list:
                formula_crop_res = _expand_formula_crop_res(
                    formula_res, layout_res, np_img.shape, self.formula_bbox_expand_px
                )
                formula_img, _ = crop_img(formula_crop_res, np_img)
                formula_res_all_page.append({
                    'formula_res': formula_res,
                    'lang': _lang,
                    'formula_img': formula_img,
                })

        return ocr_res_all_page, table_res_all_page, formula_res_all_page

    def _run_formula_recognition(self, formula_res_all_page: List[Dict]):
        """执行公式识别"""
        formula_imgs = [d['formula_img'] for d in formula_res_all_page]
        if not formula_imgs:
            return
        formula_results = self.model.formula_model.batch_predict(
            formula_imgs, batch_size=self.formula_base_batch_size
        )
        for d, res in zip(formula_res_all_page, formula_results):
            if res:
                d['formula_res']['latex'] = res
            else:
                logger.warning('latex recognition processing fails')

    def _run_custom_ocr(self, ocr_res_all_page: List[Dict]):
        """
        使用 VL 模型进行 OCR 识别

        VL OCR 直接识别整个文本区域，返回多行文本，不产生单行 span
        """
        # 收集所有需要 OCR 的区域
        all_ocr_regions = []

        for ocr_res_dict in ocr_res_all_page:
            for res in ocr_res_dict['ocr_res_list']:
                # 裁剪图像
                new_image, useful_list = crop_img(res, ocr_res_dict['np_img'])
                bgr_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)

                all_ocr_regions.append({
                    'image': bgr_image,
                    'res': res,
                    'layout_res': ocr_res_dict['layout_res'],
                    'useful_list': useful_list,
                })

        if not all_ocr_regions:
            return

        # 批量 VL OCR
        images = [r['image'] for r in all_ocr_regions]
        ocr_texts = self.model.ocr_model.batch_predict(
            images, batch_size=self.ocr_det_base_batch_size
        )

        # 将结果填充到 layout_res
        for region, text in zip(all_ocr_regions, ocr_texts):
            res = region['res']

            # 构建 VL OCR 结果，使用特殊的 category_id 标识
            vl_ocr_result = {
                'poly': res['poly'],
                'category_id': CategoryId.OcrText,
                'score': 0.95,  # VL 模型置信度较高
                'text': text.strip() if text else '',
                'vl_ocr': True,  # 标识这是 VL OCR 结果
                'original_label': res.get('original_label'),
                'original_order': res.get('original_order'),
                'polygon_points': res.get('polygon_points'),
            }

            region['layout_res'].append(vl_ocr_result)

    def _run_traditional_ocr(
        self,
        atom_model_manager,
        ocr_res_all_page: List[Dict],
        pdf_dict_list: List[Dict],
        scale_list: List[float],
    ):
        """传统 OCR 处理流程 (det + rec)"""

        # PDF 文本提取模式
        if self.use_det_mode != 'ocr':
            _extract_text_from_pdf(ocr_res_all_page, pdf_dict_list, scale_list)

        # OCR 检测处理
        _run_ocr_det_batch(ocr_res_all_page, atom_model_manager, self.ocr_config)

    def _run_table_recognition(
        self,
        atom_model_manager,
        table_res_all_page: List[Dict],
        pdf_dict_list: List[Dict],
        scale_list: List[float]
    ):
        """执行表格识别"""
        if isinstance(self.model.table_model, CustomBaseModel):
            table_imgs = []
            fill_image_res_list = []
            for table_res_dict in table_res_all_page:
                page_idx = table_res_dict['page_idx']
                page_dict = pdf_dict_list[page_idx]
                scale = scale_list[page_idx]
                fill_image_res = []
                if self.table_image_enable:
                    fill_image_res = extract_table_fill_image(
                        page_dict, table_res_dict, scale, self.table_extract_original_image
                    )
                table_imgs.append(table_res_dict["table_img"])
                fill_image_res_list.append(fill_image_res)

            if table_imgs:
                table_results = self.model.table_model.batch_predict(
                    table_imgs, fill_image_res_list=fill_image_res_list
                )
                for table_res_dict, table_result in zip(table_res_all_page, table_results):
                    table_res_dict['table_res'].pop('layout_image_list', None)
                    if table_result:
                        table_res_dict['table_res']['html'] = table_result
        else:
            # 传统模式表格识别
            self._run_traditional_table_recognition(atom_model_manager, table_res_all_page, pdf_dict_list, scale_list)
    def _run_traditional_table_recognition(
        self,
        atom_model_manager,
        table_res_all_page: List[Dict],
        pdf_dict_list: List[Dict],
        scale_list: List[float]
    ):
        """传统表格识别 —— 跨表格 batch 分类器+检测+OCR-rec+结构推理"""
        table_res_grouped = {}
        for x in table_res_all_page:
            table_res_grouped.setdefault(x["page_idx"], []).append(x)
        total_tables = sum(len(tables) for tables in table_res_grouped.values())

        # Phase 0: compute global max det image size for GPU uniform padding
        RES_STRIDE = 64
        max_h = max_w = 0
        for page_idx, table_list in table_res_grouped.items():
            for td in table_list:
                td["table_img"] = td["rect_table_img"]
                h, w = td["rect_table_img"].shape[:2]
                max_h = max(max_h, ((h + RES_STRIDE - 1) // RES_STRIDE) * RES_STRIDE)
                max_w = max(max_w, ((w + RES_STRIDE - 1) // RES_STRIDE) * RES_STRIDE)

        uniform_det_size = (max_h, max_w) if max_h > 0 else None

        # Phase 1a: batch table classifier
        table_model = atom_model_manager.get_atom_model(
            atom_model_name='table', lang='ch', ocr_config=self.ocr_config,
            table_config=self.table_config,
        )
        precomputed_classes = {}
        if hasattr(table_model, 'table_cls'):
            try:
                all_imgs = []
                idx_map = []
                for page_idx, table_list in table_res_grouped.items():
                    for td in table_list:
                        td["table_img"] = td["rect_table_img"]
                        all_imgs.append(cv2.cvtColor(np.asarray(td["table_img"]), cv2.COLOR_RGB2BGR))
                        idx_map.append((page_idx, id(td)))
                if all_imgs:
                    cls_results, cls_scores, _ = table_model.table_cls(all_imgs, batch_size=32, return_scores=True)
                    for (pi, tid), cls, score in zip(idx_map, cls_results, cls_scores):
                        precomputed_classes[tid] = (cls, score)
            except Exception:
                pass

        # Phase 1b: per-table preprocessing
        batch_entries = []
        with tqdm(total=total_tables, desc="Table Predict") as pbar:
            for page_idx, table_list in table_res_grouped.items():
                page_dict = pdf_dict_list[page_idx]
                scale = scale_list[page_idx]
                for table_res_dict in table_list:
                    table_res_dict["table_img"] = table_res_dict["rect_table_img"]
                    pc = precomputed_classes.get(id(table_res_dict))
                    result = _prepare_table_data(
                        table_res_dict, page_dict, scale, atom_model_manager,
                        self.table_config, self.ocr_config,
                        skip_table_rec=True,
                        uniform_det_size=uniform_det_size,
                        precomputed_table_class=pc,
                    )
                    if result is not None:
                        batch_entries.append(result)
                    pbar.update(1)

        # Phase 1c: cross-table batch OCR-rec for deferred entries
        deferred_indices = []
        all_crops = []
        for idx, entry in enumerate(batch_entries):
            pk = entry[2]
            deferred = pk.pop('_deferred_rec', None)
            if deferred is None:
                continue
            ocr_model, bgr_image, det_res, table_use_word_box, ocr_cfg = deferred
            for det_box in det_res:
                crop = get_rotate_crop_image(bgr_image, np.asarray(det_box, dtype=np.float32))
                all_crops.append((crop, idx))
            deferred_indices.append(idx)

        if all_crops:
            crop_imgs = [c[0] for c in all_crops]
            max_h = max(img.shape[0] for img in crop_imgs)
            max_w = max(img.shape[1] for img in crop_imgs)

            ocr_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.OCR,
                lang=batch_entries[deferred_indices[0]][2].get('_lang', 'ch'),
                ocr_config=self.ocr_config,
            )
            rec_model = getattr(getattr(ocr_model, 'ocr_engine', None), 'text_rec', None)
            GPU_BATCH = 128
            if rec_model is not None and hasattr(rec_model, 'cfg') and hasattr(rec_model.cfg, 'rec_batch_num'):
                rec_model.cfg.rec_batch_num = min(GPU_BATCH, len(crop_imgs))

            # 按批 pad + 推理：全量 pad 到全局最大尺寸再一次性推理在表格
            # 较多时可达数十 GB（数千 cell × max 尺寸），必须分批降低峰值。
            rec_results_raw = []
            for start in range(0, len(crop_imgs), GPU_BATCH):
                batch = crop_imgs[start:start + GPU_BATCH]
                padded_imgs = []
                for img in batch:
                    h, w = img.shape[:2]
                    if h < max_h or w < max_w:
                        p = np.ones((max_h, max_w, 3), dtype=np.uint8) * 255
                        p[:h, :w] = img
                        padded_imgs.append(p)
                    else:
                        padded_imgs.append(img)
                try:
                    batch_res = ocr_model.ocr(padded_imgs, det=False, tqdm_enable=False)[0]
                except Exception as exc:
                    logger.warning(f'table OCR-rec batch failed, use empty results: {exc}')
                    batch_res = []
                if len(batch_res) != len(batch):
                    batch_res = (batch_res + [('', 0.0)] * len(batch))[:len(batch)]
                rec_results_raw.extend(batch_res)

            idx_to_crops = defaultdict(list)
            for (crop_img, tbl_idx), rec_res in zip(all_crops, rec_results_raw or []):
                idx_to_crops[tbl_idx].append(rec_res or ('', 0.0))

            for tbl_idx in deferred_indices:
                pk = batch_entries[tbl_idx][2]
                recs = idx_to_crops.get(tbl_idx, [])
                dt_boxes = [np.asarray(det_box, dtype=np.float32) for det_box in
                            pk.get('_det_res', [])]
                if recs:
                    texts = [r[0] if isinstance(r, (list, tuple)) else '' for r in recs]
                    scores = [float(r[1]) if isinstance(r, (list, tuple)) else 0.0 for r in recs]
                    pk['ocr_result'] = [dt_boxes, texts, scores]
                else:
                    pk['ocr_result'] = []

        # Phase 2: batched model inference grouped by wired/wireless
        _run_batched_table_inference(self.table_config, batch_entries)


    def _run_seal_ocr(
        self,
        atom_model_manager,
        np_images,
        images_layout_res,
    ):
        """印章 处理流程"""
        seal_ocr_items = []
        for index, np_img in enumerate(np_images):
            layout_res = images_layout_res[index]
            for layout_re in layout_res:
                if 'seal' == layout_re.get("original_label"):
                    seal_img, _ = crop_img(layout_re, np_img)
                    seal_crop_bgr = cv2.cvtColor(seal_img, cv2.COLOR_RGB2BGR)
                    seal_ocr_items.append((seal_crop_bgr, layout_re))

        seal_ocr_model = None
        for seal_crop_bgr, layout_re in tqdm(seal_ocr_items, desc="Seal Predict"):
            if (isinstance(self.model.ocr_model, CustomBaseModel)
                    and 'is_seal' in inspect.signature(self.model.ocr_model.batch_predict).parameters):
                seal_texts = self.model.ocr_model.batch_predict([seal_crop_bgr], is_seal=True)
                seal_texts = seal_texts[0].split('\n')
            else:
                if seal_ocr_model is None:
                    seal_ocr_model = atom_model_manager.get_atom_model(
                        atom_model_name=AtomicModel.OCR,
                        is_seal=True,
                    )
                seal_ocr_res = seal_ocr_model.ocr(seal_crop_bgr, det=True, rec=True)[0]
                if not seal_ocr_res:
                    continue
                seal_texts = []
                for seal_item in seal_ocr_res:
                    if not seal_item or len(seal_item) != 2:
                        continue
                    rec_result = seal_item[1]
                    if not rec_result or len(rec_result) < 1:
                        continue
                    rec_text = rec_result[0]
                    if rec_text:
                        seal_texts.append(rec_text)
            layout_re["text"] = seal_texts
