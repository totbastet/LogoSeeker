import logging
from typing import Dict, Any

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from classifier import LogoClassifier
from utils import crop_bounding_box, resize_image, top_n_closest_embs
import config
import json
import torch

logger = logging.getLogger(__name__)


class LogoPipeline:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Считаем на: {self.device}")

        logger.info("Инициализация детектора YOLO (путь: %s)...", config.DETECTOR_PATH)

        self.detector = YOLO(config.DETECTOR_PATH)
        self.conf_threshold = config.DETECTOR_THRESHOLD

        if self.device == "cuda":
            self.detector.to("cuda")
            logger.info("Детектор YOLO загружен на GPU")
        else:
            logger.info("Детектор YOLO загружен на CPU")

        logger.info("Инициализация классификатора (Embeddings)...")
        self.classifier = LogoClassifier()
        self.restricted_db = np.load(config.RESTRICTED_EMB_BASE_PATH)
        self.competitors_db = np.load(config.COMPETITORS_EMB_BASE_PATH)
        self.employers_db = np.load(config.EMPLOYERS_EMB_BASE_PATH)

        with open(config.RESTRICTED_EMB_MAP_PATH, "r") as f:
            self.restricted_db_map = json.load(f)
        with open(config.COMPETITORS_EMB_MAP_PATH, "r") as f:
            self.competitors_db_map = json.load(f)
        with open(config.EMPLOYERS_EMB_MAP_PATH, "r") as f:
            self.employers_db_map = json.load(f)

    def _search_with_mapping(self, emb, database, database_map, top_n=1):
        """
        Ищет top_n ближайших эмбеддингов в базе с маппингом

        Args:
            emb (np.array): анализируемый эмбеддинг [dim]
            database (np.array): матрциа с веекторами эталонов [base_size, dim]
            database_map (dict): словарь с мапингом
            top_n (int): количество ближайших соседей

        Returns:
            list of dict: [{'name': name, 'similarity': sim}, ...]
        """
        positions, similarities = top_n_closest_embs(emb, database, top_n)
        if len(positions) == 0:
            return []

        results = []
        for pos, sim in zip(positions, similarities):
            name = database_map.get(str(pos), f"unknown_{pos}")

            results.append({"name": name, "similarity": float(sim)})

        return results

    def _process_match(
        self, embedding, database, db_map, threshold, category, coords_yolo, det_conf
    ):
        """
        Функция ищет ближайший эмбеддинг в указанной базе и сравнивает его схожесть
        с заданным порогом. При превышении порога формирует словарь с результатами.

        Args:
            embedding (np.ndarray): Вектор эмбеддинга логотипа размерности [dim]
            database (np.ndarray): Матрица эталонных эмбеддингов размером [base_size, dim].
            db_map (dict): Словарь маппинга индексов из database к названиям логотипов.
                       Формат: {индекс: "название_логотипа"}.
            threshold(float): Порог схожести (0.0-1.0).
            category (str): Категория базы данных restricted|employer|competitor
            coords_yolo (list): Нормализованные координаты bounding box от YOLO.
            det_conf (float): Уверенность детектора YOLO в обнаружении логотипа (0.0 - 1.0).

        Returns:
            dict or None: Если найдено совпадение выше порога, возвращает словарь с полями:
            - box (list): Координаты бокса, округленные до 4 знаков
            - detector_confidence (float): Уверенность детектора
            - best_match (str): Название совпавшего логотипа
            - similarity_score (float): Значение схожести с совпавшим логотипом
            - verdict (str): Вердикт ('blocked' или 'manual_moderation')
            - logo_category (str): Категория логотипа (переданная в category)

        Если совпадение не найдено (ближайший neighbor ниже порога или база пуста),
        возвращает None.
        """
        best_match = self._search_with_mapping(embedding, database, db_map, top_n=1)
        if best_match and best_match[0]["similarity"] > threshold:
            if category == "restricted":
                verdict = "blocked"
            elif category == "competitor":
                verdict = "manual_moderation"
            else:
                verdict = "ok"

            return {
                "box": [round(c, 4) for c in coords_yolo],
                "detector_confidence": round(det_conf, 4),
                "best_match": best_match[0]["name"],
                "similarity_score": round(best_match[0]["similarity"], 4),
                "verdict": verdict,
                "logo_category": category,
            }
        return None

    def process_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Основной метод обработки изображения. Выполняет полный цикл модерации:
        декодирование -> ресайз -> детекция (YOLO) -> извлечение признаков (DINO) ->
        мэтчинг по базам.

        Логика поиска каскадная: сначала проверяется база 'restricted', затем
        'competitor', затем 'employer'. При первом же превышении порога совпадения
        поиск для конкретного логотипа останавливается.

        Args:
            image_bytes (bytes): Сырые байты изображения, полученные из HTTP-запроса.

        Returns:
            Dict[str, Any]: JSON с результатами модерации.
                            Формат:
                            {
                                "status": "success" | "error",
                                "found_logos": int (количество найденных лого),
                                "details": list[dict] (список вердиктов по каждому логотипу)
                            }
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_cv is None:
            logger.error("Не удалось декодировать байты в изображение.")
            return {"status": "error", "message": "Не удалось прочитать изображение"}

        DETECTION_SIZE = config.DETECTION_SIZE
        h, w = img_cv.shape[:2]
        if max(w, h) > DETECTION_SIZE:
            scale = min(DETECTION_SIZE / w, DETECTION_SIZE / h)
            new_w, new_h = int(w * scale), int(h * scale)
            img_resized = cv2.resize(
                img_cv, (new_w, new_h), interpolation=cv2.INTER_AREA
            )
        else:
            img_resized = img_cv

        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        results_data = []

        det_results = self.detector(
            img_pil, conf=config.DETECTOR_THRESHOLD, verbose=False
        )
        for r in det_results:
            for box in r.boxes:
                det_conf = float(box.conf[0])

                coords_abs = box.xyxy[0].tolist()
                coords_yolo = box.xywhn[0].tolist()
                try:
                    crop_img = crop_bounding_box(img_pil, coords_abs)
                    processed_crop = resize_image(crop_img)
                    if isinstance(processed_crop, np.ndarray):
                        processed_crop = Image.fromarray(processed_crop)

                    embedding = self.classifier.get_embedding(processed_crop)

                    categories = [
                        (
                            "restricted",
                            self.restricted_db,
                            self.restricted_db_map,
                            config.RESTRICTED_SIMILARITY_THRESHOLD,
                        ),
                        (
                            "competitor",
                            self.competitors_db,
                            self.competitors_db_map,
                            config.COMPETITORS_SIMILARITY_THRESHOLD,
                        ),
                        (
                            "employer",
                            self.employers_db,
                            self.employers_db_map,
                            config.EMPLOYERS_SIMILARITY_THRESHOLD,
                        ),
                    ]

                    for category, db, db_map, threshold in categories:
                        temp = self._process_match(
                            embedding,
                            db,
                            db_map,
                            threshold,
                            category,
                            coords_yolo,
                            det_conf,
                        )
                        if temp:
                            results_data.append(temp)
                            break
                    else:
                        results_data.append(
                            {
                                "box": [round(c, 4) for c in coords_yolo],
                                "detector_confidence": round(det_conf, 4),
                                "best_match": "unknown",
                                "similarity_score": 0.0,
                                "verdict": "ok",
                                "logo_category": "unknown",
                            }
                        )

                except Exception as e:
                    logger.warning("Ошибка при обработке кропа: %s", e)
                    continue

        return {
            "status": "success",
            "found_logos": sum(len(r.boxes) for r in det_results),
            "details": results_data,
        }
