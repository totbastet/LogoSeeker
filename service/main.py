import logging
from contextlib import asynccontextmanager
from enum import Enum
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ml_pipeline import LogoPipeline

logger = logging.getLogger(__name__)


class VerdictStatus(str, Enum):
    OK = "ok"
    BLOCKED = "blocked"
    MANUAL_MODERATION = "manual_moderation"


class LogoDetail(BaseModel):
    box: List[float] = Field(..., description="Координаты BBox [x, y, w, h]")
    detector_confidence: float
    best_match: str
    similarity_score: float
    verdict: VerdictStatus
    logo_category: str


class ModerationResponse(BaseModel):
    status: str
    found_logos: int
    details: List[LogoDetail]
    overall_status: VerdictStatus


def calculate_overall_status(details: List[dict]) -> VerdictStatus:
    """Вычисляет общий статус"""
    overall = VerdictStatus.OK
    for detail in details:
        verdict = detail.get("verdict")
        if verdict == VerdictStatus.BLOCKED.value:
            return VerdictStatus.BLOCKED
        elif verdict == VerdictStatus.MANUAL_MODERATION.value:
            overall = VerdictStatus.MANUAL_MODERATION
    return overall


def get_pipeline(request: Request) -> LogoPipeline:
    """Возвращает пайплайн из состояния приложения в момент HTTP-запроса"""
    return request.app.state.pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Инициализация LogoPipeline")
    app.state.pipeline = LogoPipeline()
    yield
    logger.info("Очистка ресурсов LogoPipeline")
    del app.state.pipeline


app = FastAPI(title="LogoSeeker API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post(
    "/api/v1/moderate",
    response_model=ModerationResponse,
    summary="Модерация изображения",
    description="Принимает файл изображения, прогоняет через YOLO+DINOv2 и возвращает вердикт.",
)
async def moderate_image(
    file: UploadFile = File(...), pipeline: LogoPipeline = Depends(get_pipeline)
):
    if not (file.content_type and file.content_type.startswith("image/")):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")

    try:
        image_bytes = await file.read()
    except Exception as e:
        logger.error("Ошибка чтения файла %s: %s", file.filename, e)
        raise HTTPException(
            status_code=400, detail="Не удалось прочитать загруженный файл"
        )

    try:
        ml_result = pipeline.process_image(image_bytes)
        ml_result["overall_status"] = calculate_overall_status(
            ml_result.get("details", [])
        )
        return ml_result

    except Exception as e:
        logger.error(
            "Ошибка ML-пайплайна при обработке %s: %s", file.filename, e, exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="Внутренняя ошибка сервера при обработке изображения",
        )
