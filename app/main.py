"""
Приложение FastAPI: распознавание номера пледжированной машины по кадру с
камеры телефона, для активности «Выезд на место».

Что делает сервис. Принимает кадр, находит на нём номерной знак, читает
символы, устойчиво их подтверждает по нескольким кадрам и отдаёт
подтверждённый номер. Поиск машины по номеру это отдельный запрос.

Что сервис НЕ делает. Он не обращается к OCRM. Справочник машин здесь
тестовый, из файла с придуманными записями, и назначение у него одно:
показать сценарий до конца. Подключение настоящего OCRM это замена одного
модуля (app/api/ocrm.py), форма ответов под это уже рассчитана.

Модели грузятся один раз при старте, в lifespan. Пока загрузка не прошла,
/health/ready отвечает 503, и кластер не пускает трафик на этот экземпляр.
Загрузка занимает единицы секунд, но делать её на первом запросе означало бы
отдать первому клиенту таймаут.

    GET  /                        кто это и какая версия
    GET  /health/live             процесс жив
    GET  /health/ready            модели загружены, можно присылать кадры
    POST /api/v1/recognize        кадр -> подтверждённый номер
    GET  /api/v1/vehicle/{plate}  машина по номеру (тестовый справочник)
    POST /api/v1/visit            создать выезд (тестовый справочник)
    GET  /docs                    Swagger
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import state
from app.api import health, ocrm, recognition
from app.config import get_settings
from app.lpr import ocrm_stub
from app.lpr.gateway_ocr import build_reader
from app.lpr.sessions import SessionStore

ROOT = Path(__file__).resolve().parents[1]

logger = logging.getLogger("lpr")


def _limit_thread_pools(threads: int) -> None:
    """Ограничивает число потоков математических библиотек.

    ONNX Runtime и OpenCV по умолчанию берут столько потоков, сколько
    ядер видят на УЗЛЕ, а не сколько разрешено контейнеру. При квоте в одно
    ядро это десятки потоков на одно ядро: время уходит на переключение между
    ними, и запрос отвечает медленнее, чем в один поток. Переменные надо
    выставить до импорта библиотек, поэтому это делается здесь, в начале
    lifespan, а не в конфигурации.
    """
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, str(threads))
    try:
        import cv2
        cv2.setNumThreads(threads)
    except Exception:  # noqa: BLE001
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    _limit_thread_pools(settings.ort_threads)

    # Правило подтверждения здесь всегда одно: по числу совпадений. Модель
    # через шлюз не возвращает уверенности, а подставлять вместо неё число
    # нельзя: при единице каждое чтение стало бы «очень уверенным», номер
    # подтверждался бы с первого раза, и слой устойчивости исчез бы
    # незаметно. Числа правила в app/lpr/recognizer.py.
    #
    # single_row_only: вырез не делится на строки. Модель читает обе строки
    # сама и отдаёт номер одним текстом, разделить его нечем, а голосование
    # по строкам требует их отдельно.
    state.sessions = SessionStore(
        ttl_sec=settings.session_ttl_sec,
        max_sessions=settings.max_sessions,
        plate_hold_sec=settings.plate_hold_sec,
        policy="agreement",
        single_row_only=True,
    )
    logger.info("символы читает шлюз, подтверждение по числу совпадений")

    loaded = ocrm_stub.load(settings.test_ocrm_file)
    if loaded:
        logger.info("тестовый справочник: %d машин (придуманные данные, не OCRM)",
                    loaded)
    else:
        logger.warning("тестовый справочник не загружен, поиск машины ответит "
                       "database=not_configured")

    if not settings.auth_enabled:
        logger.warning("ключи доступа не заданы: кадры примет любой клиент. "
                       "В кластере LPR_API_KEYS приходит из Vault")

    try:
        from app.lpr.engine import InProcessEngine

        state.engine = InProcessEngine(
            model_path=settings.model_path,
            reader=build_reader(settings),
            device=settings.device,
            detect_conf=settings.detect_conf,
        )
        logger.info("модели загружены: детектор %s, распознаватель %s",
                    state.engine.yolo_info, state.engine.ocr_info)
    except Exception as exc:  # noqa: BLE001
        # Не падаем. Процесс, который умер на старте, кластер перезапускает по
        # кругу, и причину видно только в журнале предыдущей попытки. Живой
        # процесс с честным 503 на /health/ready и причиной в ответе
        # диагностируется за один запрос.
        state.startup_error = repr(exc)
        logger.exception("модели не загрузились, сервис остаётся не готов")

    try:
        yield
    finally:
        if state.engine is not None:
            state.engine.close()


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description=__doc__,
    version=settings.version,
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(recognition.router, prefix="/api/v1")
app.include_router(ocrm.router, prefix="/api/v1")


@app.exception_handler(StarletteHTTPException)
async def http_error(request, exc: StarletteHTTPException):
    """Ошибки в одной форме: ok, error, error_code.

    Клиент уже написан под эти поля, и FastAPI по умолчанию отдаёт {"detail":
    ...}. Если detail уже наша структура, отдаём как есть.
    """
    detail = exc.detail
    if isinstance(detail, dict) and "error_code" in detail:
        return JSONResponse(status_code=exc.status_code, content=detail)
    return JSONResponse(status_code=exc.status_code, content={
        "ok": False, "error_code": "http_error", "error": str(detail)})


@app.exception_handler(RequestValidationError)
async def validation_error(request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={
        "ok": False, "error_code": "bad_request", "error": exc.errors()})


@app.get("/", tags=["service"], summary="Кто это и какая версия")
def root():
    return {
        "message": "сервис распознавания номерных знаков",
        "service": settings.app_name,
        "version": settings.version,
        "environment": settings.environment,
        # Прямо в ответе, чтобы это нельзя было принять за интеграцию с OCRM.
        "ocrm": "тестовый справочник с придуманными машинами, не OCRM банка",
    }


if settings.serve_demo_page:
    @app.get("/demo", include_in_schema=False)
    def demo():
        """Страница съёмки с телефона. Нужна только для демонстрации.

        Камеру браузер отдаёт странице лишь по HTTPS, так что открывать её
        надо через ingress, а не по адресу пода.
        """
        page = ROOT / "web" / "demo.html"
        if not page.is_file():
            return JSONResponse(status_code=404, content={
                "ok": False, "error_code": "not_found", "error": "страницы нет в образе"})
        return FileResponse(page, media_type="text/html")
