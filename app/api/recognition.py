"""
Приём кадра и распознавание.

    POST /api/v1/recognize?session_id=<id>

Кадр принимается двумя способами, и оба равноправны:

    Content-Type: image/jpeg            тело запроса это сам кадр;
    multipart/form-data, поле image     как обычная загрузка файла.

Первый способ дешевле: со страницы съёмки кадр уходит как есть, без обёртки
multipart, а кадры идут несколько раз в секунду. Второй удобнее для curl и
для Swagger, и его же ждут те, кто читал шаблон. Форма ответа одинаковая.

Порядок работы внутри одного запроса:

    кадр -> детектор -> вырез -> распознаватель -> нормализация ->
    голосование -> подтверждённый номер

Поиск в базе здесь не делается нарочно. Это отдельный запрос
(GET /api/v1/vehicle), и разделение не формальное: «номер прочитан» и «машина
найдена» это два разных исхода, и смешать их в одном ответе означает рано или
поздно показать сотруднику «не распознано» вместо «такой машины нет в базе».

Кадры одной сессии обрабатываются по одному. Это требование корректности, а
не настройка: голосование копит прочитанное по времени, и два кадра одной
сессии, посчитанные одновременно, перемешали бы голоса.
"""
from __future__ import annotations

import re
import time

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile

from app import state
from app.api.security import require_key
from app.config import get_settings
from app.lpr.engine import EngineError, EngineStartupError
from app.models.recognition import RecognizeResponse

router = APIRouter(tags=["recognition"], dependencies=[Depends(require_key)])

# Идентификатор сессии приходит от клиента, значит это недоверенные данные.
# Набор символов ограничен, чтобы он не попадал в журнал и в имена с чем-то
# неожиданным.
SESSION_ID_RE = re.compile(r"[A-Za-z0-9._:@-]{1,128}")

_PROFILE_MS_FIELDS = ("yolo_ms", "ocr_total_ms", "ocr_square_ms", "ocr_normal_ms",
                      "crop_ms", "voting_ms", "total_ms")


def _fail(status: int, code: str, message: str):
    return HTTPException(status_code=status,
                         detail={"ok": False, "error_code": code, "error": message})


async def _frame_bytes(request: Request, upload: UploadFile | None, limit: int) -> bytes:
    """Кадр из тела запроса или из поля формы. Проверяет размер до чтения."""
    if upload is not None:
        data = await upload.read()
        if len(data) > limit:
            raise _fail(413, "payload_too_large",
                        f"кадр {len(data)} байт, предел {limit}")
        return data

    raw_length = request.headers.get("content-length")
    if raw_length is None:
        raise _fail(400, "bad_request", "нужен заголовок Content-Length")
    try:
        length = int(raw_length)
    except ValueError:
        raise _fail(400, "bad_request", "Content-Length не число") from None
    if length <= 0:
        raise _fail(400, "bad_request", "пустое тело запроса")
    if length > limit:
        # Проверка до чтения: иначе клиент успеет прислать сто мегабайт, и
        # отказ обойдётся дороже, чем обработка кадра.
        raise _fail(413, "payload_too_large", f"кадр {length} байт, предел {limit}")
    return await request.body()


@router.post("/recognize", response_model=RecognizeResponse,
             summary="Один кадр: найти номер, прочитать, проголосовать")
async def recognize(
    request: Request,
    session_id: str = Query("default",
                            description="Своё состояние голосования на каждую сессию. "
                                        "Один телефон это одна сессия"),
    profile: bool = Query(False, description="Добавить разбивку времени по этапам"),
    t: float | None = Query(None,
                            description="Метка времени для голосования вместо "
                                        "серверных часов. Только для оффлайновых "
                                        "прогонов: окна голосования считаются в "
                                        "секундах, и быстрый прогон иначе уложит в "
                                        "окно больше кадров и будет выглядеть лучше "
                                        "просто потому, что быстрее"),
    image: UploadFile | None = File(None,
                                    description="Кадр как файл. Либо так, либо "
                                                "кадр в теле запроса"),
):
    if state.engine is None:
        raise _fail(503, "engine_unavailable",
                    state.startup_error or "модели ещё загружаются")
    if not SESSION_ID_RE.fullmatch(session_id):
        raise _fail(400, "bad_request",
                    "session_id: от 1 до 128 символов из букв, цифр и . _ : @ -")

    settings = get_settings()
    t_req = time.perf_counter()
    data = await _frame_bytes(request, image, settings.max_frame_bytes)
    t_read = time.perf_counter()

    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise _fail(400, "decode_failed", "тело запроса не разбирается как картинка")
    t_decoded = time.perf_counter()

    session = state.sessions.get(session_id)
    try:
        with session.lock:
            vote_t = t if t is not None else time.monotonic()
            t0 = time.perf_counter()
            result = session.recognizer.process_frame(frame, state.engine, vote_t)
            proc_ms = (time.perf_counter() - t0) * 1000.0
            prof = dict(session.recognizer.last_profile or {}) if profile else None
    except (EngineError, EngineStartupError) as exc:
        raise _fail(503, "engine_unavailable", str(exc)) from exc

    result["session_id"] = session_id
    result["processing_time_ms"] = round(proc_ms, 1)
    result["frame_size"] = [int(frame.shape[1]), int(frame.shape[0])]
    result["ocr_variant_mode"] = state.engine.ocr_variant_mode

    if prof is not None:
        prof["body_read_ms"] = round((t_read - t_req) * 1000.0, 2)
        prof["jpeg_decode_ms"] = round((t_decoded - t_read) * 1000.0, 2)
        prof["request_bytes"] = len(data)
        for key in _PROFILE_MS_FIELDS:
            if key in prof:
                prof[key] = round(prof[key], 2)
        prof["worker_detail"] = [
            {k: round(v, 2) if isinstance(v, float) else v for k, v in d.items()}
            for d in prof.get("worker_detail", [])
        ]
        result["profile"] = prof

    return {"ok": True, **result}
