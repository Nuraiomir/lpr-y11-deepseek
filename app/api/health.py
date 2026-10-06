"""
Проверки состояния для кластера.

Две пробы, и разница между ними важна:

  /health/live    процесс жив. Отвечает 200 и пока модели ещё грузятся: иначе
                  кластер решит, что контейнер завис, и убьёт его на середине
                  загрузки, и так по кругу. 503 только если загрузка упала и
                  перезапуск это единственное, что может помочь.

  /health/ready   сервис готов принимать кадры. Пока модели не загружены,
                  503, и балансировщик на этот экземпляр трафик не пускает.
                  Загрузка занимает секунды, и отдать в это время 500 клиенту
                  было бы хуже, чем подождать.

Полезно помнить, что модель детектора в кластере приходит из смонтированного
каталога, то есть «не готов» может означать и «каталог пустой». Поэтому в
ответ включена причина: по ней видно, чинить сервис или монтирование.
"""
from __future__ import annotations

from fastapi import APIRouter, Response

from app import state

router = APIRouter(tags=["health"])


@router.get("/health/live", summary="Процесс жив")
def live(response: Response):
    if state.startup_error is not None:
        response.status_code = 503
        return {"status": "error", "reason": state.startup_error}
    return {"status": "ok" if state.engine is not None else "loading"}


@router.get("/health/ready", summary="Готов принимать кадры")
def ready(response: Response):
    if state.engine is None:
        response.status_code = 503
        return {
            "status": "not_ready",
            "reason": state.startup_error or "модели ещё загружаются",
        }
    return {
        "status": "ready",
        "detector": state.engine.yolo_info,
        "ocr": state.engine.ocr_info,
        "sessions": state.sessions.count() if state.sessions else 0,
    }
