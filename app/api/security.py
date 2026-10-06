"""
Проверка ключа доступа.

Включается только если задана переменная LPR_API_KEYS. Пусто означает, что
проверки нет: так удобно разрабатывать, и сервис пишет об этом в журнал при
запуске, чтобы открытый режим не уехал в кластер незамеченным.

Ключи сравниваются через hmac.compare_digest. Обычное сравнение строк
прекращается на первом несовпавшем символе, и по времени ответа ключ можно
подобрать посимвольно; compare_digest тратит одинаковое время на любой ключ.

Ключи берутся из окружения, то есть в кластере из Vault через ExternalSecret.
Ни в коде, ни в файлах репозитория их нет и быть не должно: ключ, попавший в
историю git один раз, остаётся в ней и после удаления из файлов.
"""
from __future__ import annotations

import hmac

from fastapi import Header, HTTPException

from app.config import get_settings


def require_key(authorization: str | None = Header(default=None)) -> None:
    """Зависимость FastAPI. Молчит, если ключ верный или проверка выключена."""
    settings = get_settings()
    keys = settings.api_keys
    if not keys:
        return

    if not authorization:
        raise HTTPException(status_code=401, detail={
            "ok": False, "error_code": "unauthorized",
            "error": "пришлите заголовок Authorization: Bearer <ключ>"})

    scheme, _, key = authorization.partition(" ")
    if scheme.lower() != "bearer" or not key.strip():
        raise HTTPException(status_code=401, detail={
            "ok": False, "error_code": "unauthorized",
            "error": "ожидается схема Bearer"})

    candidate = key.strip().encode("utf-8")
    if not any(hmac.compare_digest(candidate, known.encode("utf-8")) for known in keys):
        raise HTTPException(status_code=401, detail={
            "ok": False, "error_code": "unauthorized", "error": "ключ не принят"})
