"""
Тестовый справочник машин и создание выезда.

    GET  /api/v1/vehicle/{plate}
    POST /api/v1/visit

Это НЕ OCRM банка, а файл с придуманными машинами. Он нужен, чтобы сценарий
показывался до конца: после подтверждения номера на экране либо карточка, либо
прямая фраза, что такого номера в базе нет.

Главное в этих двух ручках, и причина, по которой они отделены от
распознавания: ненайденная машина это код 200 с found=false, а не 404. Если
отвечать 404, клиент покажет сбой, и сотрудник решит, что номер не прочитан,
хотя прочитан он верно. Распознавание и поиск в базе это два разных этапа, и
их исходы не должны выглядеть одинаково.

При переходе на настоящий OCRM меняется здесь: вместо ocrm_stub вызов его API.
Форма ответа остаётся, клиент уже написан под неё.
"""
from __future__ import annotations

import re

from fastapi import APIRouter, Depends, HTTPException, Path

from app.api.security import require_key
from app.lpr import ocrm_stub
from app.models.ocrm import VehicleResponse, VisitRequest, VisitResponse

router = APIRouter(tags=["ocrm"], dependencies=[Depends(require_key)])

# Единственная форма номера, которую сервис подтверждает, значит единственная,
# которую имеет смысл искать: три цифры, три буквы, две цифры.
PLATE_RE = re.compile(r"\d{3}[A-Z]{3}\d{2}")


def _check_plate(plate: str) -> str:
    plate = plate.upper().strip()
    if not PLATE_RE.fullmatch(plate):
        raise HTTPException(status_code=400, detail={
            "ok": False, "error_code": "bad_request",
            "error": "номер: три цифры, три буквы, две цифры"})
    return plate


@router.get("/vehicle/{plate}", response_model=VehicleResponse,
            summary="Найти машину по номеру в тестовом справочнике")
def vehicle(plate: str = Path(description="Например 322BDM02", examples=["322BDM02"])):
    answer = ocrm_stub.lookup(_check_plate(plate))
    answer["ok"] = True
    return answer


@router.post("/visit", response_model=VisitResponse, status_code=201,
             summary="Создать активность «Выезд на место»")
def visit(payload: VisitRequest):
    created = ocrm_stub.create_visit(_check_plate(payload.plate), payload.note)
    created["ok"] = True
    return created
