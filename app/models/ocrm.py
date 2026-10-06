"""
Ответы тестового справочника машин.

Это НЕ OCRM банка. Это файл с придуманными машинами, который загружается при
старте, и нужен он затем, чтобы сценарий можно было показать до конца: после
подтверждения номера на экране либо карточка машины, либо прямая фраза, что
такого номера в базе нет.

Поле found отвечает на вопрос «есть ли машина в базе», и ненайденная машина
это нормальный ответ, а не ошибка. Поэтому запрос отвечает кодом 200 с
found=false, а не 404: 404 клиент показал бы как сбой, и сотрудник решил бы,
что номер не распознан.

При переходе на настоящий OCRM меняется реализация поиска, а эта форма ответа
остаётся: клиент уже написан под неё.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class VehicleResponse(BaseModel):
    ok: bool = True
    found: bool = Field(
        description="Машина есть в справочнике. false это полноценный ответ: "
                    "номер распознан, машины в базе нет")
    plate: str
    database: str = Field(
        description="test справочник загружен, not_configured файла нет. "
                    "Во втором случае found=false говорит только об этом")
    # Остальные поля приходят из записи как есть и зависят от того, что в ней
    # заполнено. Жёсткая схема здесь была бы вымыслом: настоящий OCRM отдаёт
    # свой набор полей, и подогнать его под придуманный сейчас нельзя.
    model_config = {"extra": "allow"}

    borrower: dict[str, Any] | None = None
    credit: dict[str, Any] | None = None
    vehicle: dict[str, Any] | None = None
    collateral: dict[str, Any] | None = None
    organization: dict[str, Any] | None = None


class VisitRequest(BaseModel):
    plate: str = Field(
        description="Номер в формате три цифры, три буквы, две цифры",
        examples=["322BDM02"],
    )
    note: str = Field("", max_length=500, description="Комментарий сотрудника")


class VisitResponse(BaseModel):
    ok: bool = True
    visit_id: str
    plate: str
    note: str
    created_at: str
    database: str
