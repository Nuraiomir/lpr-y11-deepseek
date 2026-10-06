"""
Общая обвязка тестов.

Главное здесь это подставной движок. Настоящие модели в тестах не грузятся по
двум причинам: их нет в сборочном окружении, и они делают проверку HTTP-слоя
зависящей от того, что ответит шлюз на чёрном кадре, и от того, доступен
ли он вообще.

Подставной движок отдаёт записанные чтения в заданном порядке. Записаны они с
настоящих прогонов по видео (те же последовательности, что в
tests/test_recognizer.py), поэтому проверяется не выдуманное поведение, а то
же, что происходило на реальных кадрах. Под проверкой при этом HTTP-слой:
разбор параметров, разделение сессий, форма ответа, коды ошибок.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

# Записанное чтение квадратного номера, видео 2. Девять кадров, из которых
# голосование собирает 979CBB02, и видно, что отдельные чтения ошибаются
# («Kz 919», «Kz 9/9»), а результат всё равно верный.
SQUARE_SEQUENCE = [
    ("KZ979", 0.862, "UZCBB", 0.829),
    ("Kz 979", 0.876, "UZCBB", 0.906),
    ("Kz 979", 0.866, "02CBB", 0.988),
    ("Kz 919", 0.802, "UZCBB", 0.86),
    ("Kz 979", 0.747, "UZCBB", 0.896),
    ("Kz 979", 0.871, "UZCBB", 0.875),
    ("Kz 9/9", 0.698, "UZCBB", 0.879),
    ("kz 979", 0.828, "02CBB", 0.98),
    ("5", 0.773, "9U", 0.19),
]
NORMAL_FOLLOWUP = [("221ZVZ05", 0.867), ("221ZVZ05", 0.777)]


class FakeEngine:
    """Подставной движок. Интерфейс тот же, что у настоящих сборок.

    Форму рамки задаёт set_phase, а не содержимое кадра: от формы рамки
    зависит, пойдёт вырез по однострочному пути или по квадратному, и тест
    должен управлять этим сам, а не надеяться на то, что детектор увидит на
    чёрном прямоугольнике.
    """

    def __init__(self, script=None):
        self._script = list(script or [])
        self._i = 0
        self._phase = "square"
        self.ocr_variant_mode = "gateway"
        self.yolo_info = {"providers": ["FAKE"]}
        self.ocr_info = {"backend": "FAKE"}
        self.calls = {"detect": 0, "ocr": 0}

    def set_phase(self, phase):
        self._phase = phase

    @property
    def last_timing(self):
        return None

    @property
    def restarts(self):
        return {"yolo": 0, "ocr": 0}

    def detect(self, frame_bgr):
        self.calls["detect"] += 1
        if self._phase == "none":
            return None
        if self._phase == "square":
            return (0, 0, 200, 150, 0.90)   # отношение 1.33 -> квадратный путь
        return (0, 0, 300, 60, 0.90)        # отношение 5.0  -> однострочный

    def ocr(self, crop_bgr, mode):
        self.calls["ocr"] += 1
        if self._i >= len(self._script):
            if mode == "normal":
                return {"mode": "normal", "text": "", "conf": 0.0}
            return {"mode": "square", "top_text": "", "top_conf": 0.0,
                    "bottom_text": "", "bottom_conf": 0.0}
        payload = self._script[self._i]
        self._i += 1
        return payload

    def close(self):
        pass


def build_script():
    script = []
    for top, tc, bottom, bc in SQUARE_SEQUENCE:
        script.append({"mode": "square", "top_text": top, "top_conf": tc,
                       "bottom_text": bottom, "bottom_conf": bc})
    for plate, conf in NORMAL_FOLLOWUP:
        script.append({"mode": "normal", "text": plate, "conf": conf})
    return script


@pytest.fixture
def jpeg_frame():
    """Кадр, который разбирается как картинка. Содержимое не важно: что
    увидит движок, задаёт FakeEngine, а не пиксели."""
    ok, buf = cv2.imencode(".jpg", np.zeros((150, 200, 3), np.uint8))
    assert ok
    return buf.tobytes()


@pytest.fixture
def engine():
    return FakeEngine(build_script())


@pytest.fixture
def client(engine, monkeypatch):
    """Приложение с подставным движком, без загрузки настоящих моделей.

    TestClient создаётся без контекстного менеджера нарочно: тогда lifespan не
    запускается и приложение не пытается загрузить модели. Состояние, которое
    в обычной жизни заполняет lifespan, заполняется здесь вручную.
    """
    from app import state
    from app.lpr import ocrm_stub
    from app.lpr.sessions import SessionStore
    from app.main import app

    monkeypatch.delenv("LPR_API_KEYS", raising=False)
    state.engine = engine
    state.startup_error = None
    state.sessions = SessionStore(plate_hold_sec=2.0, policy="agreement",
                                  single_row_only=True)
    ocrm_stub.load()

    yield TestClient(app)

    state.engine = None
    state.sessions = None
