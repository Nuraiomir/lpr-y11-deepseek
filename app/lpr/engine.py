"""
Модели в одном процессе. Интерфейс тот же, что у двухпроцессной сборки.

У распознающего слоя (app/lpr/recognizer.py) ровно два требования к тому, кто
считает модели:

    detect(кадр)         -> (x1, y1, x2, y2, уверенность) или None
    ocr(вырез, режим)    -> словарь с прочитанным текстом

Как именно это посчитано, он не знает и знать не должен. Существуют две
сборки:

Сборка одна: детектор считается в этом же процессе на процессоре, символы
читает модель в кластере по сети. Требование банка это работа на CPU, и
GPU-вариант здесь не предусмотрен: он усложнял бы код, который всё равно не
поедет в кластер, где GPU-квоты нет.

Из этого следует неочевидное: детектор стоит дешевле распознавателя. Рамка
на кадре считается за десятки миллисекунд локально, а чтение символов это
запрос по сети на сотни миллисекунд. Поэтому бюджет кадра тратится не там,
где обычно ожидают, и экономить надо на числе ОБРАЩЕНИЙ к распознавателю, а
не на размере модели детектора.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path


class EngineError(RuntimeError):
    """Модель не смогла посчитать кадр. Наружу превращается в 503."""


class EngineStartupError(RuntimeError):
    """Модель не загрузилась. Сервис остаётся не готов (/health/ready -> 503)."""


def resolve_model_path(path) -> Path:
    """Путь к .onnx из того, что дали: файл или папка с моделью.

    MODEL_PATH в кластере показывает на каталог, который подкладывает
    init-контейнер, а локально удобнее указать файл. Принимается и то и то,
    как в sam3-distill, чтобы одна и та же переменная работала везде.
    """
    path = Path(path).expanduser()
    if path.is_file():
        return path
    if path.is_dir():
        preferred = path / "best11n_512.onnx"
        if preferred.is_file():
            return preferred
        found = sorted(path.glob("*.onnx"))
        if found:
            return found[0]
        raise EngineStartupError(f"в каталоге {path} нет файла .onnx")
    raise EngineStartupError(f"модель не найдена: {path}")


class InProcessEngine:
    """Детектор и распознаватель в этом же процессе.

    Обе модели грузятся в __init__, то есть один раз за время жизни процесса.
    Пока он не прошёл, сервис отвечает на /health/ready кодом 503, и кластер
    не пускает на него трафик.
    """

    def __init__(self, model_path, reader, device="cpu", detect_conf=None):
        """reader: кто читает символы. Обязателен.

        Любой объект с методом run(вырез, режим). В этом репозитории это
        клиент AI Gateway (app/lpr/gateway_ocr.py). Параметр не имеет
        значения по умолчанию нарочно: если распознавателя нет, сервис не
        должен подниматься и делать вид, что работает.
        """
        from app.lpr.detector import PlateDetector

        if reader is None:
            raise EngineStartupError("распознаватель не задан")

        self.ocr_variant_mode = "gateway"
        resolved = resolve_model_path(model_path)
        try:
            providers = (["CPUExecutionProvider"] if device == "cpu"
                         else ["CUDAExecutionProvider", "CPUExecutionProvider"])
            self._detector = PlateDetector(resolved, providers=providers,
                                           min_conf=detect_conf or None)
            self._ocr = reader
            # Уверенности этот распознаватель не возвращает, и об этом должно
            # быть видно снаружи: от этого зависит правило подтверждения.
            ocr_info = {"device": "gateway",
                        "backend": getattr(reader, "model", "внешний"),
                        "confidence": False}
        except Exception as exc:
            raise EngineStartupError(repr(exc)) from exc

        self.detector_path = str(resolved)
        self.yolo_info = {"providers": self._detector.providers,
                          "model": self.detector_path,
                          # В журнал и в пробу готовности: по какому порогу
                          # эта модель работает. Без этого при сравнении двух
                          # детекторов невозможно сказать, в каких условиях
                          # получены числа.
                          "min_conf": self._detector.min_conf}
        self.ocr_info = ocr_info
        # Детектор тоже один на процесс. onnxruntime умеет считать из разных
        # потоков, но тогда несколько запросов делят ядра и каждый отвечает
        # медленнее. В кластере квота одно ядро, так что очередь честнее.
        self._detect_lock = threading.Lock()
        self._timing = threading.local()

    @property
    def last_timing(self):
        return getattr(self._timing, "value", None)

    @property
    def restarts(self):
        # Перезапускать нечего: процесс один. Поле оставлено, потому что его
        # печатает тот же код состояния, что и для двухпроцессной сборки.
        return {"yolo": 0, "ocr": 0}

    def detect(self, frame_bgr):
        t0 = time.perf_counter()
        try:
            with self._detect_lock:
                lock_taken = time.perf_counter()
                det = self._detector.detect(frame_bgr)
        except Exception as exc:
            raise EngineError(f"detect: {exc!r}") from exc
        done = time.perf_counter()
        self._timing.value = {
            # Пересылки кадра нет, поэтому нет и кодирования в JPEG: в
            # двухпроцессной сборке это заметная часть времени.
            "jpeg_encode_ms": 0.0,
            "lock_wait_ms": (lock_taken - t0) * 1000.0,
            "ipc_roundtrip_ms": 0.0,
            "worker_inference_ms": (done - lock_taken) * 1000.0,
            "jpeg_bytes": 0,
        }
        return det

    def ocr(self, crop_bgr, mode):
        t0 = time.perf_counter()
        try:
            payload = self._ocr.run(crop_bgr, mode)
        except Exception as exc:
            raise EngineError(f"ocr: {exc!r}") from exc
        self._timing.value = {
            "mode": mode,
            "jpeg_encode_ms": 0.0,
            "lock_wait_ms": 0.0,
            "ipc_roundtrip_ms": 0.0,
            "worker_inference_ms": float(payload.get("ms", 0.0)),
            "jpeg_bytes": 0,
            "crop_size": [int(crop_bgr.shape[1]), int(crop_bgr.shape[0])],
            "worker_decode_ms": payload.get("decode_ms"),
            "worker_prep_ms": payload.get("prep_total_ms"),
            "worker_infer_ms": payload.get("infer_total_ms"),
            "variant_count": payload.get("variant_count"),
            "variants": payload.get("variants", []),
        }
        # Общее время считаем здесь, а не внутри: сюда входит и ожидание
        # блокировки, то есть то, что видит клиент.
        self._timing.value["total_ms"] = (time.perf_counter() - t0) * 1000.0
        return payload

    def close(self):
        # Ничего не закрываем: сессия ONNX Runtime освобождается вместе с
        # процессом, а у распознавателя по сети закрывать нечего. Метод есть,
        # чтобы вызывающему коду не приходилось проверять, умеет ли движок
        # закрываться.
        pass
