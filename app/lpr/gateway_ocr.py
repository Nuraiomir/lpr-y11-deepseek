"""
Чтение символов номера моделью в кластере, через AI Gateway (deepseek-ocr-2).

В этом репозитории это ЕДИНСТВЕННЫЙ распознаватель. Локальных весов у него
нет: сервис обращается к модели по сети, запросом в формате
chat/completions с картинкой внутри.

ЧТО ИЗВЕСТНО ПРО ЕГО КАЧЕСТВО, с измерений на лабораторной машине, сто
вырезов части test из datasets_ocr_kz, bench/ocr_engine_compare.py:

    вариант входа                правильно   чужой номер      мс
    вырез как есть                 44/100             3     150
    формулировка «plate»            6/100             1     149
    вырез увеличен до 1024         46/100             5     353

Три вещи из этих чисел, которые определяют устройство всего остального кода.

ПЕРВОЕ. Формулировку запроса подбирать бессмысленно. Это не модель,
выполняющая инструкции, а модель, которая транскрибирует изображение, и
«Free OCR.» её родной режим. Задача, сформулированная словами, сделала
результат в семь раз хуже.

ВТОРОЕ. Уверенности в ответе НЕТ, и взять её неоткуда: chat/completions её не
содержит. Поэтому подтверждение номера здесь построено на числе совпадений, а
не на взвешивании по уверенности (app/lpr/recognizer.py, правило AGREEMENT).
Подставить вместо уверенности число нельзя: при единице каждое чтение стало
бы «очень уверенным», номер подтверждался бы с первого чтения, и слой
устойчивости исчез бы незаметно.

ТРЕТЬЕ, и самое важное для производительности. Один вызов стоит около 150 мс,
то есть примерно столько же, сколько весь бюджет кадра при трёх кадрах в
секунду. Значит экономить надо на ЧИСЛЕ ОБРАЩЕНИЙ, а не на размере картинки:
увеличение выреза дало плюс два правильных чтения и минус две с половиной
кратную скорость. Отсюда и устройство сервиса: распознаватель вызывается
только после устойчивой рамки, а не на каждом кадре.

Из этих же чисел следует, чего от модели ждать не надо. Чужой номер верного
формата, которых на сотне от одного до пяти, это самая дорогая ошибка в
нашем сценарии: сервис покажет залог другой машины, и сотрудник не увидит
подмены. Правило подтверждения по совпадениям существует именно против этого.

ЧЕГО МЫ С ЭТОЙ МОДЕЛЬЮ СДЕЛАТЬ НЕ МОЖЕМ. Обучить её. У нас есть только адрес
шлюза, весов нет. Всё, что в нашей власти, это что мы ей посылаем (размер и
подготовка выреза), как часто, и как обрабатываем ответ.

КЛЮЧ. Берётся только из переменной окружения LPR_GATEWAY_KEY, то есть в
кластере из Vault. В файлы не пишется, в журнал не печатается, в репозиторий
не попадает. Ни одна строка здесь не выводит его даже частично.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import time
import urllib.error
import urllib.request

import cv2

logger = logging.getLogger("lpr.gateway")

# Запрос, который шлюз понимает. Выяснено зондом: из двух опробованных
# формулировок эта даёт разборчивый ответ, а не рассуждение вокруг картинки.
PROMPT = "Free OCR."


class GatewayUnavailable(RuntimeError):
    """Шлюз не ответил. Наружу это не ошибка сервиса: сравнение не состоялось."""


class GatewayOCR:
    """Клиент шлюза. Один на процесс, состояния не держит."""

    def __init__(self, base_url: str, model: str, timeout: float = 10.0,
                 prompt: str = PROMPT, upscale_to: int = 0):
        if not base_url:
            raise ValueError("не задан адрес шлюза")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.timeout = timeout
        self.prompt = prompt
        # Вырез можно увеличить перед отправкой. Документная модель обучалась
        # на страницах, и плотный вырез номера для неё нетипичен, так что
        # размер входа может влиять на качество чтения. Значение берётся из
        # настройки, чтобы на стенде и в сервисе был один и тот же вход:
        # иначе замер и работа разойдутся, и сравнивать их будет нельзя.
        self.upscale_to = upscale_to

    @property
    def configured(self) -> bool:
        """Есть ли ключ. Без него вызывать нечего."""
        return bool(os.environ.get("LPR_GATEWAY_KEY", "").strip())

    def read(self, crop_bgr) -> tuple[str, float]:
        """(что прочитано, миллисекунды). Бросает GatewayUnavailable.

        Уверенности в ответе нет, и возвращать её неоткуда: см. шапку файла.
        """
        key = os.environ.get("LPR_GATEWAY_KEY", "").strip()
        if not key:
            raise GatewayUnavailable("не задан LPR_GATEWAY_KEY")

        crop = self._prepare(crop_bgr)
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise GatewayUnavailable("вырез не закодировался в JPEG")
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        body = json.dumps({
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt},
                    {"type": "image_url",
                     "image_url": {"url": "data:image/jpeg;base64," + b64}},
                ],
            }],
        }).encode("utf-8")

        request = urllib.request.Request(self.url, data=body, method="POST")
        request.add_header("Authorization", f"Bearer {key}")
        request.add_header("Content-Type", "application/json")

        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            # Тело ответа может содержать что угодно, в том числе эхо запроса,
            # а в запросе ключ. В журнал идёт только код.
            raise GatewayUnavailable(f"шлюз ответил кодом {exc.code}") from None
        except Exception as exc:
            raise GatewayUnavailable(f"{type(exc).__name__}") from None
        ms = (time.perf_counter() - t0) * 1000.0

        try:
            text = json.loads(raw)["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError, TypeError):
            raise GatewayUnavailable("ответ не разобрался") from None

        return str(text), ms

    def _prepare(self, crop_bgr):
        """Вырез в том виде, в каком он уходит на шлюз."""
        if not self.upscale_to:
            return crop_bgr
        h, w = crop_bgr.shape[:2]
        if w >= self.upscale_to or w <= 0:
            return crop_bgr
        scale = self.upscale_to / w
        return cv2.resize(crop_bgr, (self.upscale_to, max(1, int(round(h * scale)))),
                          interpolation=cv2.INTER_CUBIC)

    # ------------------------------------------------------- как распознаватель

    def run(self, crop_bgr, mode):
        """Ответ в том виде, в каком его ждёт app/lpr/recognizer.py.

        Нужно для того, чтобы шлюз мог встать на место основного
        распознавателя: app/lpr/recognizer.py не знает, кто читал номер, он
        получает словарь с текстом.

        Ключа conf в ответе НЕТ, и это не упущение. Шлюз уверенности не
        возвращает, а поставить на её место число значило бы выдумать её:
        распознаватель отличает отсутствие ключа от нулевой уверенности и во
        втором случае считает, что чтение плохое. Отсутствие ключа включает
        подтверждение по числу совпадений (правило AGREEMENT).

        Квадратный номер читается ОДНИМ вызовом, а не двумя. Причина в цене:
        один вызов это около 150 мс, и два вызова на кадр сделали бы его
        неподъёмным. Модель читает две строки сразу, так что делить вырез
        незачем, и номер достаётся из её ответа обычным разбором.
        """
        t0 = time.perf_counter()
        text, ms = self.read(crop_bgr)
        payload = {
            "mode": mode,
            "ms": (time.perf_counter() - t0) * 1000.0,
            "gateway_ms": round(ms, 1),
            "variants": [],
            "variant_count": 1,
            "prep_total_ms": 0.0,
            "infer_total_ms": round(ms, 1),
            "decode_ms": 0.0,
        }
        if mode == "normal":
            payload["text"] = text
        else:
            # Разделить ответ на строки нечем: модель отдаёт текст целиком.
            # Отдаём его как верхнюю строку, а нижнюю пустой, и дальше номер
            # достаётся из текста обычным разбором. Для квадратного номера
            # это значит, что голосование по строкам не работает, и это
            # честное ограничение, а не обход.
            payload["top_text"] = text
            payload["bottom_text"] = ""
        return payload


def build_reader(settings) -> GatewayOCR:
    """Клиент шлюза на роль ОСНОВНОГО распознавателя.

    Бросает, если не настроен: если сервису сказали читать шлюзом, а шлюза
    нет, он не должен молча читать чем-то другим. Пусть лучше не станет
    готовым и скажет причину.
    """
    if not settings.gateway_url:
        raise ValueError("LPR_OCR_ENGINE=gateway, но LPR_GATEWAY_URL не задан")
    client = GatewayOCR(settings.gateway_url, settings.gateway_model,
                        timeout=settings.gateway_timeout,
                        prompt=settings.gateway_prompt,
                        upscale_to=settings.gateway_upscale)
    if not client.configured:
        raise ValueError("LPR_OCR_ENGINE=gateway, но LPR_GATEWAY_KEY не задан "
                         "(в кластере он приходит из Vault)")
    logger.info("символы читает шлюз: модель %s, вход %s",
                settings.gateway_model,
                "как есть" if not settings.gateway_upscale
                else f"увеличен до {settings.gateway_upscale} точек")
    return client
