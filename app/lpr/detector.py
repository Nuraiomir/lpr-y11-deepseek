"""
Поиск номерного знака в кадре. ONNX Runtime, YOLO11n, экспорт в 512x512.

Этот модуль отвечает только за «где на кадре номер», и ничего не знает ни о
чтении символов, ни о голосовании, ни о базе. На выходе одна рамка:
(x1, y1, x2, y2, уверенность) в пикселях исходного кадра, либо None.

Почему одна рамка, а не все найденные. Сценарий такой: сотрудник наводит
камеру на номер конкретной машины. Ближайший номер он же и самый уверенный,
а вторая рамка породила бы вопрос, к какой из машин относится ответ. Выбор
«самая уверенная» разрешает этот вопрос один раз и в одном месте.

Считается в том же процессе, что и сервис, на процессоре. Это самая
дешёвая часть конвейера: рамка находится за двадцать миллисекунд локально,
тогда как чтение символов это запрос по сети на сотни миллисекунд. Поэтому
бюджет кадра тратится не здесь, и оптимизировать детектор смысла мало.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

# Размер входа модели. Пришит к экспорту: модель принимает 512x512, и
# другое число здесь молча испортит координаты, а не вызовет ошибку.
INPUT_SIZE = 512

# Ниже этого порога рамка не рассматривается.
#
# 0.40 измерено для НАШЕЙ модели на отложенной выборке из 79 фотографий
# (66 с номером, 13 без). На этом пороге модель попала в номер на 64 из 66
# картинок, ни разу не выдала рамку мимо номера и ни разу не сработала там,
# где номера нет. На 0.25 результат тот же, на 0.55 попаданий 62, на 0.70
# уже 58, то есть 0.40 это верхняя граница, на которой ещё ничего не
# теряется, и запас против кадров похуже, которых в этой выборке нет.
#
# Порог вынесен в настройку, потому что он СВОЙ У КАЖДОЙ ОБУЧЕННОЙ ВЕРСИИ.
# Переобучили модель, значит порог надо измерить заново тем же стендом, а не
# переносить прежний: с новыми весами меняется и распределение уверенности.
# Стенд лежит в исследовательском репозитории, bench/detector_conf_sweep.py.
DEFAULT_MIN_CONF = 0.40
MIN_CONF = float(os.environ.get("LPR_DETECT_CONF", DEFAULT_MIN_CONF))

# Цвет заливки при вписывании кадра в квадрат. Серый 114 это то, с чем модель
# обучалась (значение по умолчанию у Ultralytics), и менять его нельзя:
# модель видела именно такие поля.
PAD_VALUE = 114


def letterbox(image, size: int = INPUT_SIZE):
    """Вписывает кадр в квадрат модели, не искажая пропорции.

    Кадр масштабируется целиком и кладётся в центр серого квадрата. Номер
    остаётся номером, а не растягивается в зависимости от того, какой формат
    прислала камера. Возвращает квадрат и три числа, которыми потом рамку с
    квадрата переводят обратно в координаты исходного кадра.
    """
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), PAD_VALUE, np.uint8)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized
    return canvas, scale, dx, dy


class PlateDetector:
    """Загруженная модель детектора. Создаётся один раз на процесс."""

    def __init__(self, model_path, providers=None, min_conf=None):
        import onnxruntime as ort

        # Порог на экземпляре, а не только в модуле: так два детектора можно
        # держать рядом с разными порогами, что и нужно для сравнения.
        self.min_conf = MIN_CONF if min_conf is None else float(min_conf)
        self.model_path = str(model_path)
        # Порядок важен: ONNX Runtime берёт первый провайдер, который смог
        # инициализироваться. На машине без CUDA первый отпадает сам, и
        # остаётся CPU, поэтому один и тот же список годится везде.
        self.session = ort.InferenceSession(
            self.model_path,
            providers=providers or ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    @property
    def providers(self):
        return self.session.get_providers()

    @property
    def on_gpu(self) -> bool:
        return "CUDAExecutionProvider" in self.providers

    def detect(self, image):
        """Самая уверенная рамка номера, либо None.

        Координаты в пикселях исходного кадра. Разные экспорты раскладывают
        выход по-разному, поэтому здесь приняты оба порядка осей и обе шкалы
        координат: и пиксели входа модели, и доли от него.
        """
        canvas, scale, dx, dy = letterbox(image)
        # BGR -> RGB, 0..255 -> 0..1, HWC -> NCHW. Так модель обучалась.
        arr = canvas[:, :, ::-1].astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))[None]
        pred = self.session.run(None, {self.input_name: arr})[0]

        if pred.ndim == 3:
            pred = pred[0]
        # Выход бывает и (признаки, рамки), и (рамки, признаки). Признаков у
        # нашей модели пять, рамок тысячи, так что по форме видно, что к чему.
        if pred.ndim == 2 and pred.shape[0] < pred.shape[1] and pred.shape[0] <= 10:
            pred = pred.T

        best = None
        fh, fw = image.shape[:2]
        for row in pred:
            if len(row) < 5:
                continue
            cx, cy, bw, bh = map(float, row[:4])
            conf = float(row[4]) if len(row) == 5 else float(np.max(row[4:]))
            if conf < self.min_conf:
                continue
            # Если все четыре числа не больше двух, это доли, а не пиксели.
            if max(abs(cx), abs(cy), abs(bw), abs(bh)) <= 2:
                cx *= INPUT_SIZE
                cy *= INPUT_SIZE
                bw *= INPUT_SIZE
                bh *= INPUT_SIZE
            x1 = max(0, min(fw - 1, int((cx - bw / 2 - dx) / scale)))
            y1 = max(0, min(fh - 1, int((cy - bh / 2 - dy) / scale)))
            x2 = max(1, min(fw, int((cx + bw / 2 - dx) / scale)))
            y2 = max(1, min(fh, int((cy + bh / 2 - dy) / scale)))
            if x2 <= x1 or y2 <= y1:
                continue
            if best is None or conf > best[4]:
                best = (x1, y1, x2, y2, conf)
        return best
