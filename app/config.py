"""
Настройки сервиса. Всё, что меняется между машинами, читается отсюда.

Правило одно: в коде нет ни адресов, ни путей, ни ключей. Несекретные
значения приходят из configmaps_envs в deployment/values/<env>/values.yaml,
секретные из Vault через ExternalSecret. Полная таблица переменных в README.

Ключи доступа (LPR_API_KEYS) в файл не пишутся и в репозиторий не попадают.
Пустое значение означает, что проверка выключена: так удобно разрабатывать
локально, и сервис честно говорит об этом при запуске, чтобы это не уехало в
кластер незамеченным.
"""
from __future__ import annotations

import os
from functools import lru_cache


def _bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


class Settings:
    """Прочитано один раз при старте. Менять на ходу нечего."""

    def __init__(self) -> None:
        self.app_name = os.getenv("APP_NAME", "example-service")
        self.version = os.getenv("APP_VERSION", "0.1.0")
        self.environment = os.getenv("ENVIRONMENT", "dev")
        self.log_level = os.getenv("LOG_LEVEL", "INFO").upper()

        # Модель. Путь или каталог: в кластере каталог подкладывает
        # init-контейнер, локально удобнее указать файл.
        self.model_path = os.getenv("MODEL_PATH", "model/best11n_512.onnx")
        # Порог уверенности детектора. Свой у каждой модели, см. комментарий
        # в app/lpr/detector.py. 0 означает «взять значение по умолчанию той
        # модели», то есть не навязывать.
        self.detect_conf = _float("LPR_DETECT_CONF", 0.0)

        # Детектор считается на процессоре, в этом же процессе. Других
        # вариантов в этом репозитории нет: требование банка это работа на
        # CPU, и GPU-сборка здесь только усложняла бы код, который всё равно
        # не поедет в кластер.
        self.device = os.getenv("LPR_DEVICE", "cpu")

        # Символы читает модель в кластере через AI Gateway, и другого
        # распознавателя в этом репозитории нет. Переменная оставлена, чтобы
        # имя режима было видно в журнале и в пробе готовности.
        self.ocr_engine = "gateway"

        # Сколько потоков отдаём onnxruntime. При квоте в одно ядро
        # значение 1: иначе библиотеки запускают столько потоков, сколько
        # ядер видят на узле, а не сколько им разрешено, и время уходит на
        # переключение между ними.
        self.ort_threads = _int("ORT_THREADS", 1)

        # Пределы запроса.
        self.max_frame_bytes = _int("MAX_FRAME_BYTES", 10 * 1024 * 1024)
        self.session_ttl_sec = _int("SESSION_TTL_SEC", 30 * 60)
        self.max_sessions = _int("MAX_SESSIONS", 1000)
        # Подтверждённый номер, который не прочитали заново столько секунд,
        # снимается. Клиент показывает данные залога только пока confirmed.
        self.plate_hold_sec = _float("PLATE_HOLD_SEC", 2.0)

        # Тестовые машины для GET /api/v1/vehicle. Это придуманные данные для
        # демонстрации, а не выгрузка из OCRM.
        self.test_ocrm_file = os.getenv("TEST_OCRM_FILE", "config/test_ocrm.json")

        # Шлюз: адрес, модель, тайм-аут, вход. Ключ только из окружения и
        # только через свойство ниже, чтобы он не оказался в объекте настроек,
        # который может попасть в журнал целиком.
        self.gateway_url = os.getenv("LPR_GATEWAY_URL", "").strip()
        self.gateway_model = os.getenv("LPR_GATEWAY_MODEL",
                                       "deepseek-ocr-2/deepseek-ocr-2")
        self.gateway_timeout = _float("LPR_GATEWAY_TIMEOUT", 10.0)
        # Увеличение выреза перед отправкой на шлюз, по ширине. 0 означает
        # «как есть». Значение должно совпадать с тем, на котором мерили:
        # иначе замер и работа разойдутся.
        self.gateway_upscale = _int("LPR_GATEWAY_UPSCALE", 0)
        self.gateway_prompt = os.getenv("LPR_GATEWAY_PROMPT", "Free OCR.")


        # Страница для съёмки с телефона. В кластере не нужна, включается
        # только там, где её показывают.
        self.serve_demo_page = _bool("SERVE_DEMO_PAGE", False)

    @property
    def api_keys(self) -> set[str]:
        """Ключи доступа из LPR_API_KEYS, через запятую. Пусто = проверки нет."""
        raw = os.getenv("LPR_API_KEYS", "")
        return {k.strip() for k in raw.split(",") if k.strip()}

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_keys)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
