# Образ сервиса распознавания номеров. Один процесс, CPU.
#
# Собирается в два этапа. На первом ставятся зависимости, на втором остаётся
# только готовое окружение и код: инструменты сборки, кэш пакетов и заголовки
# в итоговый образ не попадают.
#
# Почему CPU. Требование банка это работа на процессоре. GPU-сборка
# onnxruntime притащила бы несколько гигабайт библиотек CUDA, которым в
# кластере нечего делать, а GPU-квоты там и нет.

FROM python:3.12-slim AS builder

# UV ставит зависимости в разы быстрее pip.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Сертификаты банка. Без них uv не достанет пакеты из внутреннего зеркала.
COPY certs/ca-bundle.crt /usr/local/share/ca-certificates/bcc-ca-bundle.crt
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    UV_NATIVE_TLS=1

WORKDIR /build

# Зависимости отдельным слоем и раньше кода: они меняются редко, а весят
# заметно, и правка в app/ не должна тянуть переустановку onnxruntime.
COPY pyproject.toml ./
RUN uv venv /opt/venv \
    && VIRTUAL_ENV=/opt/venv uv pip install --no-cache-dir -r pyproject.toml

# ---------------------------------------------------------------------------

FROM python:3.12-slim

# libgomp1: OpenMP, нужен ONNX Runtime.
# libglib2.0-0 и libgl1: OpenCV ссылается на них даже в headless-сборке
#   (та убирает окна, но не библиотеки обработки изображений под ними).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libgl1 \
        libglib2.0-0 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY certs/ca-bundle.crt /usr/local/share/ca-certificates/bcc-ca-bundle.crt
RUN update-ca-certificates

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

WORKDIR /app

# Непривилегированный пользователь. Домашний каталог настоящий, а не
# /nonexistent: библиотеки иногда пишут в него временные файлы.
RUN useradd --create-home --uid 10001 app
ENV HOME=/home/app

COPY --chown=app:app app/ app/
COPY --chown=app:app config/test_ocrm.json config/test_ocrm.json
COPY --chown=app:app web/ web/
COPY --chown=app:app entrypoints/ entrypoints/

# Модель детектора кладётся в образ. Она весит 12 МБ, а init-контейнер и
# запись в реестре моделей это отдельная инфраструктура, которой пока нет:
# положить модель в образ значит, что первый деплой в TEST зеленеет сам.
#
# Путь при этом настраиваемый: MODEL_PATH может показать на смонтированный
# каталог, и тогда образ возьмёт модель оттуда. Когда модель переедет в
# реестр, менять нужно будет одно значение в values.yaml, а не образ.
#
# Весов распознавателя в образе НЕТ и быть не может: символы читает модель в
# кластере, сервис обращается к ней по сети. Отсюда и размер образа: без
# локального распознавателя в нём нет примерно гигабайта зависимостей.
COPY --chown=app:app model/ model/

USER app

# По умолчанию модель лежит в образе. В кластере значение переопределяется
# через configmaps_envs, если модель монтируется.
ENV MODEL_PATH=/app/model/best11n_512.onnx \
    LPR_DEVICE=cpu \
    ORT_THREADS=1 \
    PORT=8080

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["python", "entrypoints/healthcheck.py"]

ENTRYPOINT ["/app/entrypoints/entrypoint.sh"]
