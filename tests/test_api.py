"""
HTTP-слой: разбор запроса, разделение сессий, форма ответа, коды ошибок.

Что здесь НЕ проверяется: арифметика голосования и нормализация номера. Это
проверено отдельно, в tests/test_recognizer.py, на записанных чтениях, и
дублировать её через HTTP значило бы проверять одно и то же дважды, а падать
в двух местах при одной причине.

Что проверяется здесь: что вокруг этой арифметики всё соединено правильно.
Отдельно и подробно проверяется разделение «номер распознан» и «машина
найдена»: это главное требование сценария, и именно его проще всего
нарушить незаметно.
"""
from __future__ import annotations

from app import state

# --------------------------------------------------------------- состояние

def test_root_says_what_the_service_is(client):
    body = client.get("/").json()
    assert body["service"]
    # В ответе прямо сказано, что справочник тестовый. Это защита от того,
    # чтобы прототип приняли за интеграцию с OCRM.
    assert "OCRM" in body["ocrm"]


def test_live_is_ok_and_ready_is_ready_when_models_are_loaded(client):
    assert client.get("/health/live").json()["status"] == "ok"
    ready = client.get("/health/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"


def test_ready_is_503_with_a_reason_while_models_are_not_loaded(client):
    saved = state.engine
    state.engine = None
    state.startup_error = "модель не найдена: /mnt/models/нет"
    try:
        response = client.get("/health/ready")
        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "not_ready"
        # Причина в ответе нужна, чтобы по одному запросу было понятно, чинить
        # сервис или монтирование каталога модели.
        assert "не найдена" in body["reason"]

        # live при упавшей загрузке тоже 503: это единственный случай, когда
        # перезапуск может помочь, и кластер должен его сделать. Пока модели
        # просто грузятся, live отвечает 200, иначе под убьют на середине
        # загрузки и так по кругу.
        assert client.get("/health/live").status_code == 503
    finally:
        state.engine = saved
        state.startup_error = None


def test_frames_are_refused_while_not_ready(client, jpeg_frame):
    saved = state.engine
    state.engine = None
    try:
        response = client.post("/api/v1/recognize", content=jpeg_frame,
                               headers={"Content-Type": "image/jpeg"})
        assert response.status_code == 503
        assert response.json()["error_code"] == "engine_unavailable"
    finally:
        state.engine = saved


# ------------------------------------------------------------ распознавание

def gateway_reads(plate, times=6):
    """Ответы шлюза: текст есть, ключа conf нет. Так отвечает модель без
    уверенности, и именно по отсутствию ключа распознаватель это понимает."""
    return [{"mode": "normal", "text": plate}] * times


def test_matching_reads_confirm_the_plate_over_http(client, engine, jpeg_frame):
    """Подтверждение по совпадениям, через весь HTTP-слой.

    Рамка квадратная, но вырез читается одной строкой: модель отдаёт номер
    целиком, и single_row_only это учитывает.
    """
    engine.set_phase("square")
    engine._script = gateway_reads("545BDR05", 10)
    engine._i = 0

    plates = []
    for _ in range(4):
        response = client.post("/api/v1/recognize?session_id=camA",
                               content=jpeg_frame,
                               headers={"Content-Type": "image/jpeg"})
        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is True
        assert body["session_id"] == "camA"
        assert body["plate_type"] == "normal"
        assert body["frame_size"] == [200, 150]
        # Уверенности нет, и в ответе это null, а не ноль.
        assert body["ocr_confidence"] is None
        plates.append(body["plate"])

    assert plates[0] == "", "подтвердил с первого чтения"
    assert plates[-1] == "545BDR05"


def test_a_second_session_does_not_inherit_the_first_ones_plate(client, engine, jpeg_frame):
    """Разделение сессий. Голосование копит прочитанное, и общее состояние
    означало бы, что номер одной машины попадёт в голоса другой."""
    engine.set_phase("square")
    engine._script = gateway_reads("545BDR05", 20)
    engine._i = 0
    for _ in range(4):
        client.post("/api/v1/recognize?session_id=camA", content=jpeg_frame,
                    headers={"Content-Type": "image/jpeg"})

    body = client.post("/api/v1/recognize?session_id=camB", content=jpeg_frame,
                       headers={"Content-Type": "image/jpeg"}).json()
    assert body["session_id"] == "camB"
    assert body["plate"] != "545BDR05"


def test_frame_as_multipart_upload_works_the_same(client, engine, jpeg_frame):
    engine.set_phase("none")
    response = client.post("/api/v1/recognize?session_id=up",
                           files={"image": ("frame.jpg", jpeg_frame, "image/jpeg")})
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["bbox"] is None
    assert body["frame_size"] == [200, 150]


def test_no_plate_on_the_frame_is_a_normal_answer(client, engine, jpeg_frame):
    engine.set_phase("none")
    body = client.post("/api/v1/recognize", content=jpeg_frame,
                       headers={"Content-Type": "image/jpeg"}).json()
    assert body["ok"] is True
    assert body["plate"] == ""
    assert body["confirmed"] is False
    assert body["confidence"] == 0.0
    # Пустого кадра распознавателю не отдавали: OCR запускается только после
    # рамки. Иначе сервис жёг бы процессор на фоне.
    assert engine.calls["ocr"] == 0


def test_profile_appears_only_when_asked(client, engine, jpeg_frame):
    engine.set_phase("none")
    plain = client.post("/api/v1/recognize", content=jpeg_frame,
                        headers={"Content-Type": "image/jpeg"}).json()
    assert plain["profile"] is None

    detailed = client.post("/api/v1/recognize?profile=1", content=jpeg_frame,
                           headers={"Content-Type": "image/jpeg"}).json()
    assert detailed["profile"] is not None
    assert "yolo_ms" in detailed["profile"]


# -------------------------------------------------------------- ошибки входа

def test_empty_body_is_a_bad_request(client):
    response = client.post("/api/v1/recognize", content=b"",
                           headers={"Content-Type": "image/jpeg"})
    assert response.status_code == 400
    assert response.json()["error_code"] == "bad_request"


def test_body_that_is_not_an_image_is_rejected_as_such(client):
    response = client.post("/api/v1/recognize", content=b"not a picture at all",
                           headers={"Content-Type": "image/jpeg"})
    assert response.status_code == 400
    # Отдельный код, а не общий bad_request: по нему видно, что запрос дошёл,
    # а не разобрался кадр.
    assert response.json()["error_code"] == "decode_failed"


def test_oversized_frame_is_refused_before_it_is_read(client, monkeypatch):
    from app.config import get_settings
    settings = get_settings()
    monkeypatch.setattr(settings, "max_frame_bytes", 100)
    response = client.post("/api/v1/recognize", content=b"x" * 500,
                           headers={"Content-Type": "image/jpeg"})
    assert response.status_code == 413
    assert response.json()["error_code"] == "payload_too_large"


def test_bad_session_id_is_refused(client, jpeg_frame):
    response = client.post("/api/v1/recognize?session_id=" + "a" * 200,
                           content=jpeg_frame,
                           headers={"Content-Type": "image/jpeg"})
    assert response.status_code == 400
    assert response.json()["error_code"] == "bad_request"


# ------------------------------------------------------------------- доступ

def test_key_is_required_once_it_is_configured(client, jpeg_frame, monkeypatch):
    from app.config import get_settings
    monkeypatch.setenv("LPR_API_KEYS", "right-key-0001,second-key-0002")
    get_settings.cache_clear()
    try:
        headers = {"Content-Type": "image/jpeg"}
        assert client.post("/api/v1/recognize", content=jpeg_frame,
                           headers=headers).status_code == 401

        headers["Authorization"] = "Bearer wrong-key"
        assert client.post("/api/v1/recognize", content=jpeg_frame,
                           headers=headers).status_code == 401

        headers["Authorization"] = "Bearer right-key-0001"
        assert client.post("/api/v1/recognize", content=jpeg_frame,
                           headers=headers).status_code == 200

        # Пробы состояния остаются открытыми: их спрашивает кластер, и он
        # ключа не носит.
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 200
    finally:
        monkeypatch.delenv("LPR_API_KEYS", raising=False)
        get_settings.cache_clear()


# ---------------------------------------------- номер распознан и машина в базе

def test_known_plate_is_found(client):
    # Номер из config/test_ocrm.json. Машина придуманная, номер взят с нашей
    # же тестовой съёмки, чтобы сценарий можно было показать на живой машине.
    body = client.get("/api/v1/vehicle/545BDR05").json()
    assert body["ok"] is True
    assert body["found"] is True
    assert body["plate"] == "545BDR05"


def test_unknown_plate_is_200_and_found_false_not_404(client):
    """Главное требование сценария.

    Номера нет в справочнике это нормальный ответ, а не ошибка: номер прочитан
    верно. Если отвечать 404, клиент покажет сбой, и сотрудник решит, что
    распознавание не сработало, хотя сработало.
    """
    # 221ZVZ05 есть на тестовой съёмке, но из справочника убран нарочно:
    # путь «номер прочитан, машины нет» показывается на настоящей машине, а
    # не на выдуманном номере.
    response = client.get("/api/v1/vehicle/221ZVZ05")
    assert response.status_code == 200, "ненайденная машина не должна быть ошибкой"
    body = response.json()
    assert body["ok"] is True
    assert body["found"] is False
    assert body["plate"] == "221ZVZ05"
    assert body["database"] == "test"


def test_a_string_that_is_not_a_plate_is_a_bad_request(client):
    response = client.get("/api/v1/vehicle/ABC123")
    assert response.status_code == 400
    assert response.json()["error_code"] == "bad_request"


def test_visit_is_created(client):
    response = client.post("/api/v1/visit",
                           json={"plate": "545BDR05", "note": "проверка"})
    assert response.status_code == 201
    body = response.json()
    assert body["visit_id"]
    assert body["plate"] == "545BDR05"
    assert body["note"] == "проверка"
    assert body["database"] == "test"


def test_visit_for_a_malformed_plate_is_refused(client):
    response = client.post("/api/v1/visit", json={"plate": "нет", "note": ""})
    assert response.status_code == 400
