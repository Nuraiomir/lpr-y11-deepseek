"""
Подтверждение номера без уверенности.

Распознаватель через шлюз уверенности не возвращает. Прежнее правило на ней
стояло целиком: слабое чтение отбрасывалось, уверенное весило больше, одно
очень уверенное подтверждало номер сразу. Если просто подставить вместо
уверенности число, слой устойчивости исчезнет, причём незаметно: при единице
каждое чтение станет «очень уверенным» и номер будет подтверждаться с первого
раза.

Поэтому правил два, и здесь проверяется второе: считаются не веса, а
совпадения. Самое важное, что должны доказать эти тесты, это что одного
чтения недостаточно.

Отдельно проверяется, что первое правило не сдвинулось. Это доказывают 20
тестов в tests/test_recognizer.py, которые не менялись вовсе, а здесь только
сверяются числа обоих правил, чтобы разница была видна в одном месте.
"""
from __future__ import annotations

from app.lpr.recognizer import AGREEMENT, WEIGHTED, LPRRecognizer


def gateway_payload(text):
    """Ответ шлюза: текст есть, ключа conf нет.

    Отсутствие ключа и есть признак распознавателя без уверенности, и именно
    по нему распознаватель отличает «уверенность неизвестна» от «уверенность
    нулевая».
    """
    return {"mode": "normal", "text": text}


def test_the_two_policies_differ_where_it_matters():
    assert WEIGHTED.strong_conf == 0.95
    # Короткого пути «одно уверенное чтение» у второго правила нет: взять
    # уверенность неоткуда.
    assert AGREEMENT.strong_conf is None
    assert AGREEMENT.switch_reads > WEIGHTED.switch_reads
    # Окно шире: один вызов шлюза около 150 мс, чтения идут по кадрам с
    # найденной рамкой, и совпадения набираются за секунды.
    assert AGREEMENT.switch_window_sec > WEIGHTED.switch_window_sec
    # Вес у каждого чтения одинаковый, то есть веса это счётчик совпадений.
    assert AGREEMENT.fixed_weight == 1.0
    assert WEIGHTED.fixed_weight is None
    # Отбрасывать чтения по уверенности, которой нет, значило бы выбросить всё.
    assert AGREEMENT.vote_min_conf < 0


def test_one_reading_does_not_confirm_a_plate():
    """Главный тест файла.

    В прежнем правиле одно чтение с уверенностью 0.96 подтверждало номер
    сразу. Здесь уверенности нет, и такого пути быть не должно, иначе одна
    ошибка распознавания станет ответом.
    """
    rec = LPRRecognizer(policy=AGREEMENT)
    changed, conf, _raw = rec._handle_normal_result(gateway_payload("545BDR05"), 0.0)
    assert changed is False
    assert rec.confirmed_plate == ""
    # Уверенности нет, и в ответ уходит именно это, а не ноль: ноль читался бы
    # как «прочитал, но совсем не уверен».
    assert conf is None


def test_three_matching_readings_confirm():
    rec = LPRRecognizer(policy=AGREEMENT)
    for i in range(AGREEMENT.switch_reads - 1):
        changed, _, _ = rec._handle_normal_result(gateway_payload("545BDR05"), i * 0.3)
        assert changed is False, f"подтвердил с {i + 1} чтения"
    changed, _, _ = rec._handle_normal_result(
        gateway_payload("545BDR05"), AGREEMENT.switch_reads * 0.3)
    assert changed is True
    assert rec.confirmed_plate == "545BDR05"


def test_readings_that_disagree_do_not_confirm_anything():
    """Три чтения, но три разных номера. Подтверждать нечего."""
    rec = LPRRecognizer(policy=AGREEMENT)
    for i, plate in enumerate(("545BDR05", "545BDR06", "545BDB05")):
        rec._handle_normal_result(gateway_payload(plate), i * 0.3)
    assert rec.confirmed_plate == ""


def test_matching_readings_too_far_apart_do_not_confirm():
    """Совпадения считаются в окне. Три чтения за полминуты это не
    устойчивое чтение одной машины, а три разных момента."""
    rec = LPRRecognizer(policy=AGREEMENT)
    step = AGREEMENT.switch_window_sec + 1.0
    for i in range(AGREEMENT.switch_reads):
        rec._handle_normal_result(gateway_payload("545BDR05"), i * step)
    assert rec.confirmed_plate == ""


def test_a_misread_among_matches_does_not_derail_confirmation():
    """Одна ошибка среди совпадений не мешает: совпадений всё равно хватает.

    Это то же свойство, ради которого существует голосование. Без него
    единственная ошибка шлюза ломала бы подтверждение.
    """
    rec = LPRRecognizer(policy=AGREEMENT)
    script = ["545BDR05", "545BDR05", "546BDR05", "545BDR05"]
    for i, plate in enumerate(script):
        rec._handle_normal_result(gateway_payload(plate), i * 0.3)
    assert rec.confirmed_plate == "545BDR05"


def test_a_reading_that_is_not_a_plate_is_ignored():
    """Шлюз может вернуть что угодно, вплоть до рассуждения о картинке."""
    rec = LPRRecognizer(policy=AGREEMENT)
    for i, text in enumerate(("на изображении автомобиль", "", "ABC")):
        changed, _, _ = rec._handle_normal_result(gateway_payload(text), i * 0.3)
        assert changed is False
    assert rec.confirmed_plate == ""


def test_the_confirmed_plate_still_expires(monkeypatch):
    """Снятие номера, который перестали читать, работает в обоих правилах:
    иначе на экране осталась бы уехавшая машина."""
    rec = LPRRecognizer(plate_hold_sec=2.0, policy=AGREEMENT)
    for i in range(AGREEMENT.switch_reads):
        rec._handle_normal_result(gateway_payload("545BDR05"), i * 0.3)
    assert rec.confirmed_plate == "545BDR05"
    assert rec.drop_stale_plate(10.0) is True
    assert rec.confirmed_plate == ""


def test_switching_to_another_vehicle_needs_the_same_agreement():
    """Смена машины по тем же правилам: иначе чужой номер вытеснил бы
    подтверждённый с первого чтения."""
    rec = LPRRecognizer(policy=AGREEMENT)
    for i in range(AGREEMENT.switch_reads):
        rec._handle_normal_result(gateway_payload("545BDR05"), i * 0.3)
    assert rec.confirmed_plate == "545BDR05"

    base = 2.0
    rec._handle_normal_result(gateway_payload("221ZVZ05"), base)
    assert rec.confirmed_plate == "545BDR05", "сменил машину с одного чтения"
    for i in range(1, AGREEMENT.switch_reads):
        rec._handle_normal_result(gateway_payload("221ZVZ05"), base + i * 0.3)
    assert rec.confirmed_plate == "221ZVZ05"


def test_weighted_policy_is_unchanged_by_the_new_parameter():
    """Прежнее правило по умолчанию и работает ровно как раньше.

    Одно уверенное чтение подтверждает номер сразу. Это и есть тот путь,
    которого нет у второго правила, и здесь он проверяется напрямую.
    """
    rec = LPRRecognizer()
    assert rec.policy is WEIGHTED
    assert rec.single_row_only is False
    changed, conf, _ = rec._handle_normal_result(
        {"mode": "normal", "text": "545BDR05", "conf": 0.96}, 0.0)
    assert changed is True
    assert rec.confirmed_plate == "545BDR05"
    assert conf == 0.96


def test_missing_confidence_still_means_zero_for_the_weighted_policy():
    """Старое поведение сохранено: для правила взвешивания отсутствие ключа
    conf это ноль, как было до появления режима без уверенности."""
    rec = LPRRecognizer()
    changed, conf, _ = rec._handle_normal_result(gateway_payload("545BDR05"), 0.0)
    assert conf == 0.0
    assert changed is False
    # Два чтения в окне подтверждают и при нулевой уверенности: так было и
    # раньше, это путь по числу совпадений внутри прежнего правила.
    rec._handle_normal_result(gateway_payload("545BDR05"), 0.3)
    assert rec.confirmed_plate == "545BDR05"


# ---------------------------------------------------------------- через HTTP

def test_the_whole_path_works_with_a_reader_without_confidence(engine, jpeg_frame,
                                                               monkeypatch):
    """Весь путь запроса с распознавателем без уверенности.

    Проверяет то, что отдельными тестами не поймать: что выбор правила
    доезжает от настройки до сессии, что вырез не уходит по квадратному пути,
    и что в ответе уверенность приходит как null, а не как ноль.
    """
    from fastapi.testclient import TestClient

    from app import state
    from app.lpr import ocrm_stub
    from app.lpr.sessions import SessionStore
    from app.main import app

    monkeypatch.delenv("LPR_API_KEYS", raising=False)

    # Рамка квадратная, то есть по обычным правилам вырез пошёл бы по
    # квадратному пути. При single_row_only он должен читаться целиком.
    engine.set_phase("square")
    engine._script = [{"mode": "normal", "text": "545BDR05"}] * 10
    engine._i = 0
    engine.ocr_variant_mode = "gateway"

    state.engine = engine
    state.startup_error = None
    state.sessions = SessionStore(plate_hold_sec=2.0, policy="agreement",
                                  single_row_only=True)
    ocrm_stub.load()
    client = TestClient(app)
    try:
        bodies = []
        for _ in range(AGREEMENT.switch_reads):
            response = client.post("/api/v1/recognize?session_id=gw",
                                   content=jpeg_frame,
                                   headers={"Content-Type": "image/jpeg"})
            assert response.status_code == 200
            bodies.append(response.json())

        # Вырез читался как одна строка, несмотря на квадратную рамку.
        assert all(b["plate_type"] == "normal" for b in bodies), \
            [b["plate_type"] for b in bodies]
        # Уверенности нет, и в ответе это null, а не ноль.
        assert all(b["ocr_confidence"] is None for b in bodies)
        # Номер подтверждён, и не раньше, чем набрались совпадения.
        assert bodies[-1]["plate"] == "545BDR05"
        assert bodies[0]["plate"] == ""
    finally:
        state.engine = None
        state.sessions = None
