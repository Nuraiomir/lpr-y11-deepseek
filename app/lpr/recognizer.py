"""
License plate recognition core for Kazakhstan plates.

Everything that happens after YOLO has found a plate lives here: choosing
between the single-row and two-row (square) OCR path, normalizing OCR text to
the Kazakhstan format, temporal voting across frames, and deciding when the
camera has moved on to a different vehicle.

Both entry points use this module, so they run the same rules:
  app/api/recognition.py    the HTTP service, one LPRRecognizer per session
  the offline video evaluation on the laboratory repository, which uses
  consider_plate (that pipeline is not part of the service image)

This module is deliberately independent of where the models are computed: it
is handed an object with detect() and ocr(), and both the single-process CPU
build (app/lpr/engine.py) and the two-process GPU build
(app/lpr/workers_client.py) satisfy it. Same rules on both.

Plate format: three digits, three Latin letters, two digits (e.g. 545BDR05).
A square plate carries the digits on the top row and region + letters on the
bottom row (e.g. top "633", bottom "02BBT" -> 633BBT02).

The HTTP path (process_frame) drops a confirmed plate that has not been read
for PLATE_HOLD_SEC seconds. The offline pipeline only uses consider_plate and
is not affected.

Note on OCR_EVERY_N_DETECTIONS: only every third square-shaped detection is
sent to OCR. The value was tuned for a 60 fps offline loop. A live camera that
sends far fewer frames per second may need a smaller value.
"""

import os
import re
import time
from collections import defaultdict
from contextlib import contextmanager

# Detection and plate shape
YOLO_CONF = 0.40

# Wider than this -> single-row plate, narrower -> the two-row path that cuts
# the crop in half. The value is tied to ONE detector's idea of where a plate
# ends: measured on four videos, three different detectors all sent 34-37% of
# their boxes down the square path, and a fourth, retrained on the same data
# with 150 frames removed, sent 65% and found 10 vehicles of 28 instead of 25.
# Its own box metrics were perfect (recall 1.000): it finds plates fine, it
# just draws them taller, and this threshold then routes single-row plates
# into the two-row path, where they are cut through the characters.
#
# So this constant is not a property of Kazakh plates, it is a property of the
# detector in model/. Any replacement detector has to be checked against it,
# and LPR_SQUARE_ASPECT_MAX exists to find the right value without editing
# code. A pipeline that routes on box shape has this coupling by construction;
# removing it means routing on something the detector cannot shift.
SQUARE_ASPECT_MAX = float(os.environ.get("LPR_SQUARE_ASPECT_MAX", "1.80"))
MIN_SQUARE_W = 130            # smaller crops are treated as single-row
MIN_SQUARE_H = 85
OCR_EVERY_N_DETECTIONS = 3    # square detections: OCR every Nth one

# Temporal voting for square plates
WINDOW_SEC = 3.0
MIN_TOP_WEIGHT = 1.60
MIN_BOTTOM_WEIGHT = 2.00
# Used only by the end-of-run report. The online decision needs both row
# weights above their minimums, whose sum (3.60) already exceeds this value.
MIN_FINAL_WEIGHT = 2.50

# Vehicle switching
SWITCH_WINDOW_SEC = 1.5
SWITCH_CONFIRM_READS = 2      # matching reads needed to switch plates
SWITCH_STRONG_CONF = 0.95     # ...or one read at least this confident


# ---------------------------------------------------------------------------
# Правила подтверждения
# ---------------------------------------------------------------------------
# Все числа выше описывают ОДИН способ подтверждать номер: взвешивать чтения
# уверенностью распознавателя. Этот способ требует, чтобы распознаватель
# уверенность возвращал. Распознаватель через AI Gateway её не возвращает:
# chat/completions не содержит уверенности в ответе. Подставить вместо неё
# единицу нельзя, это сделало бы каждое чтение «очень уверенным» и номер
# подтверждался бы с первого чтения, то есть слой устойчивости просто исчез
# бы, причём незаметно.
#
# Поэтому правил подтверждения два, и выбираются они при создании
# распознавателя. Числа вынесены сюда, в один объект, а не разбросаны по
# функциям, чтобы было видно, что именно отличается.
#
#   AGREEMENT    то, что работает в этом сервисе. Вес у каждого чтения
#                одинаковый, то есть считаются не веса, а СОВПАДЕНИЯ: номер
#                подтверждается, когда одно и то же прочитано нужное число
#                раз. Короткого пути «одно уверенное чтение» здесь нет, его
#                неоткуда взять.
#
#   WEIGHTED     взвешивание по уверенности. В этом сервисе НЕ используется,
#                потому что уверенности нет. Оставлено по двум причинам: это
#                те же самые функции голосования, только с другими числами,
#                так что удаление ничего не упростило бы; и на нём держатся
#                двадцать регрессионных тестов нормализации и голосования,
#                которые проверяют логику, общую для обоих правил.
#
# Числа для AGREEMENT выбраны из простого расчёта, и их надо проверить
# замером, а не принимать на слово. Три совпадения строгее двух, потому что
# проверять уверенность каждого чтения нечем. Окно шире, потому что один
# вызов шлюза это около 150 мс, чтения идут только по кадрам с найденной
# рамкой, и три совпадения набираются за секунды, а не за полторы.


class ConfirmPolicy:
    """Числа, по которым номер считается подтверждённым.

    vote_min_conf   чтение слабее этого в голосование не попадает. При -1
                    не отбрасывается ничего: отбрасывать по уверенности,
                    которой нет, значило бы выбросить все чтения.
    fixed_weight    если задан, вес чтения всегда такой, независимо от
                    уверенности. Так веса превращаются в счётчик совпадений.
    strong_conf     одно чтение такой уверенности подтверждает номер сразу.
                    None отключает этот путь.
    """

    __slots__ = ("name", "vote_min_conf", "fixed_weight", "strong_conf",
                 "top_min", "bottom_min", "window_sec",
                 "switch_reads", "switch_window_sec")

    def __init__(self, name, vote_min_conf, fixed_weight, strong_conf,
                 top_min, bottom_min, window_sec, switch_reads,
                 switch_window_sec):
        self.name = name
        self.vote_min_conf = vote_min_conf
        self.fixed_weight = fixed_weight
        self.strong_conf = strong_conf
        self.top_min = top_min
        self.bottom_min = bottom_min
        self.window_sec = window_sec
        self.switch_reads = switch_reads
        self.switch_window_sec = switch_window_sec

    def weight_of(self, conf):
        """Вес одного чтения."""
        if self.fixed_weight is not None:
            return self.fixed_weight
        weight = float(conf)
        if conf >= 0.90:
            weight += 0.40
        return weight

    def __repr__(self):
        return f"ConfirmPolicy({self.name})"


WEIGHTED = ConfirmPolicy(
    name="weighted",
    vote_min_conf=0.50,
    fixed_weight=None,
    strong_conf=SWITCH_STRONG_CONF,
    top_min=MIN_TOP_WEIGHT,
    bottom_min=MIN_BOTTOM_WEIGHT,
    window_sec=WINDOW_SEC,
    switch_reads=SWITCH_CONFIRM_READS,
    switch_window_sec=SWITCH_WINDOW_SEC,
)

AGREEMENT = ConfirmPolicy(
    name="agreement",
    vote_min_conf=-1.0,
    fixed_weight=1.0,
    strong_conf=None,
    top_min=3.0,
    bottom_min=3.0,
    window_sec=8.0,
    switch_reads=3,
    switch_window_sec=4.0,
)

POLICIES = {p.name: p for p in (WEIGHTED, AGREEMENT)}

# A confirmed plate that has not been read again for this many seconds is
# dropped, so the previous car is not shown after the camera has moved away.
# On the reference videos the longest gap between reads of a plate that was
# still in view was 0.7 s. Only reads of the confirmed plate itself keep it
# alive; a detection of some other plate does not. 0 disables the timeout.
PLATE_HOLD_SEC = 2.0

# A detection this small or this oddly shaped cannot be a plate. On real
# parking footage the detector also fires on badges, lights and lettering, and
# such a crop costs an OCR call without ever producing a plate.
#
# The thresholds are fractions of the frame's SHORTER side. A phone held
# upright has a narrower horizontal view, so the same plate covers a larger
# share of the width than in a landscape frame; the shorter side is the same
# in both and keeps one rule valid for either orientation, and for a
# phone-sized frame as well.
#
# Measured over four videos (244 detections that never produced a plate): the
# smallest crop that did produce a plate was about 17% of the shorter side,
# so 7% keeps a wide margin.
PLATE_MIN_WIDTH_FRAC = 0.07
PLATE_MIN_HEIGHT_FRAC = 0.025
PLATE_ASPECT_MIN = 0.55
PLATE_ASPECT_MAX = 6.5

# Memory bounds for long-running sessions. Voting only looks back WINDOW_SEC,
# so older votes can never influence a decision. They are kept for ten
# windows before being dropped, which leaves room for small clock steps
# backwards without changing any result.
VOTE_RETENTION_SEC = WINDOW_SEC * 10
MAX_READINGS = 500            # diagnostic history kept per session


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def looks_like_plate(width, height, frame_width, frame_height):
    """Can a crop of this size and shape be a plate at all?"""
    if width <= 0 or height <= 0:
        return False
    short_side = min(frame_width, frame_height)
    if width < PLATE_MIN_WIDTH_FRAC * short_side:
        return False
    if height < PLATE_MIN_HEIGHT_FRAC * short_side:
        return False
    return PLATE_ASPECT_MIN <= width / height <= PLATE_ASPECT_MAX


# A Kazakh plate carries a printed "KZ" block, and OCR reads it as part of the
# text: a plate 545BDR05 comes back as "KZ545BDR05". Matching the whole string
# against the format threw every such reading away. Measured on the val split
# (1001 labelled crops, AUTO.RIA KZ, 823 of them plates of the current format):
# pulling the plate out of the string took correct readings from 294 to 785.
#
# The stray-digit rule in extract_plate was added later and gave 92 of those
# back: the same measurement now reads 693, or 84.2% of the 823. That is the
# price of never building another car's plate out of a misread digit, and it
# was paid on purpose. Any quoted accuracy has to say which of the two it is.
#
# The lookahead finds overlapping matches, and a plate is taken only when the
# string holds exactly one candidate: two candidates mean the crop caught two
# plates, and guessing between them could send OCRM to the wrong car.
PLATE_INSIDE = re.compile(r"(?=(\d{3}[A-Z]{3}\d{2}))")


def extract_plate(text):
    """The plate inside a reading, or "" if there is none or more than one.

    For a string that is already exactly a plate this returns that same plate,
    so no reading that was accepted before can change its value.

    Nothing outside the plate may be a digit. The printed KZ block is letters,
    so "KZ545BDR05" still yields its plate, while a stray digit means OCR
    returned more digits than a plate has: on real footage "822AKH02" came back
    as "822AKH102", and the window "822AKH10" is a perfectly shaped plate of a
    different car. That reading was confirmed before this check existed.
    """
    found = set()
    for m in PLATE_INSIDE.finditer(text):
        plate = m.group(1)
        left = text[:m.start()]
        right = text[m.start() + len(plate):]
        if (left and left[-1].isdigit()) or (right and right[0].isdigit()):
            continue
        found.add(plate)
    return found.pop() if len(found) == 1 else ""


def valid_kz_plate(text):
    text = re.sub(r"[^A-Z0-9]", "", (text or "").upper())
    return bool(re.fullmatch(r"\d{3}[A-Z]{3}\d{2}", text))


def clean_text(s):
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def normalize_top(s):
    """
    Square top row should be exactly 3 digits.
    We deliberately do NOT aggressively convert arbitrary letters to digits.
    This avoids turning a stray word into a false numeric candidate.
    """
    s = clean_text(s)

    # More than three digits is ambiguous: "KZ 049" read with the Z as a 2
    # gives "2049", and the extra digit can sit on either side ("16330" for
    # 633). Taking the first three produced wrong plates such as 204BXS02,
    # so such a read does not vote at all.
    if len(re.sub(r"\D", "", s)) > 3:
        return ""

    m = re.search(r"(\d{3})", s)
    if m:
        return m.group(1)

    if len(s) == 3:
        trans = str.maketrans({
            "O": "0", "Q": "0", "D": "0", "I": "1", "L": "1",
            "Z": "2", "E": "3", "A": "4", "S": "5", "G": "6",
            "T": "7", "B": "8", "P": "9",
        })
        mapped = s.translate(trans)
        if mapped.isdigit() and len(mapped) == 3:
            return mapped

    return ""


def normalize_bottom(s):
    """Returns bottom row as RRLLL, e.g. 02BBT."""
    s = clean_text(s).replace("KZ", "")

    m = re.fullmatch(r"(\d{2})([A-Z]{3})", s)
    if m:
        return m.group(1) + m.group(2)

    m = re.fullmatch(r"([A-Z]{3})(\d{2})", s)
    if m:
        return m.group(2) + m.group(1)

    m = re.search(r"(\d{2})([A-Z]{3})", s)
    if m:
        return m.group(1) + m.group(2)

    m = re.search(r"([A-Z]{3})(\d{2})", s)
    if m:
        return m.group(2) + m.group(1)

    return ""


# ---------------------------------------------------------------------------
# Temporal voting
# ---------------------------------------------------------------------------

def add_vote(votes, value, conf, t, policy=WEIGHTED):
    # conf может быть None: распознаватель без уверенности её не возвращает,
    # и подставлять вместо неё число значило бы выдумать уверенность.
    if conf is None:
        conf = 0.0
    if not value or conf < policy.vote_min_conf:
        return
    votes.append({"time": float(t), "value": value,
                  "weight": policy.weight_of(conf), "conf": float(conf)})


def aggregate(votes, now, window_sec=WINDOW_SEC):
    agg = defaultdict(lambda: {"weight": 0.0, "count": 0, "best_conf": 0.0})
    for v in votes:
        if now - v["time"] <= window_sec:
            a = agg[v["value"]]
            a["weight"] += v["weight"]
            a["count"] += 1
            a["best_conf"] = max(a["best_conf"], v["conf"])
    result = [(value, a["weight"], a["count"], a["best_conf"]) for value, a in agg.items()]
    result.sort(key=lambda x: x[1], reverse=True)
    return result


def best_top(votes, now, policy=WEIGHTED):
    agg = aggregate(votes, now, policy.window_sec)
    if agg and agg[0][1] >= policy.top_min:
        return agg[0][0], agg[0][1], agg[0][2], agg

    # No single top-row value is strong enough: vote per character position.
    recent = [v for v in votes if now - v["time"] <= policy.window_sec
              and len(v["value"]) == 3 and v["value"].isdigit()]
    if not recent:
        return "", 0.0, 0, agg

    pos = [defaultdict(float) for _ in range(3)]
    for v in recent:
        for i, ch in enumerate(v["value"]):
            pos[i][ch] += v["weight"]
    if any(not p for p in pos):
        return "", 0.0, 0, agg

    candidate = "".join(max(p.items(), key=lambda kv: kv[1])[0] for p in pos)
    # A candidate built digit by digit is only as well supported as its
    # weakest digit. Summing the three positions counted every read three
    # times, so a single read could pass MIN_TOP_WEIGHT on its own.
    weight = min(pos[i][candidate[i]] for i in range(3))
    if weight >= policy.top_min:
        return candidate, weight, len(recent), agg
    return "", 0.0, 0, agg


def best_bottom(votes, now, policy=WEIGHTED):
    agg = aggregate(votes, now, policy.window_sec)
    if not agg:
        return "", 0.0, 0, agg
    return agg[0][0], agg[0][1], agg[0][2], agg


def square_candidate(top_votes, bottom_votes, square_readings, t, policy=WEIGHTED):
    """
    Decides whether the square-plate votes support a plate at time t.
    Returns (plate, confidence), or ("", 0.0) if they do not.

    The top and bottom rows are voted on separately, so their winners can come
    from different frames, even from two different cars in view within
    WINDOW_SEC. The plate is accepted only if its top and bottom were read
    together in at least one frame of the last WINDOW_SEC. square_readings are
    the normalized per-frame readings, newest last, including the current one.
    """
    best_t, top_weight, _, _ = best_top(top_votes, t, policy)
    best_b, bottom_weight, _, _ = best_bottom(bottom_votes, t, policy)
    if not best_t or not best_b:
        return "", 0.0
    plate = best_t + best_b[2:] + best_b[:2]
    if not valid_kz_plate(plate):
        return "", 0.0
    if top_weight < policy.top_min or bottom_weight < policy.bottom_min:
        return "", 0.0
    read_together = any(
        r["top"] == best_t and r["bottom"] == best_b
        and t - r["time"] <= policy.window_sec
        for r in square_readings
    )
    if not read_together:
        return "", 0.0
    if policy.fixed_weight is not None:
        # Веса здесь это число совпадений, а не уверенность, и превращать их
        # в уверенность нельзя: получилось бы число, похожее на уверенность,
        # но ничего о ней не говорящее. None честнее, и дальше по коду оно
        # доезжает до ответа как null.
        return plate, None
    return plate, min(0.99, (top_weight + bottom_weight) / 4.0)


def aggregate_all(votes):
    """Totals over the whole history. Used for end-of-run reports only."""
    groups = {}
    for v in votes:
        value = v["value"]
        groups.setdefault(value, {"weight": 0.0, "count": 0, "best": 0.0})
        groups[value]["weight"] += float(v["weight"])
        groups[value]["count"] += 1
        groups[value]["best"] = max(groups[value]["best"], float(v["conf"]))
    rows = [(k, d["weight"], d["count"], d["best"]) for k, d in groups.items()]
    rows.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
    return rows


# ---------------------------------------------------------------------------
# Per-frame profiling
# ---------------------------------------------------------------------------

class _FrameProfile:
    """Collects stage timings for one frame. Measurement only."""

    def __init__(self):
        self._start = time.perf_counter()
        self.data = {
            "yolo_ms": 0.0,
            "ocr_calls": 0,
            "ocr_total_ms": 0.0,
            "ocr_square_ms": 0.0,
            "ocr_normal_ms": 0.0,
            "ocr_fallback_ms": 0.0,
            "crop_ms": 0.0,
            "voting_ms": 0.0,
            "worker_detail": [],
        }

    @contextmanager
    def timed(self, key):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.data[key] += (time.perf_counter() - t0) * 1000.0

    def record_worker(self, stage, workers):
        timing = getattr(workers, "last_timing", None)
        if timing:
            self.data["worker_detail"].append({"stage": stage, **timing})

    def ocr(self, kind, workers, image, mode):
        """Run one OCR call; kind is 'normal', 'square' or 'fallback'."""
        t0 = time.perf_counter()
        payload = workers.ocr(image, mode)
        dt = (time.perf_counter() - t0) * 1000.0
        self.data["ocr_calls"] += 1
        self.data["ocr_total_ms"] += dt
        self.data[f"ocr_{kind}_ms"] += dt
        self.record_worker(f"ocr_{kind}", workers)
        return payload

    def finish(self, plate_type, had_detection):
        self.data["total_ms"] = (time.perf_counter() - self._start) * 1000.0
        self.data["plate_type"] = plate_type
        self.data["had_detection"] = had_detection
        return self.data


# ---------------------------------------------------------------------------
# Recognizer
# ---------------------------------------------------------------------------

class LPRRecognizer:
    """
    One recognizer is one independent recognition session. Create one per
    camera or session_id; sharing an instance mixes the vote history of
    unrelated video sources.

    Per incoming frame:
        result = recognizer.process_frame(frame_bgr, workers, t)

    `workers` must provide:
        workers.detect(frame_bgr) -> (x1, y1, x2, y2, conf) or None
        workers.ocr(crop_bgr, mode) -> the OCR worker's "normal" or "square"
            payload (see workers/ocr_gpu_worker.py)

    `t` is a clock in seconds: time.monotonic() for a live server, or
    frame_idx / fps for a recorded video. A result whose t is older than the
    last accepted decision is ignored by consider_plate, so a late
    asynchronous OCR result cannot roll the confirmed plate back.
    """

    def __init__(self, plate_hold_sec=PLATE_HOLD_SEC, policy=WEIGHTED,
                 single_row_only=False):
        # policy решает, как номер считается подтверждённым: взвешиванием по
        # уверенности (WEIGHTED) или по числу совпадений (AGREEMENT, для
        # распознавателя без уверенности, то есть для нашего). Значение по
        # умолчанию WEIGHTED оставлено ради регрессионных тестов; сервис
        # всегда создаёт распознаватель с AGREEMENT, см. app/main.py.
        self.policy = POLICIES[policy] if isinstance(policy, str) else policy
        self.single_row_only = single_row_only
        self.plate_hold_sec = plate_hold_sec
        self.last_confirmed_read_t = float("-inf")
        self.plate_clears = 0

        self.confirmed_plate = ""
        self.confirmed_history = []
        self.last_decision_t = -1.0
        self.out_of_order_decisions = 0
        self.switch_events = []
        self.last_profile = None

        self.top_votes = []
        self.bottom_votes = []
        self.final_votes = []

        self.square_ocr_frames = 0
        self.detections = 0
        self.ocr_attempts = 0

        self.square_readings = []
        self.last_box = (0, 0, 0.0)
        self.normal_readings = []
        # Diagnostics only: the row votes behind the last square decision.
        self.last_square_votes = None

        self._latest_t = float("-inf")

    def consider_plate(self, candidate, conf, t, source, fresh=True):
        """Returns True if this read switched the confirmed plate.

        fresh=False means the candidate comes from accumulated votes rather
        than from what was read in this frame. It counts for switching exactly
        as before, but does not keep the confirmed plate on screen."""
        if not candidate or not valid_kz_plate(candidate):
            return False

        if t < self.last_decision_t - 1e-6:
            self.out_of_order_decisions += 1
            return False

        self.last_decision_t = max(self.last_decision_t, t)

        self.confirmed_history = [
            x for x in self.confirmed_history
            if t - x["time"] <= self.policy.switch_window_sec
        ]
        self.confirmed_history.append({
            "plate": candidate, "time": t, "source": source, "conf": conf,
        })

        if candidate == self.confirmed_plate:
            if fresh:
                self.last_confirmed_read_t = t
            return False

        reads = [x for x in self.confirmed_history if x["plate"] == candidate]
        # Короткий путь «одно очень уверенное чтение» существует только там,
        # где уверенность есть. У правила AGREEMENT strong_conf равен None, и
        # тогда остаётся единственный путь: нужное число совпадений.
        strong_single = (self.policy.strong_conf is not None
                         and conf is not None
                         and float(conf) >= self.policy.strong_conf)
        if len(reads) >= self.policy.switch_reads or strong_single:
            old = self.confirmed_plate
            self.confirmed_plate = candidate
            self.confirmed_history = []
            self.last_confirmed_read_t = t
            self.switch_events.append({
                "time": round(t, 2), "from": old, "to": candidate,
                "source": source,
                "confidence": None if conf is None else round(float(conf), 3),
            })
            return True
        return False

    def process_frame(self, frame_bgr, workers, t):
        """
        Runs detect -> classify -> OCR -> normalize -> vote -> switch for one
        frame. Stage timings are stored in self.last_profile.
        """
        prof = _FrameProfile()

        with prof.timed("yolo_ms"):
            det = workers.detect(frame_bgr)
        prof.record_worker("yolo", workers)

        changed = False
        ocr_confidence = None
        plate_type = None
        raw_text = None

        if det is not None:
            self.detections += 1
            x1, y1, x2, y2, det_conf = det
            with prof.timed("crop_ms"):
                crop = frame_bgr[y1:y2, x1:x2]

            if crop.size != 0 and looks_like_plate(x2 - x1, y2 - y1, frame_bgr.shape[1], frame_bgr.shape[0]):
                h, w = crop.shape[:2]
                aspect = w / max(1, h)
                # Kept for the diagnostics below: which box shape sent
                # this crop down which path. A single-row plate seen at
                # an angle has a taller box than its plate, and the
                # recorded shape is how that is checked rather than
                # argued about.
                self.last_box = (w, h, round(aspect, 2))

                # single_row_only: вырез не делится на строки, а читается
                # целиком. Нужно для распознавателя, который сам читает обе
                # строки и отдаёт номер одним текстом (так работает модель
                # через шлюз). Делить вырез для него незачем, а главное,
                # голосование по строкам требует, чтобы верхняя и нижняя
                # приходили отдельно, чего он не даёт. Для распознавателя,
                # который отдаёт строки по отдельности, флаг выключается, и
                # тогда работает квадратный путь ниже.
                if (self.single_row_only or aspect > SQUARE_ASPECT_MAX
                        or w < MIN_SQUARE_W or h < MIN_SQUARE_H):
                    plate_type = "normal"
                    self.ocr_attempts += 1
                    payload = prof.ocr("normal", workers, crop, "normal")
                    with prof.timed("voting_ms"):
                        changed, ocr_confidence, raw_text = self._handle_normal_result(payload, t)
                else:
                    plate_type = "square"
                    self.square_ocr_frames += 1
                    if self.square_ocr_frames % OCR_EVERY_N_DETECTIONS == 1:
                        self.ocr_attempts += 1
                        py = max(2, int(h * .05))
                        px = max(2, int(w * .03))
                        sq = crop[py:max(py + 1, h - py), px:max(px + 1, w - px)]

                        payload = prof.ocr("square", workers, sq, "square")
                        with prof.timed("voting_ms"):
                            sq_changed, _ = self._handle_square_result(payload, t)
                        changed = changed or sq_changed
                        prof.data["square_votes"] = self.last_square_votes

                        # What OCR read in this frame. A square plate is only
                        # as reliable as its weaker row. (The vote-based
                        # confidence used for switching is not reported: it
                        # stays high for seconds after the plate leaves view.)
                        top_text = str(payload.get("top_text", ""))
                        bottom_text = str(payload.get("bottom_text", ""))
                        row_confs = [self._conf_of(payload, "top_conf"),
                                     self._conf_of(payload, "bottom_conf")]
                        ocr_confidence = (None if None in row_confs
                                          else min(row_confs))
                        raw_text = f"{top_text} / {bottom_text}"

                        # While a vehicle is confirmed, also read this crop as
                        # a single-row plate. When the camera moves to the next
                        # car, YOLO's box can stay square for a few frames;
                        # this catches the new plate sooner.
                        if self.confirmed_plate:
                            payload2 = prof.ocr("fallback", workers, crop, "normal")
                            with prof.timed("voting_ms"):
                                n_changed, n_conf, n_raw = self._handle_normal_result(payload2, t)
                            changed = changed or n_changed
                            if n_changed:
                                # this single-row read is what confirmed the plate
                                ocr_confidence, raw_text = n_conf, n_raw

        self.drop_stale_plate(t)
        self.last_profile = prof.finish(plate_type, det is not None)

        return {
            "plate": self.confirmed_plate,
            "confirmed": bool(self.confirmed_plate),
            "bbox": list(det) if det is not None else None,
            "confidence": float(det[4]) if det is not None else 0.0,
            "changed": changed,
            "ocr_confidence": ocr_confidence,
            "plate_type": plate_type,
            "raw_text": raw_text,
        }

    def drop_stale_plate(self, t):
        """Clears the confirmed plate if it was not read for plate_hold_sec.
        Returns True if the plate was cleared."""
        if not self.confirmed_plate or not self.plate_hold_sec or self.plate_hold_sec <= 0:
            return False
        if t - self.last_confirmed_read_t <= self.plate_hold_sec:
            return False
        self.confirmed_plate = ""
        self.confirmed_history = []
        # The square votes behind the cleared plate are stale by definition;
        # left in place they would confirm it again from old evidence as soon
        # as the next square crop is read, even if that crop is another car.
        self.top_votes.clear()
        self.bottom_votes.clear()
        self.final_votes.clear()
        self.plate_clears += 1
        return True

    @staticmethod
    def _conf_of(payload, key):
        """Уверенность из ответа распознавателя, либо None, если её там нет.

        Отсутствие ключа и значение 0.0 это разные вещи, и путать их нельзя.
        Распознаватель через шлюз уверенность не возвращает вовсе, и ноль на
        её месте выглядел бы как «прочитал, но совсем не уверен», тогда как
        на самом деле это «уверенность неизвестна». Решения на этом стоят
        разные, и в ответ сервиса уходит null, а не ноль.
        """
        if key not in payload or payload[key] is None:
            return None
        try:
            return float(payload[key])
        except (TypeError, ValueError):
            return None

    def _handle_normal_result(self, payload, t):
        raw_text = str(payload.get("text", ""))
        raw_conf = self._conf_of(payload, "conf")
        if raw_conf is None and self.policy.strong_conf is not None:
            # Правило взвешивания требует числа. Прежнее поведение: нет ключа
            # значит ноль.
            raw_conf = 0.0
        text = extract_plate(clean_text(raw_text))
        changed = False
        if valid_kz_plate(text):
            bw, bh, basp = self.last_box
            self.normal_readings.append({
                "time": round(t, 2), "plate": text,
                "confidence": None if raw_conf is None else round(raw_conf, 3),
                "box_w": bw, "box_h": bh, "box_aspect": basp,
            })
            changed = self.consider_plate(text, raw_conf, t, "normal")
        self._trim_history(t)
        return changed, raw_conf, raw_text

    def _handle_square_result(self, payload, t):
        top_text = payload.get("top_text", "")
        bottom_text = payload.get("bottom_text", "")
        top_conf = self._conf_of(payload, "top_conf")
        bottom_conf = self._conf_of(payload, "bottom_conf")
        if self.policy.strong_conf is not None:
            # Прежнее поведение для правила взвешивания: нет ключа значит ноль.
            top_conf = 0.0 if top_conf is None else top_conf
            bottom_conf = 0.0 if bottom_conf is None else bottom_conf

        top = normalize_top(top_text)
        bottom = normalize_bottom(bottom_text)

        bw, bh, basp = self.last_box
        self.square_readings.append({
            "time": round(t, 2),
            "box_w": bw, "box_h": bh, "box_aspect": basp,
            "top_raw": top_text, "top": top,
            "top_conf": None if top_conf is None else round(top_conf, 3),
            "bottom_raw": bottom_text, "bottom": bottom,
            "bottom_conf": None if bottom_conf is None else round(bottom_conf, 3),
        })

        add_vote(self.top_votes, top, top_conf, t, self.policy)
        add_vote(self.bottom_votes, bottom, bottom_conf, t, self.policy)

        candidate, vote_conf = square_candidate(
            self.top_votes, self.bottom_votes, self.square_readings, t,
            self.policy)

        # Diagnostics only: which row values are competing right now, and how
        # much weight each has. Needed to judge when two candidates are too
        # close to tell apart. Does not affect any decision.
        self.last_square_votes = {
            "top": [{"value": v, "weight": round(w, 3), "reads": c}
                    for v, w, c, _ in aggregate(self.top_votes, t,
                                                 self.policy.window_sec)[:4]],
            "bottom": [{"value": v, "weight": round(w, 3), "reads": c}
                       for v, w, c, _ in aggregate(self.bottom_votes, t,
                                                    self.policy.window_sec)[:4]],
            "chosen": candidate,
        }

        changed = False
        if top_conf is None or bottom_conf is None:
            final_conf = None
        else:
            final_conf = max(top_conf, bottom_conf)
        if candidate:
            final_conf = vote_conf
            add_vote(self.final_votes, candidate, final_conf, t, self.policy)
            # Both row weights above their minimums is enough to confirm;
            # waiting for the combined vote to repeat made short-lived square
            # plates (e.g. 979CBB02) leave the frame before confirmation.
            #
            # The candidate comes from votes over the last WINDOW_SEC seconds,
            # so it can still name a car that has already left while the
            # camera reads the next one. It keeps the plate on screen only if
            # this frame's own reading matches at least one of its rows.
            plate_bottom = candidate[6:8] + candidate[3:6]
            fresh = (top == candidate[:3]) or (bottom == plate_bottom)
            changed = self.consider_plate(candidate, final_conf, t, "square", fresh=fresh)

        self._trim_history(t)
        return changed, final_conf

    def _trim_history(self, t):
        """Keeps per-session memory bounded without affecting any decision."""
        self._latest_t = max(self._latest_t, t)
        cutoff = self._latest_t - VOTE_RETENTION_SEC
        for votes in (self.top_votes, self.bottom_votes, self.final_votes):
            if votes and votes[0]["time"] < cutoff:
                votes[:] = [v for v in votes if v["time"] >= cutoff]
        for readings in (self.square_readings, self.normal_readings):
            if len(readings) > MAX_READINGS:
                del readings[:len(readings) - MAX_READINGS]
