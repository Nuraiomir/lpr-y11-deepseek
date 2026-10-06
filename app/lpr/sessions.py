"""
Состояние распознавания на каждого клиента.

У каждого session_id свой LPRRecognizer, поэтому кадры с разных телефонов и
от разных сотрудников не делят между собой голосование. Это не удобство, а
правило корректности: голосование копит прочитанные номера, и если состояние
общее, номер одной машины попадёт в голоса другой.

Кадры одной сессии обрабатываются по одному, разные сессии параллельно.
Сессия, в которую не присылали кадров SESSION_TTL_SEC, удаляется, иначе
память растёт на каждый новый идентификатор.
"""
from __future__ import annotations

import threading
import time

from app.lpr.recognizer import LPRRecognizer


class Session:
    __slots__ = ("recognizer", "lock", "last_used")

    def __init__(self, plate_hold_sec: float, policy="weighted",
                 single_row_only=False):
        self.recognizer = LPRRecognizer(plate_hold_sec=plate_hold_sec,
                                        policy=policy,
                                        single_row_only=single_row_only)
        self.lock = threading.Lock()
        self.last_used = time.monotonic()


class SessionStore:
    """session_id -> распознаватель. Потокобезопасно, с истечением и пределом."""

    def __init__(self, ttl_sec: int = 1800, max_sessions: int = 1000,
                 plate_hold_sec: float = 2.0, policy="weighted",
                 single_row_only=False):
        self._ttl = ttl_sec
        self._max = max_sessions
        self.plate_hold_sec = plate_hold_sec
        # Правило подтверждения одно на все сессии: его задаёт то, кто читает
        # символы, а это один на процесс.
        self.policy = policy
        self.single_row_only = single_row_only
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> Session:
        now = time.monotonic()
        with self._lock:
            expired = [k for k, s in self._sessions.items()
                       if now - s.last_used > self._ttl]
            for key in expired:
                del self._sessions[key]

            session = self._sessions.get(session_id)
            if session is None:
                if len(self._sessions) >= self._max:
                    # Предел нужен против роста памяти, когда клиент
                    # присылает новый идентификатор на каждый кадр. Вытесняем
                    # самую давнюю: она с наибольшей вероятностью уже не нужна.
                    oldest = min(self._sessions,
                                 key=lambda k: self._sessions[k].last_used)
                    del self._sessions[oldest]
                session = self._sessions[session_id] = Session(
                    self.plate_hold_sec, self.policy, self.single_row_only)
            session.last_used = now
            return session

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)
