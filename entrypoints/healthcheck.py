#!/usr/bin/env python3
"""
Проверка состояния для HEALTHCHECK в Dockerfile.

Спрашивает /health/live, то есть «процесс жив», а не «готов принимать кадры».
Разница существенная: модели грузятся несколько секунд, и если проверка будет
смотреть на готовность, Docker убьёт контейнер на середине загрузки и начнёт
заново, и так по кругу.

Готовность проверяет кластер отдельной пробой на /health/ready, и он умеет
ждать: пока готовности нет, трафик на под не идёт, но и перезапуска нет.

Код выхода 0 здоров, 1 нет. Ничего не печатает при успехе: вывод проверки
попадает в журнал каждые 30 секунд.
"""
import os
import sys
import urllib.error
import urllib.request

PORT = os.environ.get("PORT", "8080")
URL = f"http://127.0.0.1:{PORT}/health/live"

try:
    with urllib.request.urlopen(URL, timeout=4) as response:
        sys.exit(0 if response.status == 200 else 1)
except urllib.error.HTTPError as exc:
    # 503 это ответ сервиса, а не отсутствие сервиса: значит загрузка упала.
    print(f"{URL}: код {exc.code}", file=sys.stderr)
    sys.exit(1)
except Exception as exc:  # noqa: BLE001
    print(f"{URL}: {exc!r}", file=sys.stderr)
    sys.exit(1)
