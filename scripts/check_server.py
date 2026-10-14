"""Проверка живого HTTP-сервера: поднимает uvicorn и бьёт по эндпоинтам.

Запросы идут через opener с пустым ProxyHandler({}), чтобы системные
HTTP_PROXY/HTTPS_PROXY не уводили запросы к 127.0.0.1 в прокси
(иначе на машинах с локальным прокси все роуты отвечают 503).

Сквозной путь: POST /v1/tasks -> GET статуса -> SSE -> /metrics.
БД — временный файл, чтобы не плодить local.db в репозитории.

Запуск:
    uv run python scripts/check_server.py
"""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HOST = "127.0.0.1"
PORT = 8011
BASE = f"http://{HOST}:{PORT}"
ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "check_server.sqlite3"

ok = True

# Пустой ProxyHandler отключает использование proxy-переменных окружения.
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def check(label: str, condition: bool, detail: str = "") -> None:
    global ok
    print(f"  [{'OK  ' if condition else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not condition:
        ok = False


def _read(request: urllib.request.Request) -> tuple[int, str]:
    try:
        with OPENER.open(request, timeout=10) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:  # 404/422 и т.п. — валидный ответ сервера
        return exc.code, exc.read().decode("utf-8")
    except Exception as exc:  # noqa: BLE001
        return -1, f"{type(exc).__name__}: {exc}"


def fetch(path: str) -> tuple[int, str]:
    return _read(urllib.request.Request(BASE + path))


def post_json(path: str, payload: dict, headers: dict | None = None) -> tuple[int, str]:
    request = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    return _read(request)


def port_open() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex((HOST, PORT)) == 0


if DB_PATH.exists():
    DB_PATH.unlink()

env = {**os.environ, "DB_URL": f"sqlite+aiosqlite:///{DB_PATH.as_posix()}"}
proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "aicore.api.app:app", "--host", HOST, "--port", str(PORT)],
    cwd=ROOT,
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)

try:
    # Ждём, пока порт начнёт отвечать.
    deadline = time.time() + 30
    while time.time() < deadline:
        if port_open():
            break
        if proc.poll() is not None:
            print("  uvicorn умер при старте:")
            print(proc.stdout.read().decode("utf-8", "replace") if proc.stdout else "")
            sys.exit(1)
        time.sleep(0.4)

    print("== live server ==")
    check("server listens on port", port_open(), f"{HOST}:{PORT}")

    code, body = fetch("/health")
    check("GET /health -> 200", code == 200, str(code))
    check("health reports stub backends", '"llm":"stub"' in body and '"broker":"memory"' in body)
    check("health has no secrets", "sk-" not in body and "://" not in body)

    code, _ = fetch("/ready")
    check("GET /ready -> 200", code == 200, str(code))

    code, _ = fetch("/version")
    check("GET /version -> 200", code == 200, str(code))

    code, _ = fetch("/openapi.json")
    check("GET /openapi.json -> 200", code == 200, str(code))

    code, _ = fetch("/definitely-missing")
    check("unknown route -> 404", code == 404, str(code))

    print("\n== task flow (POST -> task -> answer) ==")
    code, body = post_json("/v1/tasks", {"question": "hello from check_server"})
    check("POST /v1/tasks -> 202", code == 202, str(code))
    check("task succeeded", '"status":"succeeded"' in body)
    check("answer present", '"answer"' in body)
    check("metrics present (cost_usd)", '"cost_usd"' in body)
    task_id = json.loads(body).get("id") if code == 202 else ""

    code, body = fetch(f"/v1/tasks/{task_id}")
    check("GET /v1/tasks/{id} -> 200", code == 200, str(code))
    check("status round-trips", '"status":"succeeded"' in body)

    code, _ = fetch("/v1/tasks/not-a-uuid")
    check("invalid uuid -> 422", code == 422, str(code))

    print("\n== idempotency ==")
    _, first = post_json("/v1/tasks", {"question": "dup"}, {"Idempotency-Key": "live-1"})
    _, second = post_json("/v1/tasks", {"question": "dup"}, {"Idempotency-Key": "live-1"})
    check("same id on replay", json.loads(first)["id"] == json.loads(second)["id"])

    print("\n== SSE stream ==")
    code, body = fetch(f"/v1/tasks/{task_id}/stream")
    check("GET stream -> 200", code == 200, str(code))
    check("stream ends with done event", "event: done" in body)

    print("\n== metrics ==")
    code, body = fetch("/metrics")
    check("GET /metrics -> 200", code == 200, str(code))
    check("metrics expose task counter", "aicore_tasks_total" in body)
    check("metrics expose llm cost", "aicore_llm_cost_usd_total" in body)
finally:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    if DB_PATH.exists():
        DB_PATH.unlink()

print()
print("LIVE SERVER " + ("PASSED" if ok else "FAILED"))
sys.exit(0 if ok else 1)
