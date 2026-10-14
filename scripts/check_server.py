"""Проверка живого HTTP-сервера: поднимает uvicorn и бьёт по эндпоинтам.

Запросы идут через opener с пустым ProxyHandler({}), чтобы системные
HTTP_PROXY/HTTPS_PROXY не уводили запросы к 127.0.0.1 в прокси
(иначе на машинах с локальным прокси все роуты отвечают 503).

Запуск:
    uv run python scripts/check_server.py
"""

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

ok = True

# Пустой ProxyHandler отключает использование proxy-переменных окружения.
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def check(label: str, condition: bool, detail: str = "") -> None:
    global ok
    print(f"  [{'OK  ' if condition else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not condition:
        ok = False


def fetch(path: str) -> tuple[int, str]:
    try:
        with OPENER.open(BASE + path, timeout=5) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:  # 404 и т.п. — тоже валидный ответ сервера
        return exc.code, exc.read().decode("utf-8")
    except Exception as exc:  # noqa: BLE001
        return -1, f"{type(exc).__name__}: {exc}"


def port_open() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex((HOST, PORT)) == 0


proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "aicore.api.app:app", "--host", HOST, "--port", str(PORT)],
    cwd=ROOT,
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

    code, body = fetch("/ready")
    check("GET /ready -> 200", code == 200, str(code))

    code, body = fetch("/version")
    check("GET /version -> 200", code == 200, str(code))

    code, _ = fetch("/openapi.json")
    check("GET /openapi.json -> 200", code == 200, str(code))

    code, _ = fetch("/definitely-missing")
    check("unknown route -> 404", code == 404, str(code))
finally:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()

print()
print("LIVE SERVER " + ("PASSED" if ok else "FAILED"))
sys.exit(0 if ok else 1)
