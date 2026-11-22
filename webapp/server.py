"""GalsenAI chat web app: static UI + streaming proxy to any OpenAI-compatible server (vLLM, mlx_lm.server).

    VLLM_BASE_URL=http://localhost:8000/v1 python3 webapp/server.py [--port 8080]

Standard library only. Env: VLLM_BASE_URL, MODEL (default: first model the server lists),
SYSTEM_PROMPT, API_KEY (sent as Bearer if set).
"""

import argparse
import json
import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STATIC = Path(__file__).parent / "static"
BASE = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1").rstrip("/")
SYSTEM = os.environ.get("SYSTEM_PROMPT", "")
TYPES = {".html": "text/html", ".css": "text/css", ".js": "text/javascript", ".svg": "image/svg+xml"}
MAX_BODY = 1_000_000


def upstream(path: str, body: dict | None = None):
    headers = {"Content-Type": "application/json"}
    if key := os.environ.get("API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    data = json.dumps(body).encode() if body is not None else None
    return urllib.request.urlopen(urllib.request.Request(BASE + path, data, headers), timeout=300)


def model_name() -> str:
    if name := os.environ.get("MODEL"):
        return name
    with upstream("/models") as r:
        return json.load(r)["data"][0]["id"]


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code: int, msg: str) -> None:
        self._send(code, json.dumps({"error": msg}).encode())

    def do_GET(self) -> None:
        if self.path == "/api/config":
            try:
                self._send(200, json.dumps({"model": model_name(), "system": SYSTEM}).encode())
            except (urllib.error.URLError, OSError, KeyError, IndexError) as e:
                self._error(502, f"Serveur d'inférence injoignable ({BASE}). Lancez ./scripts/serve.sh, ou démarrez vLLM / mlx_lm.server puis VLLM_BASE_URL. Détail : {e}")
            return
        rel = "index.html" if self.path in ("/", "") else self.path.lstrip("/").split("?")[0]
        f = (STATIC / rel).resolve()
        if STATIC.resolve() not in f.parents or not f.is_file():
            return self._error(404, "not found")
        self._send(200, f.read_bytes(), TYPES.get(f.suffix, "application/octet-stream"))

    def do_POST(self) -> None:
        if self.path != "/api/chat":
            return self._error(404, "not found")
        n = int(self.headers.get("Content-Length", 0))
        if not 0 < n <= MAX_BODY:
            return self._error(413, "bad body size")
        try:
            req = json.loads(self.rfile.read(n))
            messages = req["messages"]
            assert isinstance(messages, list) and messages
        except (ValueError, KeyError, AssertionError):
            return self._error(400, "expected {messages: [...]}")
        body = {
            "model": model_name(),
            "messages": messages,
            "stream": True,
            "max_tokens": min(int(req.get("max_tokens", 512)), 2048),
            "temperature": float(req.get("temperature", 0.7)),
        }
        try:
            with upstream("/chat/completions", body) as r:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                while chunk := r.readline():
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except urllib.error.HTTPError as e:
            self._error(e.code, e.read().decode(errors="replace")[:500])
        except (urllib.error.URLError, OSError) as e:
            self._error(502, f"Serveur d'inférence injoignable ({BASE}). Lancez ./scripts/serve.sh, ou démarrez vLLM / mlx_lm.server puis VLLM_BASE_URL. Détail : {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    print(f"GalsenAI chat on http://{a.host}:{a.port}  ->  {BASE}")
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()
