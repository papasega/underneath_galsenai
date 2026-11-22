"""GalsenAI chat web app: static UI + two selectable models.

  galsenai-gemma     streaming proxy to an OpenAI-compatible server (vLLM, mlx_lm.server)
  galsenai-nanochat  the 12M nanochat-style checkpoint, generated in-process (needs the project venv)

    VLLM_BASE_URL=http://localhost:8000/v1 .venv/bin/python webapp/server.py [--port 8080]

Env: VLLM_BASE_URL, MODEL (upstream model name, default: first one listed), ACTIVE_MODEL (name shown in the UI),
SYSTEM_PROMPT (Gemma system message, default: GalsenAI assistant; "" to disable), API_KEY (Bearer), NANOCHAT_CHECKPOINT, NANOCHAT_TOKENIZER.
"""

import argparse
import json
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rag  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).parent / "static"
BASE = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1").rstrip("/")
DEFAULT_SYSTEM = (
    "Tu es l'assistant de la communauté GalsenAI. Tu réponds en français, de façon claire et concise, aux questions "
    "liées aux thématiques du groupe de discussion : modèles de langage (LLM), agents et outils, entraînement et "
    "fine-tuning, NLP et langues locales (wolof), Python et développement, maths et recherche, infrastructure et GPU, "
    "carrière et études en IA, événements et communauté au Sénégal. Appuie-toi sur ce qui a été discuté dans le "
    "groupe ; si tu n'es pas sûr ou si l'information peut être périmée, dis-le au lieu d'inventer."
)
SYSTEM = os.environ.get("SYSTEM_PROMPT", DEFAULT_SYSTEM)  # SYSTEM_PROMPT="" disables it; applied to Gemma only
NANO_CKPT = Path(os.environ.get("NANOCHAT_CHECKPOINT", ROOT / "data/private/checkpoints/galsenai_sft.pt"))
NANO_TOK = Path(os.environ.get("NANOCHAT_TOKENIZER", ROOT / "data/private/galsenai_tok/galsenai.model"))
TYPES = {".html": "text/html", ".css": "text/css", ".js": "text/javascript", ".svg": "image/svg+xml"}
MAX_BODY = 1_000_000
MODELS_DIR = ROOT / "data/private/models"  # full fine-tuning and merged models: gemma4-full-<name>/ (see scripts/run_full.sh)
_nano: dict = {}
_nano_lock = threading.Lock()


def upstream(path: str, body: dict | None = None):
    headers = {"Content-Type": "application/json"}
    if key := os.environ.get("API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    data = json.dumps(body).encode() if body is not None else None
    return urllib.request.urlopen(urllib.request.Request(BASE + path, data, headers), timeout=300)


def upstream_model() -> str:
    if name := os.environ.get("MODEL"):
        return name
    with upstream("/models") as r:
        return json.load(r)["data"][0]["id"]


def gemma_info() -> dict:
    """Name shown in the UI. mlx_lm.server calls its preloaded model "default_model": show the real one instead."""
    try:
        real = upstream_model()
        detail = os.environ.get("ACTIVE_MODEL") or (real if real != "default_model" else "modèle actif")
        return {"id": "gemma", "name": "galsenai-gemma", "detail": detail, "available": True}
    except (urllib.error.URLError, OSError, KeyError, IndexError):
        return {"id": "gemma", "name": "galsenai-gemma", "detail": f"serveur injoignable ({BASE})", "available": False}


def full_models() -> dict[str, Path]:
    """Local full-fine-tuned / merged Gemma checkpoints, served by mlx_lm.server loading them by path on demand."""
    return {d.name: d for d in sorted(MODELS_DIR.glob("gemma4-full-*")) if (d / "config.json").is_file()}


def nanochat_info() -> dict:
    ok = NANO_CKPT.is_file() and NANO_TOK.is_file()
    return {"id": "nanochat", "name": "galsenai-nanochat", "available": ok,
            "detail": "modèle de 12 M de paramètres, style nanochat" if ok else "checkpoint introuvable"}


def nano_stream(messages: list[dict], max_tokens: int, temperature: float):
    """Yield text deltas from the nanochat-style model (loaded on first use, one request at a time)."""
    with _nano_lock:
        if not _nano:
            sys.path.insert(0, str(ROOT / "src"))
            sys.path.insert(0, str(ROOT))
            import torch
            from galsenai.chat import ASSISTANT_END, encode_prompt, load_tokenizer
            from galsenai.checkpoint import load_checkpoint
            _nano.update(torch=torch, encode=encode_prompt, stop=ASSISTANT_END, tok=load_tokenizer(NANO_TOK),
                         model=load_checkpoint(NANO_CKPT, "cpu").eval())
        torch, model, tok = _nano["torch"], _nano["model"], _nano["tok"]
        block = model.cfg.block_size
        ids = _nano["encode"](tok, messages)
        while len(ids) > block - max_tokens and len(messages) > 1:  # drop the oldest turn pair
            messages = messages[2:]
            ids = _nano["encode"](tok, messages)
        stop, gen, sent = tok.special_tokens[_nano["stop"]], [], ""
        with torch.inference_mode():
            out = model(torch.tensor([ids]))
            for _ in range(min(max_tokens, block - len(ids))):
                logits = out.logits[0, -1] / max(temperature, 1e-4)
                top = torch.topk(logits, 50)
                nxt = top.indices[torch.multinomial(top.values.softmax(-1), 1)].item()
                if nxt == stop:
                    break
                gen.append(nxt)
                text = tok.decode(gen)
                if not text.endswith("�") and len(text) > len(sent):  # wait for complete UTF-8 characters
                    yield text[len(sent):]
                    sent = text
                out = model(torch.tensor([[nxt]]), past=out.past)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:  # quieter console
        if "/api/chat" in fmt % args:
            super().log_message(fmt, *args)

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
            models = [gemma_info(), *({"id": n, "name": n, "available": True,
                                      "detail": "Gemma 4 E2B, poids complets (chargé à la demande par mlx_lm.server)"}
                                     for n in full_models()), nanochat_info()]
            default = next((m["id"] for m in models if m["available"]), "gemma")
            return self._send(200, json.dumps({"models": models, "default": default}).encode())
        rel = "index.html" if self.path in ("/", "") else self.path.lstrip("/").split("?")[0]
        f = (STATIC / rel).resolve()
        if STATIC.resolve() not in f.parents or not f.is_file():
            return self._error(404, "not found")
        self._send(200, f.read_bytes(), TYPES.get(f.suffix, "application/octet-stream"))

    def _sse_start(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

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
            which = req.get("model", "gemma")
            assert which in ("gemma", "nanochat", *full_models())
            max_tokens = min(int(req.get("max_tokens", 512)), 2048)
            temperature = float(req.get("temperature", 0.7))
            method = req.get("rag", "off")
            assert method in ("off", "bm25", "dense", "hybrid")
        except (ValueError, KeyError, AssertionError):
            return self._error(400, "expected {messages: [...], model: 'gemma'|'nanochat'}")
        try:
            if which == "nanochat":
                self._nanochat(messages, max_tokens, temperature)
            else:
                self._gemma(messages, max_tokens, temperature, method, which)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the browser closed the stream

    def _nanochat(self, messages: list[dict], max_tokens: int, temperature: float) -> None:
        if not nanochat_info()["available"]:
            return self._error(503, f"checkpoint introuvable : {NANO_CKPT}")
        try:
            gen = nano_stream(messages, max_tokens, temperature)
            first = next(gen, "")
        except ValueError as e:
            return self._error(400, str(e))
        self._sse_start()
        for piece in [first, *gen]:
            chunk = {"choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}
            self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")

    def _gemma(self, messages: list[dict], max_tokens: int, temperature: float, method: str = "off", which: str = "gemma") -> None:
        try:
            sources: list[dict] = []
            if method != "off":  # the passages go into the last user turn; the client history stays clean
                sources = rag.retrieve(rag.query_from(messages), 4, method)
                if sources:
                    messages = [*messages[:-1], {**messages[-1], "content": rag.build_prompt(messages[-1]["content"], sources)}]
            if SYSTEM and messages[0].get("role") != "system":
                messages = [{"role": "system", "content": SYSTEM}, *messages]
            body = {"model": str(full_models()[which]) if which != "gemma" else upstream_model(), "messages": messages, "stream": True,
                    "max_tokens": max_tokens, "temperature": temperature}
            with upstream("/chat/completions", body) as r:
                self._sse_start()
                if method != "off":
                    self.wfile.write(b"data: " + json.dumps({"sources": sources}).encode() + b"\n\n")
                while chunk := r.readline():
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except urllib.error.HTTPError as e:
            self._error(e.code, e.read().decode(errors="replace")[:500])
        except (urllib.error.URLError, OSError) as e:
            self._error(502, f"Serveur d'inférence injoignable ({BASE}). Lancez ./scripts/serve.sh ou vLLM / "
                             f"mlx_lm.server puis VLLM_BASE_URL. Détail : {e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    server = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"GalsenAI chat on http://{a.host}:{a.port}  (Ctrl-C pour arrêter)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nArrêt du serveur.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
