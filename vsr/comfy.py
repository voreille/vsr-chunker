"""Minimal ComfyUI HTTP client: upload, queue, wait, download, free.

Talks to the server over HTTP only (no shared filesystem assumed)."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import httpx

# Errors after which the ComfyUI process is unusable and must be restarted.
FATAL_PATTERNS = (
    "cuda error",
    "illegal memory access",
    "cublas_status",
    "cudnn_status",
    "device-side assert",
    "unspecified launch failure",
)


class ComfyError(RuntimeError):
    """A prompt failed (validation or execution)."""


class ComfyFatalError(ComfyError):
    """The server is broken (CUDA error) or gone; stop queueing."""


@dataclass
class OutputFile:
    filename: str
    subfolder: str
    type: str
    kind: str  # history key it came from: images / gifs / videos / ...


def _is_fatal(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in FATAL_PATTERNS)


RESTART_HINT = "ComfyUI must be restarted before queueing anything else."


class ComfyClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8188", *, timeout: float = 60.0,
                 client_id: str | None = None, transport: httpx.BaseTransport | None = None):
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id or str(uuid.uuid4())
        self.poll = 1.0  # seconds between /history polls
        self.http = httpx.Client(base_url=self.base_url, timeout=timeout, transport=transport)

    def close(self) -> None:
        self.http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # --- plumbing -------------------------------------------------------------
    def _request(self, method: str, url: str, **kw) -> httpx.Response:
        try:
            return self.http.request(method, url, **kw)
        except httpx.ConnectError as e:
            raise ComfyFatalError(f"cannot connect to ComfyUI at {self.base_url} ({e}). Is it running?") from None
        except httpx.TimeoutException as e:
            raise ComfyError(f"ComfyUI request {method} {url} timed out ({e})") from None

    def _json(self, method: str, url: str, **kw):
        r = self._request(method, url, **kw)
        if r.status_code >= 400:
            raise ComfyError(f"{method} {url} -> HTTP {r.status_code}: {r.text[:500]}")
        return r.json() if r.content else None

    # --- API ------------------------------------------------------------------
    def system_stats(self) -> dict:
        return self._json("GET", "/system_stats")

    def upload(self, path: str | Path, name: str | None = None, subfolder: str = "") -> str:
        """Upload any file (videos too) into ComfyUI's input/ folder. Returns the value to put
        in a loader widget (``subfolder/name`` or ``name``)."""
        path = Path(path)
        name = name or path.name
        with open(path, "rb") as f:
            data = {"overwrite": "true", "type": "input"}
            if subfolder:
                data["subfolder"] = subfolder
            res = self._json("POST", "/upload/image", files={"image": (name, f, "application/octet-stream")}, data=data)
        got, sub = res.get("name", name), res.get("subfolder", "")
        return f"{sub}/{got}" if sub else got

    def queue(self, prompt: dict) -> str:
        r = self._request("POST", "/prompt", json={"prompt": prompt, "client_id": self.client_id})
        if r.status_code >= 400:
            raise ComfyError("ComfyUI rejected the workflow: " + _format_prompt_error(r))
        body = r.json()
        if body.get("node_errors"):
            raise ComfyError("ComfyUI rejected the workflow: " + _format_node_errors(body["node_errors"]))
        return body["prompt_id"]

    def history(self, prompt_id: str) -> dict | None:
        res = self._json("GET", f"/history/{prompt_id}")
        return (res or {}).get(prompt_id)

    def queue_state(self) -> dict:
        return self._json("GET", "/queue")

    def wait(self, prompt_id: str, *, timeout: float = 3600.0, poll: float | None = None) -> dict:
        """Poll /history until the prompt finishes. Returns the history entry; raises on error."""
        poll = self.poll if poll is None else poll
        deadline = time.monotonic() + timeout
        last_queue_check = 0.0
        while True:
            entry = self.history(prompt_id)
            if entry is not None:
                status = entry.get("status") or {}
                if status.get("status_str") == "error":
                    raise _execution_error(prompt_id, status)
                if status.get("completed", True):
                    return entry
            now = time.monotonic()
            if now > deadline:
                raise ComfyError(f"prompt {prompt_id} did not finish within {timeout:.0f}s")
            if now - last_queue_check > 30:  # a prompt that is neither queued nor in history was lost
                last_queue_check = now
                q = self.queue_state() or {}
                ids = {item[1] for key in ("queue_running", "queue_pending") for item in q.get(key, [])}
                if prompt_id not in ids and self.history(prompt_id) is None:
                    raise ComfyFatalError(f"prompt {prompt_id} vanished from the queue (server restarted?)")
            time.sleep(poll)

    def outputs(self, entry: dict, node_id: str) -> list[OutputFile]:
        """Files produced by ``node_id``, whatever key the save node uses
        (images / gifs (VHS) / videos / ...)."""
        node_out = (entry.get("outputs") or {}).get(str(node_id))
        if not node_out:
            produced = sorted((entry.get("outputs") or {}).keys())
            raise ComfyError(f"output node {node_id} produced nothing (nodes with outputs: {produced})")
        files = []
        for kind, items in node_out.items():
            if not isinstance(items, list):
                continue
            for it in items:
                if isinstance(it, dict) and "filename" in it:
                    files.append(OutputFile(it["filename"], it.get("subfolder", ""), it.get("type", "output"), kind))
        return files

    def download(self, f: OutputFile, dst: str | Path) -> Path:
        dst = Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".part")
        params = {"filename": f.filename, "subfolder": f.subfolder, "type": f.type}
        try:
            with self.http.stream("GET", "/view", params=params) as r:
                if r.status_code >= 400:
                    raise ComfyError(f"download of {f.subfolder}/{f.filename} failed: HTTP {r.status_code}")
                with open(tmp, "wb") as out:
                    for chunk in r.iter_bytes(1 << 20):
                        out.write(chunk)
        except httpx.ConnectError as e:
            raise ComfyFatalError(f"lost connection to ComfyUI while downloading ({e})") from None
        tmp.replace(dst)
        return dst

    def free(self, unload_models: bool = True, free_memory: bool = True) -> None:
        self._json("POST", "/free", json={"unload_models": unload_models, "free_memory": free_memory})


def _execution_error(prompt_id: str, status: dict) -> ComfyError:
    parts = []
    for msg in status.get("messages") or []:
        if not (isinstance(msg, list) and len(msg) == 2):
            continue
        kind, data = msg
        if kind in ("execution_error", "execution_interrupted") and isinstance(data, dict):
            node = f"node {data.get('node_id')} ({data.get('node_type')})" if data.get("node_id") else ""
            text = data.get("exception_message") or kind
            parts.append(f"{node}: {data.get('exception_type', '')} {text}".strip())
    text = "; ".join(parts) or f"status {status}"
    if _is_fatal(text):
        return ComfyFatalError(f"prompt {prompt_id} hit a fatal GPU error: {text}\n{RESTART_HINT}")
    return ComfyError(f"prompt {prompt_id} failed: {text}")


def _format_node_errors(node_errors: dict) -> str:
    out = []
    for nid, err in node_errors.items():
        for e in err.get("errors", []):
            out.append(f"node {nid} ({err.get('class_type')}): {e.get('message')} {e.get('details', '')}".strip())
    return "; ".join(out) or str(node_errors)


def _format_prompt_error(r: httpx.Response) -> str:
    try:
        body = r.json()
    except ValueError:
        return f"HTTP {r.status_code}: {r.text[:500]}"
    msg = (body.get("error") or {}).get("message", "")
    details = (body.get("error") or {}).get("details", "")
    ne = _format_node_errors(body.get("node_errors") or {}) if body.get("node_errors") else ""
    return f"HTTP {r.status_code}: {msg} {details} {ne}".strip()
