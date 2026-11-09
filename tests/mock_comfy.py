"""A tiny fake ComfyUI server: upload, prompt, history, queue, view, free.

"Executing" a prompt copies the file named in the input node to the output (identity model)."""

from __future__ import annotations

import email.parser
import email.policy
import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


class MockComfy:
    def __init__(self, input_node: str, input_field: str, output_node: str, prefix_field: str = "filename_prefix"):
        self.input_node, self.input_field = input_node, input_field
        self.output_node, self.prefix_field = output_node, prefix_field
        self.files: dict[tuple[str, str, str], bytes] = {}  # (type, subfolder, name) -> data
        self.prompts: list[dict] = []
        self.history: dict[str, dict] = {}
        self.frees: list[dict] = []
        self.running: list[str] = []  # prompt ids reported as running by /queue
        self.pending: list[str] = []
        self.posts: list[tuple[str, dict]] = []  # (path, body) of /queue and /interrupt calls
        self.mode = "ok"  # ok | cuda | oom | reject | nothing
        self.output_key = "images"  # images | gifs | videos
        self.output_ext = ".mp4"
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()

    def _execute(self, prompt: dict) -> str:
        pid = str(uuid.uuid4())
        self.prompts.append(prompt)
        value = prompt[self.input_node]["inputs"][self.input_field]
        sub, _, name = value.rpartition("/")
        prefix = prompt[self.output_node]["inputs"][self.prefix_field]
        osub, _, oname = prefix.rpartition("/")
        status = {"status_str": "success", "completed": True, "messages": [["execution_start", {}], ["execution_success", {}]]}
        outputs = {}
        if self.mode in ("cuda", "oom"):
            msg = ("CUDA error: an illegal memory access was encountered" if self.mode == "cuda"
                   else "Allocation on device would exceed allowed memory")
            status = {"status_str": "error", "completed": False, "messages": [
                ["execution_start", {}],
                ["execution_error", {"node_id": "66:54", "node_type": "KSampler",
                                     "exception_type": "torch.AcceleratorError", "exception_message": msg}]]}
        elif self.mode != "nothing":
            fname = f"{oname}_00001_{self.output_ext}"
            self.files[("output", osub, fname)] = self.files[("input", sub, name)]
            outputs = {self.output_node: {self.output_key: [{"filename": fname, "subfolder": osub, "type": "output"}],
                                          "animated": [True]}}
        self.history[pid] = {"prompt": [0, pid, prompt, {}, []], "outputs": outputs, "status": status, "meta": {}}
        return pid

    def _handler(self):
        mock = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body, ctype="application/json"):
                data = json.dumps(body).encode() if ctype == "application/json" else body
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _body(self):
                return self.rfile.read(int(self.headers.get("Content-Length", 0)))

            def do_GET(self):
                u = urlparse(self.path)
                q = {k: v[0] for k, v in parse_qs(u.query).items()}
                if u.path == "/system_stats":
                    return self._send(200, {"system": {"comfyui_version": "mock"},
                                            "devices": [{"name": "mock GPU", "vram_free": 2**34}]})
                if u.path.startswith("/history/"):
                    pid = u.path.split("/")[-1]
                    return self._send(200, {pid: mock.history[pid]} if pid in mock.history else {})
                if u.path == "/queue":
                    return self._send(200, {"queue_running": [[0, p, {}, {}, []] for p in mock.running],
                                            "queue_pending": [[1, p, {}, {}, []] for p in mock.pending]})
                if u.path == "/view":
                    key = (q.get("type", "output"), q.get("subfolder", ""), q["filename"])
                    if key not in mock.files:
                        return self._send(404, {"error": "not found"})
                    return self._send(200, mock.files[key], "application/octet-stream")
                self._send(404, {})

            def do_POST(self):
                u = urlparse(self.path)
                body = self._body()
                if u.path == "/upload/image":
                    msg = email.parser.BytesParser(policy=email.policy.default).parsebytes(
                        b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\n\r\n" + body)
                    fields, fdata, fname = {}, None, None
                    for part in msg.iter_parts():
                        name = part.get_param("name", header="content-disposition")
                        if name == "image":
                            fdata, fname = part.get_payload(decode=True), part.get_filename()
                        else:
                            fields[name] = part.get_content().strip()
                    sub = fields.get("subfolder", "")
                    mock.files[("input", sub, fname)] = fdata
                    return self._send(200, {"name": fname, "subfolder": sub, "type": "input"})
                if u.path == "/prompt":
                    prompt = json.loads(body)["prompt"]
                    if mock.mode == "reject":
                        return self._send(400, {
                            "error": {"type": "prompt_outputs_failed_validation", "message": "Prompt outputs failed validation", "details": ""},
                            "node_errors": {mock.input_node: {"class_type": "LoadVideo", "errors": [
                                {"message": "Value not in list", "details": "file: 'x.mp4' not in [...]"}]}}})
                    return self._send(200, {"prompt_id": mock._execute(prompt), "number": 1, "node_errors": {}})
                if u.path in ("/queue", "/interrupt"):
                    mock.posts.append((u.path, json.loads(body)))
                    return self._send(200, {})
                if u.path == "/free":
                    mock.frees.append(json.loads(body))
                    return self._send(200, {})
                self._send(404, {})

        return H
