"""Provide local HTTP servers for LLM transport and System contract tests.

These helpers run only loopback HTTP. They preserve request bodies and let each
case control timing, HTTP framing and model replies without weights or a GPU.
"""

import json
import socket
import ssl
import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager, suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


@contextmanager
def server(
    reply: Callable[[BaseHTTPRequestHandler, dict[str, Any]], None],
    *,
    tls_context: ssl.SSLContext | None = None,
) -> Generator[str]:

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self) -> None:
            super().setup()
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        def log_message(self, format: str, *args: object) -> None:
            pass

        def do_POST(self) -> None:
            payload = json.loads(
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
            )
            with suppress(BrokenPipeError, ConnectionResetError):
                reply(self, payload)

    class Server(ThreadingHTTPServer):
        # The default backlog of five drops concurrent fake-model connections.
        request_queue_size = 128

    httpd = Server(("127.0.0.1", 0), Handler)
    if tls_context is not None:
        httpd.socket = tls_context.wrap_socket(httpd.socket, server_side=True)
    worker = threading.Thread(target=lambda: httpd.serve_forever(poll_interval=0.01))
    worker.start()
    try:
        scheme = "https" if tls_context is not None else "http"
        yield f"{scheme}://127.0.0.1:{httpd.server_port}/v1"
    finally:
        httpd.shutdown()
        httpd.server_close()
        worker.join(timeout=2)
        assert not worker.is_alive()


def send(handler: BaseHTTPRequestHandler, value: object, *, status: int = 200) -> None:
    data = json.dumps(value).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)
    handler.wfile.flush()


class FakeModel:
    def __init__(
        self,
        reply: str = '{"move":"stay","combat":"no_combat"}',
        *,
        limit: int = 100_000,
    ) -> None:
        self.text = reply
        self.limit = limit
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.lock = threading.Lock()

    def serve(self, handler: BaseHTTPRequestHandler, payload: dict[str, Any]) -> None:
        prompt = payload["messages"][0]["content"]
        with self.lock:
            self.calls.append((handler.path, payload))
        if handler.path == "/tokenize":
            send(handler, {"count": len(prompt), "max_model_len": self.limit})
        else:
            send(
                handler,
                {
                    "choices": [
                        {"finish_reason": "stop", "message": {"content": self.text}}
                    ],
                    "usage": {"prompt_tokens": len(prompt)},
                },
            )

    def generations(self) -> list[dict[str, Any]]:
        return [
            value for path, value in self.calls if path.endswith("chat/completions")
        ]
