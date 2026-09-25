"""Check bounded LLM HTTP work, exact routing, deadlines and owned cleanup.

Local fake servers control replies and stalls. No model, external network or GPU
is needed. Tests cover shared capacity, keep-alive, byte limits, failed framing,
DNS and TLS stalls, cancellation and the absence of automatic request retries.
"""

import io
import shutil
import socket
import ssl
import subprocess
import sys
import threading
import time
from collections.abc import Generator, Sequence

# Private fields are inspected only for owned-resource failure proofs.
# pyright: reportPrivateUsage=false
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Literal, cast

import pytest
from tests.llm_fixtures import send, server

from marl_battlegrounds.llm.client import (
    Client,
    ContextLimitError,
    ProviderConfigurationError,
    RequestCancelledError,
    TransportError,
)


def test_construction_and_close_make_no_network_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Construction must not resolve or connect")

    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    with Client("https://model.example/v1"):
        pass


@pytest.mark.parametrize(
    "url",
    [
        "ftp://host",
        "http://u:p@host",
        "http://host/?q=x",
        "http://host/#x",
        "http://bad host/v1",
        "http://host/bad path",
        "http://host:0/v1",
        "host",
    ],
)
def test_invalid_urls_fail_before_work(url: str) -> None:
    with pytest.raises(ValueError):
        Client(url)


@pytest.mark.parametrize(
    "settings",
    [
        {"concurrency": 0},
        {"concurrency": True},
        {"timeout": 0},
        {"timeout": float("nan")},
        {"max_response_bytes": 0},
    ],
)
def test_invalid_limits_fail_before_work(settings: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        Client("http://127.0.0.1:1", **settings)


def test_routes_authentication_and_connection_reuse() -> None:
    observed: list[tuple[str, tuple[str, int], str | None]] = []

    def reply(handler: BaseHTTPRequestHandler, payload: dict[str, Any]) -> None:
        observed.append(
            (handler.path, handler.client_address, handler.headers.get("Authorization"))
        )
        send(handler, payload)

    with server(reply) as url, Client(url, api_key="secret-value") as client:
        assert client.request("chat/completions", {"id": 1}) == {"id": 1}
        assert client.request("/tokenize", {"id": 2}) == {"id": 2}
        assert client.request("chat/completions", {"id": 3}) == {"id": 3}
    assert [row[0] for row in observed] == [
        "/v1/chat/completions",
        "/tokenize",
        "/v1/chat/completions",
    ]
    assert len({row[1] for row in observed}) == 1
    assert all(row[2] == "Bearer secret-value" for row in observed)


@pytest.mark.parametrize("ordered", [True, False])
def test_two_borrowers_share_capacity_and_preserve_results(ordered: bool) -> None:
    lock = threading.Lock()
    both_arrived = threading.Event()
    release = threading.Event()
    counts = {"active": 0, "peak": 0, "built": 0}

    def reply(handler: BaseHTTPRequestHandler, payload: dict[str, Any]) -> None:
        with lock:
            counts["active"] += 1
            counts["peak"] = max(counts["peak"], counts["active"])
            if counts["active"] == 2:
                both_arrived.set()
        assert release.wait(3)
        send(handler, payload)
        with lock:
            counts["active"] -= 1

    with server(reply) as url, Client(url, concurrency=2) as client:

        def job(value: int) -> int:
            with lock:
                counts["built"] += 1
            return cast(int, client.request("chat/completions", {"id": value})["id"])

        with ThreadPoolExecutor(max_workers=2) as borrowers:
            a = borrowers.submit(
                lambda: list(client.map(job, range(8), ordered=ordered))
            )
            b = borrowers.submit(
                lambda: list(client.map(job, range(20, 28), ordered=ordered))
            )
            try:
                assert both_arrived.wait(3)
                assert counts["built"] == 2
            finally:
                release.set()
            first, second = a.result(timeout=5), b.result(timeout=5)
            assert (first if ordered else sorted(first)) == list(range(8))
            assert (second if ordered else sorted(second)) == list(range(20, 28))
    assert counts["peak"] == 2


@pytest.mark.parametrize(
    "kind",
    ["invalid_json", "short", "large", "redirect", "nan", "infinity", "overflow"],
)
def test_bad_reply_never_retries(kind: str) -> None:
    calls: list[dict[str, Any]] = []

    def reply(handler: BaseHTTPRequestHandler, payload: dict[str, Any]) -> None:
        calls.append(payload)
        data = {
            "invalid_json": b"nope",
            "nan": b'{"id":NaN}',
            "infinity": b'{"id":Infinity}',
            "overflow": b'{"id":1e999}',
        }.get(kind, b'{"id":1}')
        handler.send_response(302 if kind == "redirect" else 200)
        handler.send_header(
            "Content-Length", str(len(data) + (3 if kind == "short" else 0))
        )
        handler.send_header("Connection", "close")
        if kind == "redirect":
            handler.send_header("Location", "http://127.0.0.1:1/elsewhere")
        handler.end_headers()
        handler.wfile.write(data)
        handler.close_connection = True

    with (
        server(reply) as url,
        Client(url, max_response_bytes=7 if kind == "large" else 100) as client,
        pytest.raises(TransportError),
    ):
        client.request("chat/completions", {"id": 1})
    assert len(calls) == 1


def test_exact_response_byte_limit_is_accepted() -> None:
    with (
        server(lambda h, p: send(h, p)) as url,
        Client(url, max_response_bytes=len(b'{"id": 1}')) as client,
    ):
        assert client.request("chat/completions", {"id": 1}) == {"id": 1}


@pytest.mark.parametrize(
    "status,code,error",
    [
        (401, "unauthorized", ProviderConfigurationError),
        (400, "context_length_exceeded", ContextLimitError),
        (503, "busy", TransportError),
    ],
)
def test_declared_server_error_categories(
    status: int, code: str, error: type[Exception]
) -> None:
    with (
        server(lambda h, _: send(h, {"error": {"code": code}}, status=status)) as url,
        Client(url) as client,
        pytest.raises(error),
    ):
        client.request("chat/completions", {})


@pytest.mark.parametrize("framing", ["length", "close", "chunked", "headers"])
def test_trickled_response_cannot_extend_total_deadline(framing: str) -> None:
    stop = threading.Event()
    entered = threading.Event()

    def reply(handler: BaseHTTPRequestHandler, _payload: dict[str, Any]) -> None:
        entered.set()
        if framing == "headers":
            handler.wfile.write(b"HTTP/1.1 200 OK\r\nX-Slow: ")
        else:
            handler.send_response(200)
            if framing == "length":
                handler.send_header("Content-Length", "1000")
            elif framing == "chunked":
                handler.send_header("Transfer-Encoding", "chunked")
            else:
                handler.send_header("Connection", "close")
            handler.end_headers()
        try:
            while not stop.wait(0.015):
                handler.wfile.write(b"1\r\nx\r\n" if framing == "chunked" else b"x")
                handler.wfile.flush()
        finally:
            handler.close_connection = True

    with server(reply) as url, Client(url, timeout=0.2) as client:
        started = time.monotonic()
        try:
            with pytest.raises(TransportError):
                client.request("chat/completions", {})
            assert entered.is_set()
            assert time.monotonic() - started < 1.5
        finally:
            stop.set()


def test_close_interrupts_active_request_and_is_idempotent() -> None:
    entered = threading.Event()
    release = threading.Event()

    def reply(handler: BaseHTTPRequestHandler, payload: dict[str, Any]) -> None:
        entered.set()
        assert release.wait(3)
        send(handler, payload)

    with server(reply) as url:
        client = Client(url, timeout=10)
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(client.request, "chat/completions", {})
            try:
                assert entered.wait(2)
                client.close(grace=1)
                with pytest.raises(RequestCancelledError):
                    result.result(timeout=1)
                client.close()
                with pytest.raises(RuntimeError, match="closed"):
                    client.request("chat/completions", {})
            finally:
                release.set()


def test_dns_is_resolved_once_and_resolution_timeout_is_reaped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import marl_battlegrounds.llm.client as module

    real_popen = subprocess.Popen
    processes: list[subprocess.Popen[str]] = []

    def start(
        args: Sequence[str], *, stdout: int, stderr: int, text: Literal[True]
    ) -> subprocess.Popen[str]:
        result = real_popen(args, stdout=stdout, stderr=stderr, text=text)
        processes.append(result)
        return result

    monkeypatch.setattr(module.subprocess, "Popen", start)
    with (
        server(lambda h, p: send(h, p)) as url,
        Client(url.replace("127.0.0.1", "localhost")) as client,
    ):
        client.request("/tokenize", {})
        client.request("chat/completions", {})
    assert len(processes) == 1
    assert processes[0].poll() == 0

    def stalled(
        _args: Sequence[str], *, stdout: int, stderr: int, text: Literal[True]
    ) -> subprocess.Popen[str]:
        return start(
            [sys.executable, "-I", "-c", "import time; time.sleep(30)"],
            stdout=stdout,
            stderr=stderr,
            text=text,
        )

    monkeypatch.setattr(module.subprocess, "Popen", stalled)
    with (
        Client("http://slow.example/v1", timeout=0.1) as client,
        pytest.raises(TransportError),
    ):
        client.request("chat/completions", {})
    assert processes[-1].poll() is not None


def test_tls_handshake_stall_obeys_deadline() -> None:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    release = threading.Event()

    def peer() -> None:
        conn, _ = listener.accept()
        with conn:
            release.wait(3)

    thread = threading.Thread(target=peer)
    thread.start()
    try:
        with Client(
            f"https://127.0.0.1:{listener.getsockname()[1]}/v1", timeout=0.2
        ) as client:
            start = time.monotonic()
            with pytest.raises(TransportError):
                client.request("chat/completions", {})
            assert time.monotonic() - start < 1.5
    finally:
        release.set()
        listener.close()
        thread.join(timeout=3)
        assert not thread.is_alive()


def test_waiting_source_does_not_hold_client_lock_and_close_uses_grace() -> None:
    entered = threading.Event()
    release = threading.Event()
    client = Client("http://127.0.0.1:1", concurrency=1)

    def source() -> Generator[int]:
        entered.set()
        assert release.wait(3)
        yield 1

    def run() -> list[int]:
        return list(client.map(lambda value: value + 1, source()))

    with ThreadPoolExecutor(max_workers=2) as pool:
        work = pool.submit(run)
        assert entered.wait(1)
        closing = pool.submit(client.close, grace=2)
        try:
            with pytest.raises(TimeoutError):
                closing.result(timeout=0.1)
        finally:
            release.set()
        closing.result(timeout=2)
        with pytest.raises(RuntimeError, match="closed"):
            work.result(timeout=2)


@pytest.mark.parametrize("ordered", [True, False])
def test_cancelled_batch_does_not_cancel_another_borrower(ordered: bool) -> None:
    sibling_entered = threading.Event()
    release = threading.Event()

    def reply(handler: BaseHTTPRequestHandler, payload: dict[str, Any]) -> None:
        index = payload["id"]
        if index == 0:
            assert sibling_entered.wait(2)
            handler.send_response(200)
            handler.send_header("Content-Length", "1")
            handler.end_headers()
            handler.wfile.write(b"x")
        elif index == 1:
            sibling_entered.set()
            assert release.wait(3)
            send(handler, payload)
        else:
            send(handler, payload)

    with server(reply) as url, Client(url, concurrency=3) as client:

        def job(index: int) -> dict[str, Any]:
            return client.request("chat/completions", {"id": index})

        with ThreadPoolExecutor(max_workers=2) as pool:
            failed = pool.submit(lambda: list(client.map(job, [0, 1], ordered=ordered)))
            other = pool.submit(job, 2)
            try:
                with pytest.raises(TransportError):
                    failed.result(timeout=3)
                assert other.result(timeout=3) == {"id": 2}
            finally:
                release.set()
        assert client.request("chat/completions", {"id": 3}) == {"id": 3}


def test_stale_keep_alive_is_not_retried() -> None:
    observed: list[int] = []
    dropped = threading.Event()

    def reply(handler: BaseHTTPRequestHandler, payload: dict[str, Any]) -> None:
        observed.append(payload["id"])
        send(handler, payload)
        handler.connection.shutdown(socket.SHUT_RDWR)
        handler.close_connection = True
        dropped.set()

    with server(reply) as url, Client(url) as client:
        assert client.request("chat/completions", {"id": 1}) == {"id": 1}
        assert dropped.wait(1)
        with pytest.raises(TransportError):
            client.request("chat/completions", {"id": 2})
    assert observed == [1]


def test_connect_cancellation_closes_unpublished_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from marl_battlegrounds.llm.client import _Call

    call = _Call(time.monotonic() + 10, None)
    closed: list[bool] = []

    class Socket:
        def settimeout(self, _value: float) -> None:
            pass

        def setsockopt(self, *_args: object) -> None:
            pass

        def connect(self, _address: object) -> None:
            call.cancelled = True

        def close(self) -> None:
            closed.append(True)

    def create(*_args: object) -> Socket:
        return Socket()

    monkeypatch.setattr(socket, "socket", create)
    with Client("http://127.0.0.1:1") as client, pytest.raises(RequestCancelledError):
        client._connect(call)
    assert closed == [True]


def test_stalled_connect_is_interrupted_by_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interrupted = threading.Event()
    closed: list[bool] = []

    class Socket:
        def settimeout(self, _value: float) -> None:
            pass

        def setsockopt(self, *_args: object) -> None:
            pass

        def connect(self, _address: object) -> None:
            assert interrupted.wait(2)

        def shutdown(self, _how: int) -> None:
            interrupted.set()

        def close(self) -> None:
            closed.append(True)

    def create(*_args: object) -> Socket:
        return Socket()

    monkeypatch.setattr(socket, "socket", create)
    with (
        Client("http://127.0.0.1:1", timeout=0.1) as client,
        pytest.raises(TransportError),
    ):
        client.request("chat/completions", {})
    assert interrupted.is_set()
    assert closed == [True]


def test_tls_checks_certificate_and_preserves_original_hostname(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    openssl = shutil.which("openssl")
    if openssl is None:
        pytest.skip("TLS fixture needs the openssl test command")
    certificate, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    subprocess.run(
        [
            openssl,
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(certificate),
            "-days",
            "1",
            "-subj",
            "/CN=localhost",
            "-addext",
            "subjectAltName=DNS:localhost",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=10,
    )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certificate, key)
    hosts: list[str | None] = []

    def reply(handler: BaseHTTPRequestHandler, payload: dict[str, Any]) -> None:
        hosts.append(handler.headers.get("Host"))
        send(handler, payload)

    def trusted() -> ssl.SSLContext:
        result = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        result.load_verify_locations(certificate)
        return result

    with server(reply, tls_context=context) as url:
        named = url.replace("127.0.0.1", "localhost")
        with Client(named) as client, pytest.raises(TransportError):
            client.request("chat/completions", {})
        monkeypatch.setattr(ssl, "create_default_context", trusted)
        with Client(named) as client:
            assert client.request("chat/completions", {"id": 1}) == {"id": 1}
        with Client(url) as client, pytest.raises(TransportError):
            client.request("chat/completions", {})
    assert len(hosts) == 1
    assert hosts[0] is not None and hosts[0].startswith("localhost:")


def test_resolver_that_survives_kill_remains_owned_until_reaped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Resolver:
        def __init__(self) -> None:
            self.stdout = io.StringIO()
            self.returncode: int | None = None
            self.kills = 0

        def communicate(self, *, timeout: float) -> tuple[str, None]:
            raise subprocess.TimeoutExpired("test resolver", timeout)

        def poll(self) -> int | None:
            return self.returncode

        def kill(self) -> None:
            self.kills += 1

        def wait(self, *, timeout: float) -> int:
            if self.returncode is None:
                raise subprocess.TimeoutExpired("test resolver", timeout)
            return self.returncode

    resolver = Resolver()
    starts: list[bool] = []

    def start(*_args: object, **_kwargs: object) -> Resolver:
        starts.append(True)
        return resolver

    monkeypatch.setattr(subprocess, "Popen", start)
    client = Client("http://test.invalid:8000/v1", timeout=0.1)
    try:
        with pytest.raises(RuntimeError, match="process remains owned"):
            client.request("chat/completions", {})
        assert client._resolver is resolver
        assert client._closed
        with pytest.raises(RuntimeError, match="closed"):
            client.request("chat/completions", {})
        assert len(starts) == 1
        with pytest.raises(RuntimeError, match="remains owned after cleanup"):
            client.close(grace=0)
        assert client._resolver is resolver
        assert not resolver.stdout.closed
        assert resolver.kills == 2
    finally:
        resolver.returncode = -9
        client.close()
    assert client._resolver is None
    assert resolver.stdout.closed


def test_unordered_map_keeps_slots_busy_behind_a_slow_first_job() -> None:
    release_first = threading.Event()
    later_started = threading.Event()

    def job(index: int) -> int:
        if index == 0:
            assert release_first.wait(2)
        if index == 4:
            later_started.set()
        return index

    with Client("http://127.0.0.1:1", concurrency=4) as client:
        iterator = client.map(job, range(12), ordered=False)
        try:
            first = next(iterator)
            assert first != 0
            second = next(iterator)
            assert second != 0
            assert later_started.wait(1)
        finally:
            release_first.set()
        assert sorted([first, second, *iterator]) == list(range(12))


def test_unordered_map_reports_later_failure_without_waiting_for_first() -> None:
    release_first = threading.Event()
    entered = threading.Event()

    def job(index: int) -> int:
        if index == 0:
            entered.set()
            assert release_first.wait(2)
        else:
            assert entered.wait(1)
            raise ValueError("Later job failed")
        return index

    with Client("http://127.0.0.1:1", concurrency=2) as client:
        try:
            with pytest.raises(ValueError, match="Later job failed"):
                next(client.map(job, range(2), ordered=False))
            assert not release_first.is_set()
        finally:
            release_first.set()
