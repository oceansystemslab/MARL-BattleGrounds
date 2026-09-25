"""Send bounded JSON requests to an independently managed model server.

Client owns HTTP connections, a lazy worker pool and a deadline watcher. It
never starts or stops the model server and never retries a submitted request.
The System adapter uses map to keep prompt construction and waiting work bounded.
"""

from __future__ import annotations

# The connection and client share private transport state in this one module.
# pyright: reportPrivateUsage=false
import http.client
import ipaddress
import json
import math
import socket
import ssl
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import (
    FIRST_COMPLETED,
    CancelledError,
    Future,
    ThreadPoolExecutor,
    wait,
)
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Self, cast
from urllib.parse import urlsplit


def _finite_json_number(token: str) -> float:
    """Reject nonfinite JSON numbers, including a finite-looking overflow token."""
    number = float(token)
    if not math.isfinite(number):
        raise ValueError("Server JSON numbers must be finite")
    return number


class TransportError(RuntimeError):
    """A request failed to reach the server or receive a complete JSON reply."""


class RequestCancelledError(RuntimeError):
    """Local execution cancelled a request; game fallback must not swallow it."""


class ContextLimitError(ValueError):
    """The complete current request and reserved reply exceed the model context."""


class ProviderConfigurationError(ValueError):
    """The server rejected the configured model, credentials or request options."""


@dataclass(eq=False)
class _Call:
    """Keep one request's absolute deadline and socket until publication ends."""

    deadline: float
    group: threading.Event | None
    sock: socket.socket | None = None
    cancelled: bool = False
    expired: bool = False


class _Connection(http.client.HTTPConnection):
    """Reuse HTTP framing while the Client owns connection and TLS deadlines."""

    def __init__(self, owner: Client) -> None:
        """Keep the original host for HTTP headers and TLS certificate checks."""
        super().__init__(owner._host, owner._port)
        self.owner = owner
        self.call: _Call | None = None

    def connect(self) -> None:
        """Open exactly one checked connection without resending an HTTP request."""
        assert self.call is not None
        self.sock = self.owner._connect(self.call)


class Client:
    """Send JSON HTTP requests with bounded concurrency and no automatic retry.

    Parameters
    ----------
    server_url : str
        HTTP or HTTPS base URL, usually ``http://127.0.0.1:8000/v1``. Embedded
        credentials, queries and fragments are refused. Construction makes no
        network calls. Hostnames are resolved once on first use in an owned,
        time-limited child process; numeric addresses need no child process.
    concurrency : int
        Maximum simultaneous HTTP requests and worker jobs, default 16. A map
        call prepares at most this many jobs ahead. All borrowers share limits.
    timeout : float
        Positive complete HTTP request limit in seconds, default 60. The clock
        covers admission, DNS, connection, TLS, response reading and decoding.
        A peer sending occasional bytes cannot renew this deadline.
    api_key : str | None
        Optional bearer credential. It is sent only in the Authorization header
        and is never included in request bodies or the object's representation.
    max_response_bytes : int
        Positive decoded HTTP-body byte limit, default 2 MiB. This also bounds
        tokenizer replies, which may contain a list of token IDs.

    Notes
    -----
    Use as a context manager, or call close. Callers own explicitly supplied
    clients. The runner owns clients that it creates. Transport cancellation
    cannot promise that a server stopped generation. Custom Python work passed
    to map must cooperate: a thread cannot forcibly stop arbitrary user code.
    Redirects and failed keep-alive connections are errors, never hidden retries.
    Socket and child-process cleanup is bounded; surviving user work stays fenced
    in the closed client and raises an explicit cleanup error.
    """

    def __init__(
        self,
        server_url: str,
        *,
        concurrency: int = 16,
        timeout: float = 60.0,
        api_key: str | None = None,
        max_response_bytes: int = 2 * 1024 * 1024,
    ) -> None:
        """Validate local settings without starting workers or making requests."""
        parsed = urlsplit(server_url)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "Server URL must be HTTP or HTTPS without credentials or query"
            )
        for name, value in (
            ("concurrency", concurrency),
            ("max_response_bytes", max_response_bytes),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be a finite positive number of seconds")
        if api_key is not None and (
            not isinstance(cast(object, api_key), str)
            or any(c in api_key for c in "\r\n")
        ):
            raise ValueError("api_key must be text without line breaks")
        self.server_url = server_url.rstrip("/")
        self.concurrency = concurrency
        self.timeout = float(timeout)
        self.max_response_bytes = max_response_bytes
        self._host = parsed.hostname
        self._port = parsed.port or (443 if parsed.scheme == "https" else 80)
        self._tls = parsed.scheme == "https"
        self._base_path = parsed.path.rstrip("/")
        if any(
            ord(char) <= 32 or ord(char) == 127 for char in parsed.netloc + parsed.path
        ):
            raise ValueError("Server URL must not contain spaces or control characters")
        if not self._base_path.isascii():
            raise ValueError("Server URL path must use ASCII percent-encoded text")
        if parsed.port == 0:
            raise ValueError("Server port must be in 1..65535")
        self._api_key = api_key
        self._condition = threading.Condition()
        self._local = threading.local()
        self._calls: set[_Call] = set()
        self._connections: set[_Connection] = set()
        self._idle: deque[_Connection] = deque()
        self._futures: set[Future[Any]] = set()
        self._executor: ThreadPoolExecutor | None = None
        self._watcher: threading.Thread | None = None
        self._closed = False
        self._jobs = 0
        self._addresses: list[tuple[int, int, int, tuple[Any, ...]]] | None = None
        self._resolve_lock = threading.Lock()
        self._resolver: subprocess.Popen[str] | None = None
        self._ssl_context: ssl.SSLContext | None = None

    def __enter__(self) -> Self:
        """Return this still-open client; no network work is started."""
        with self._condition:
            self._check_open()
        return self

    def __exit__(self, *args: object) -> None:
        """Close owned connections and workers; never contact a server stop route."""
        self.close()

    def _check_open(self) -> None:
        """Reject reuse after close, including late queued worker calls."""
        if self._closed:
            raise RuntimeError("LLM client is closed")

    def _remaining(self, call: _Call) -> float:
        """Reject cancelled or expired work before it can publish a result."""
        remaining = call.deadline - time.monotonic()
        if (
            call.cancelled
            or self._closed
            or (call.group is not None and call.group.is_set())
        ):
            raise RequestCancelledError("LLM request was cancelled locally")
        if call.expired or remaining <= 0:
            raise TransportError("LLM request exceeded its deadline")
        return remaining

    @staticmethod
    def _shutdown(sock: socket.socket | None) -> None:
        """Wake a blocked socket operation without taking its buffered-reader lock."""
        if sock is not None:
            with suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)

    def _watch(self) -> None:
        """Abort whole requests at their deadlines, including trickled responses."""
        with self._condition:
            while not self._closed or self._calls:
                now = time.monotonic()
                for call in self._calls:
                    if self._closed:
                        call.cancelled = True
                        self._shutdown(call.sock)
                    elif now >= call.deadline and not call.expired:
                        call.expired = True
                        self._shutdown(call.sock)
                deadlines = [
                    c.deadline - now
                    for c in self._calls
                    if not c.cancelled and not c.expired
                ]
                self._condition.wait(max(0.001, min(deadlines, default=0.1)))

    def _start_watcher(self) -> None:
        """Create the one client watcher under the condition lock, on first use."""
        if self._watcher is None:
            self._watcher = threading.Thread(
                target=self._watch, name="marl-llm-deadlines", daemon=True
            )
            self._watcher.start()

    def _resolve(self, call: _Call) -> list[tuple[int, int, int, tuple[Any, ...]]]:
        """Cache bounded address resolution, keeping DNS outside blocked threads."""
        if not self._resolve_lock.acquire(timeout=self._remaining(call)):
            raise TransportError("LLM host lookup exceeded the request deadline")
        try:
            self._remaining(call)
            if self._addresses is not None:
                return self._addresses
            try:
                address = ipaddress.ip_address(self._host)
            except ValueError:
                address = None
            if address is not None:
                family = socket.AF_INET6 if address.version == 6 else socket.AF_INET
                target = (
                    (self._host, self._port, 0, 0)
                    if address.version == 6
                    else (self._host, self._port)
                )
                self._addresses = [
                    (family, socket.SOCK_STREAM, socket.IPPROTO_TCP, target)
                ]
                return self._addresses
            code = (
                "import json,socket,sys; "
                "print(json.dumps(socket.getaddrinfo(sys.argv[1],int(sys.argv[2]),"
                "type=socket.SOCK_STREAM,proto=socket.IPPROTO_TCP)[:16]))"
            )
            with self._condition:
                self._remaining(call)
                process = subprocess.Popen(
                    [sys.executable, "-I", "-c", code, self._host, str(self._port)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
                self._resolver = process
            try:
                output, _ = process.communicate(timeout=self._remaining(call))
                self._remaining(call)
                if process.returncode:
                    raise TransportError("LLM server hostname could not be resolved")
                rows = json.loads(output)
                self._addresses = [
                    (int(a), int(b), int(c), tuple(e)) for a, b, c, _, e in rows
                ]
                if not self._addresses:
                    raise TransportError("LLM server hostname has no usable address")
                return self._addresses
            except subprocess.TimeoutExpired as exc:
                raise TransportError(
                    "LLM host lookup exceeded the request deadline"
                ) from exc
            finally:
                try:
                    self._stop_resolver(process)
                except RuntimeError:
                    with self._condition:
                        self._closed = True
                        self._condition.notify_all()
                    raise
                with self._condition:
                    self._resolver = None
        finally:
            self._resolve_lock.release()

    @staticmethod
    def _stop_resolver(process: subprocess.Popen[str]) -> None:
        """Kill and reap an owned resolver using bounded waits only."""
        if process.poll() is None:
            process.kill()
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "LLM resolver did not stop; its process remains owned"
            ) from exc
        if process.stdout is not None:
            process.stdout.close()

    def _connect(self, call: _Call) -> socket.socket:
        """Connect within one deadline, retaining hostname checks for TLS."""
        addresses = self._resolve(call)
        last_error: OSError | None = None
        for family, kind, protocol, target in addresses:
            sock = socket.socket(family, kind, protocol)
            try:
                with self._condition:
                    sock.settimeout(self._remaining(call))
                    call.sock = sock
                sock.connect(target)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                if self._tls:
                    with self._condition:
                        self._remaining(call)
                        if self._ssl_context is None:
                            self._ssl_context = ssl.create_default_context()
                        sock = self._ssl_context.wrap_socket(
                            sock,
                            server_hostname=self._host,
                            do_handshake_on_connect=False,
                        )
                        call.sock = sock
                    sock.settimeout(self._remaining(call))
                    sock.do_handshake()
                self._remaining(call)
                return sock
            except OSError as exc:
                last_error = exc
                sock.close()
                self._remaining(call)
            except BaseException:
                sock.close()
                raise
        raise TransportError("Could not connect to the LLM server") from last_error

    def request(self, path: str, payload: Mapping[str, object]) -> dict[str, Any]:
        """POST one JSON object and return a bounded JSON object response.

        path is relative to server_url unless it starts with ``/``; for example
        ``chat/completions`` and ``/tokenize``. It cannot select another origin.
        payload must contain finite JSON values. Calls share the client limit.
        HTTP context errors stop with ContextLimitError; other 4xx errors use
        ProviderConfigurationError. Network, 5xx and malformed bodies raise
        TransportError. No request is retried. Authentication is never returned.
        """
        if (
            not isinstance(cast(object, path), str)
            or not path
            or ":" in path
            or "?" in path
            or "#" in path
            or not path.isascii()
            or any(ord(char) <= 32 or ord(char) == 127 for char in path)
        ):
            raise ValueError("Request path must name a route on this server")
        route = path if path.startswith("/") else f"{self._base_path}/{path}"
        call = _Call(
            time.monotonic() + self.timeout, getattr(self._local, "group", None)
        )
        body = json.dumps(
            dict(payload), allow_nan=False, separators=(",", ":")
        ).encode()
        connection: _Connection | None = None
        response = None
        try:
            with self._condition:
                self._check_open()
                while len(self._calls) >= self.concurrency:
                    self._condition.wait(self._remaining(call))
                self._remaining(call)
                self._calls.add(call)
                self._start_watcher()
                self._condition.notify_all()
            with self._condition:
                if self._idle:
                    connection = self._idle.popleft()
                else:
                    connection = _Connection(self)
                    self._connections.add(connection)
            connection.call = call
            with self._condition:
                self._remaining(call)
                if connection.sock is not None:
                    call.sock = connection.sock
                    connection.sock.settimeout(self._remaining(call))
            headers = {"Content-Type": "application/json", "Accept": "application/json"}
            if self._api_key:
                headers["Authorization"] = "Bearer " + self._api_key
            connection.request("POST", route, body=body, headers=headers)
            response = connection.getresponse()
            declared = response.length
            if declared is not None and declared > self.max_response_bytes:
                raise TransportError("LLM response exceeds max_response_bytes")
            data = response.read(self.max_response_bytes + 1)
            if len(data) > self.max_response_bytes:
                raise TransportError("LLM response exceeds max_response_bytes")
            if declared is not None and len(data) != declared:
                raise TransportError("LLM response ended before its declared length")
            self._remaining(call)
            if response.status >= 300:
                if response.status in (400, 413, 422):
                    try:
                        error = json.loads(data).get("error", {})
                    except ValueError, AttributeError:
                        error = {}
                    code = (
                        cast(dict[str, object], error).get("code")
                        if isinstance(error, dict)
                        else None
                    )
                    if (
                        code
                        in ("context_length_exceeded", "max_context_length_exceeded")
                        or response.status == 413
                    ):
                        raise ContextLimitError(
                            "LLM server rejected the request context length"
                        )
                if 400 <= response.status < 500:
                    raise ProviderConfigurationError(
                        f"LLM server rejected the request (HTTP {response.status})"
                    )
                raise TransportError(f"LLM server returned HTTP {response.status}")
            decoded = json.loads(
                data,
                parse_float=_finite_json_number,
                parse_constant=_finite_json_number,
            )
            if not isinstance(decoded, dict):
                raise TransportError("LLM server must return one JSON object")
            with self._condition:
                self._remaining(call)
            return cast(dict[str, Any], decoded)
        except (OSError, http.client.HTTPException, ValueError) as exc:
            if connection is not None:
                connection.close()
            if isinstance(exc, (ContextLimitError, ProviderConfigurationError)):
                raise
            self._remaining(call)
            raise TransportError("LLM request failed or returned invalid JSON") from exc
        except TransportError:
            if connection is not None:
                connection.close()
            self._remaining(call)
            raise
        except BaseException:
            if connection is not None:
                connection.close()
            raise
        finally:
            if response is not None:
                response.close()
            with self._condition:
                if connection is not None:
                    connection.call = None
                    self._idle.append(connection)
                self._calls.discard(call)
                self._condition.notify_all()

    def map[T, R](
        self, function: Callable[[T], R], items: Iterable[T], *, ordered: bool = True
    ) -> Iterator[R]:
        """Run bounded jobs without preparing every prompt in advance.

        Each worker calls function with one item. ordered defaults to True and
        yields results in input order. Set False when results carry their own
        actor identifiers: a finished result then frees its slot immediately,
        without waiting for an earlier slow actor. The team decision still waits
        for all its actors. Neither mode promises fair turns between borrowers.
        Jobs may call request several times, for example to fit history before
        generation. At most concurrency
        jobs are admitted across all map calls. A failing or closed iterator
        cancels its queued work and interrupts only its own HTTP requests.
        Other borrowers keep their independent requests. Arbitrary custom Python
        work must return cooperatively; it cannot be killed by cancellation.
        """
        group = threading.Event()
        pending: deque[Future[R]] = deque()
        source = iter(items)
        exhausted = False

        def run(item: T) -> R:
            """Bind requests to this iterator's cancellation group in its worker."""
            self._local.group = group
            try:
                if group.is_set():
                    raise RequestCancelledError("LLM batch was cancelled")
                return function(item)
            finally:
                self._local.group = None

        def finished(future: Future[R]) -> None:
            """Release a job permit only after its work actually ends."""
            with self._condition:
                self._jobs -= 1
                self._futures.discard(future)
                self._condition.notify_all()

        try:
            while pending or not exhausted:
                while not exhausted and len(pending) < self.concurrency:
                    with self._condition:
                        self._check_open()
                        while self._jobs >= self.concurrency:
                            self._condition.wait()
                            self._check_open()
                        self._jobs += 1
                    try:
                        item = next(source)
                        with self._condition:
                            self._check_open()
                            if self._executor is None:
                                self._executor = ThreadPoolExecutor(
                                    max_workers=self.concurrency,
                                    thread_name_prefix="marl-llm",
                                )
                            future = self._executor.submit(run, item)
                            self._futures.add(future)
                            future.add_done_callback(finished)
                    except BaseException as exc:
                        with self._condition:
                            self._jobs -= 1
                            self._condition.notify_all()
                        if isinstance(exc, StopIteration):
                            exhausted = True
                            break
                        raise
                    pending.append(future)
                if pending:
                    if ordered:
                        finished_future = pending.popleft()
                    else:
                        completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                        # Surface a finished failure before yielding another success.
                        finished_future = next(
                            (f for f in completed if f.cancelled() or f.exception()),
                            next(iter(completed)),
                        )
                        pending.remove(finished_future)
                    try:
                        result = finished_future.result()
                    except CancelledError as exc:
                        raise RequestCancelledError("LLM job was cancelled") from exc
                    with self._condition:
                        self._check_open()
                        if group.is_set():
                            raise RequestCancelledError("LLM batch was cancelled")
                    yield result
        finally:
            group.set()
            with self._condition:
                for call in self._calls:
                    if call.group is group:
                        call.cancelled = True
                        self._shutdown(call.sock)
                for future in pending:
                    future.cancel()
                self._condition.notify_all()

    def close(self, *, grace: float = 2.0) -> None:
        """Cancel owned work and close resources within a finite cleanup grace.

        grace is a finite nonnegative number of seconds, default 2. Running
        transport calls are interrupted. Workers still executing custom Python
        code remain owned and fenced; RuntimeError reports that they survived.
        Closing twice is safe. A supplied independent server is never stopped.
        """
        if not math.isfinite(grace) or grace < 0:
            raise ValueError("grace must be a finite nonnegative number of seconds")
        deadline = time.monotonic() + grace
        with self._condition:
            self._closed = True
            for call in self._calls:
                call.cancelled = True
                self._shutdown(call.sock)
            for connection in self._connections:
                self._shutdown(connection.sock)
            if self._resolver is not None and self._resolver.poll() is None:
                self._resolver.kill()
            if self._executor is not None:
                self._executor.shutdown(wait=False, cancel_futures=True)
            futures = tuple(self._futures)
            self._condition.notify_all()
        if futures:
            wait(futures, timeout=max(0.0, deadline - time.monotonic()))
        with self._condition:
            while (self._calls or self._jobs) and time.monotonic() < deadline:
                self._condition.wait(max(0.0, deadline - time.monotonic()))
            if self._jobs or self._calls:
                raise RuntimeError(
                    "LLM client closed, but running user work has not stopped"
                )
            for connection in self._connections:
                connection.close()
            self._connections.clear()
            self._idle.clear()
            resolver = self._resolver
        if resolver is not None:
            try:
                resolver.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    "LLM resolver remains owned after cleanup grace"
                ) from exc
            if resolver.stdout is not None:
                resolver.stdout.close()
            with self._condition:
                self._resolver = None
        if self._watcher is not None:
            self._watcher.join(timeout=max(0.0, deadline - time.monotonic()))
