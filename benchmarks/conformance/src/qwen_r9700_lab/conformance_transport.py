"""Owned qualification processes and strict OpenAI/SSE transport.

No GPU imports and no connection to a pre-existing inference endpoint. The
controller requires a fresh private workspace and an identity handshake from
its own child before sending a prompt. Logs and failed attempts are retained.
"""

from __future__ import annotations

import codecs
import contextlib
import json
import math
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from qwen_r9700_lab.diagnostic_contract import DiagnosticError, write_private


class ProtocolError(DiagnosticError):
    pass


class BackendResponseError(ProtocolError):
    """An explicit server error; never a timeout, disconnect or client failure."""

    def __init__(self, message, *, http_status=None):
        super().__init__(message)
        self.http_status = http_status


def check_response_event(event):
    if not isinstance(event, dict):
        raise ProtocolError("backend response is not a JSON object")
    if "error" in event:
        error = event["error"]
        if not isinstance(error, dict) or not isinstance(error.get("message"), str):
            raise ProtocolError("malformed backend error object")
        raise BackendResponseError("backend returned an explicit error")


class SSEDecoder:
    """Incremental UTF-8 and SSE framing, including CRLF split across reads."""

    def __init__(self):
        self.decoder = codecs.getincrementaldecoder("utf-8")("strict")
        self.buffer = ""
        self.data = []
        self.done = False

    def feed(self, raw: bytes, *, final=False):
        try:
            self.buffer += self.decoder.decode(raw, final=final)
        except UnicodeDecodeError as error:
            raise ProtocolError("invalid UTF-8 in stream") from error
        events = []
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            line = line.removesuffix("\r")
            if not line:
                if self.data:
                    value = "\n".join(self.data)
                    self.data.clear()
                    if self.done:
                        raise ProtocolError("data after stream completion")
                    if value == "[DONE]":
                        self.done = True
                    else:
                        try:
                            event = json.loads(value)
                        except ValueError as error:
                            raise ProtocolError("invalid JSON in stream") from error
                        check_response_event(event)
                        events.append(event)
            elif line.startswith("data:"):
                self.data.append(line[5:].removeprefix(" "))
            elif line.startswith((":", "event:", "id:", "retry:")):
                continue
            else:
                raise ProtocolError("unsupported stream field")
        if final and (self.buffer or self.data or not self.done):
            raise ProtocolError("stream ended without a complete DONE boundary")
        return events


class Completion:
    """Assemble one real response without inventing a finish or a tool call."""

    def __init__(self):
        self.content = ""
        self.reasoning = ""
        self.token_ids = []
        self.prompt_token_ids = None
        self.tools = {}
        self.finish = None
        self.usage = None
        self.events = 0

    def _accept_prompt_ids(self, prompt_ids):
        if prompt_ids is not None:
            if not isinstance(prompt_ids, list) or any(
                type(token) is not int or token < 0 for token in prompt_ids
            ):
                raise ProtocolError("invalid prompt token IDs")
            if self.prompt_token_ids is not None and self.prompt_token_ids != prompt_ids:
                raise ProtocolError("prompt token IDs changed during streaming")
            self.prompt_token_ids = list(prompt_ids)

    def accept(self, event):
        check_response_event(event)
        self._accept_prompt_ids(event.get("prompt_token_ids"))
        if event.get("usage") is not None:
            self.usage = event["usage"]
        choices = event.get("choices", [])
        if len(choices) > 1:
            raise ProtocolError("qualification expects exactly one choice")
        if not choices:
            return
        choice = choices[0]
        if choice.get("index", 0) != 0:
            raise ProtocolError("unexpected choice index")
        # The chat API places prompt IDs at response level; the completions API
        # places them on its choice. Preserve both without allowing replacement.
        self._accept_prompt_ids(choice.get("prompt_token_ids"))
        delta = choice.get("delta", choice.get("message", {}))
        if self.finish is not None:
            raise ProtocolError("choice data after its finish boundary")
        self.events += 1
        self.content += delta.get("content") or choice.get("text") or ""
        self.reasoning += delta.get("reasoning") or delta.get("reasoning_content") or ""
        ids = choice.get("token_ids")
        if ids is not None:
            if not isinstance(ids, list) or any(type(t) is not int or t < 0 for t in ids):
                raise ProtocolError("invalid output token IDs")
            self.token_ids.extend(ids)
        for call in delta.get("tool_calls") or []:
            index = call.get("index", len(self.tools) if "message" in choice else None)
            if type(index) is not int or index < 0:
                raise ProtocolError("missing tool delta index")
            target = self.tools.setdefault(index, {"id": None, "name": "", "arguments": ""})
            if call.get("id"):
                if target["id"] is not None and target["id"] != call["id"]:
                    raise ProtocolError("tool identity changed during streaming")
                target["id"] = call["id"]
            function = call.get("function", {})
            target["name"] += function.get("name") or ""
            target["arguments"] += function.get("arguments") or ""
        if choice.get("finish_reason") is not None:
            self.finish = choice["finish_reason"]

    def result(self, *, allow_length=False):
        if not self.events or self.finish not in {"stop", "tool_calls", "length"}:
            raise ProtocolError("missing or unsupported finish reason")
        if self.finish == "length" and not allow_length:
            raise ProtocolError("generation exhausted its declared test budget")
        if bool(self.tools) != (self.finish == "tool_calls"):
            raise ProtocolError("tool emission and finish boundary disagree")
        if sorted(self.tools) != list(range(len(self.tools))):
            raise ProtocolError("tool indices have a gap")
        tools = []
        for value in self.tools.values():
            if not value["id"] or not value["name"]:
                raise ProtocolError("incomplete tool identity")
            try:
                args = json.loads(value["arguments"])
            except ValueError as error:
                raise ProtocolError("unfinished tool arguments") from error
            if not isinstance(args, dict):
                raise ProtocolError("tool arguments must be an object")
            tools.append({**value, "parsed_arguments": args})
        return {
            "content": self.content,
            "reasoning": self.reasoning,
            "token_ids": self.token_ids,
            "prompt_token_ids": self.prompt_token_ids,
            "tools": tools,
            "finish_reason": self.finish,
            "usage": self.usage,
        }


def live_group_members(group, *, proc_root=Path("/proc")):
    """Read only the created process group, including surviving threads of a dead leader."""
    members = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            if os.getpgid(pid) != group:
                continue
            fields = (entry / "stat").read_text().rsplit(") ", 1)[1].split()
            if int(fields[2]) != group:
                continue
            if fields[0] in {"Z", "X"}:
                for task in (entry / "task").iterdir():
                    try:
                        thread = (task / "stat").read_text().rsplit(") ", 1)[1].split()
                    except (FileNotFoundError, ProcessLookupError):
                        continue
                    if thread[0] not in {"Z", "X"}:
                        members.append(
                            {
                                "pid": pid,
                                "tid": int(task.name),
                                "state": thread[0],
                                "start_ticks": int(thread[19]),
                            }
                        )
                continue
            members.append({"pid": pid, "state": fields[0], "start_ticks": int(fields[19])})
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            members.append({"pid": pid, "state": "unreadable"})
    return sorted(members, key=lambda row: (row["pid"], row.get("tid", row["pid"])))


def await_group_exit(group, *, timeout=2):
    deadline = time.monotonic() + timeout
    while members := live_group_members(group):
        if time.monotonic() >= deadline:
            return members
        time.sleep(0.02)
    return []


class OwnedProcess:
    def __init__(self, argv, root: Path, *, env, timeout):
        if not argv or any(not isinstance(a, str) or "\0" in a for a in argv):
            raise DiagnosticError("invalid qualification process arguments")
        if timeout <= 0:
            raise DiagnosticError("qualification requires a positive deadline")
        root.mkdir(mode=0o700)
        self.root, self.timeout, self.process = root, timeout, None
        self.log = None
        self.closed = False
        self.close_error = None
        self.started = time.monotonic()
        write_private(root / "invocation.json", {"argv": argv, "deadline_seconds": timeout})
        try:
            fd = os.open(root / "process.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            self.log = os.fdopen(fd, "wb")
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "qwen_r9700_lab.conformance_supervisor",
                    str(os.getpid()),
                    str((root / "invocation.json").resolve()),
                ],
                env=env,
                cwd=root,
                stdin=subprocess.DEVNULL,
                stdout=self.log,
                stderr=self.log,
                start_new_session=True,
            )
        except BaseException:
            self.close()
            raise

    def check(self):
        if self.process.poll() is not None:
            raise DiagnosticError("owned qualification process exited; private log retained")
        if time.monotonic() - self.started >= self.timeout:
            raise TimeoutError("qualification process deadline expired")

    def wait(self):
        try:
            return self.process.wait(
                timeout=max(0.001, self.timeout - (time.monotonic() - self.started))
            )
        except subprocess.TimeoutExpired as error:
            raise TimeoutError("qualification child exceeded its deadline") from error
        finally:
            self.close()

    def close(self, *, crash=False, grace_seconds=10):
        if self.closed:
            if self.close_error is not None:
                raise self.close_error
            return
        if (
            isinstance(grace_seconds, bool)
            or not isinstance(grace_seconds, (int, float))
            or not math.isfinite(grace_seconds)
            or grace_seconds <= 0
        ):
            raise DiagnosticError("owned process requires a finite positive shutdown grace")
        self.closed = True
        members = []
        started = time.monotonic()
        grace_expired = False
        try:
            if self.process is not None:
                # Only the session we created; never pidof/pkill or a service manager.
                with contextlib.suppress(ProcessLookupError):
                    if crash:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    else:
                        os.kill(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=10 if crash else grace_seconds)
                except subprocess.TimeoutExpired:
                    grace_expired = True
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait(timeout=10)
                # A launcher may have exited before its owned workers did.
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.process.pid, signal.SIGKILL)
                members = await_group_exit(self.process.pid)
                if members:
                    raise DiagnosticError("owned workers remain alive after process-group shutdown")
        except BaseException as error:
            self.close_error = error
            if self.process is not None:
                from qwen_r9700_lab.conformance_gpu_lease import block_cleanup

                block_cleanup(
                    self.root,
                    process_group=self.process.pid,
                    reason=type(error).__name__ + ": " + str(error),
                    members=members,
                )
            raise
        finally:
            if self.log is not None:
                self.log.close()
            write_private(
                self.root / "shutdown.json",
                {
                    "mode": "crash" if crash else "graceful",
                    "grace_seconds": grace_seconds,
                    "grace_expired": grace_expired,
                    "seconds": time.monotonic() - started,
                    "returncode": self.process.returncode if self.process is not None else None,
                    "cleanup_error": str(self.close_error) if self.close_error else None,
                },
            )

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        raise ProtocolError("owned qualification endpoint attempted an HTTP redirect")


class OwnedClient:
    def __init__(self, process: OwnedProcess, port: int, nonce: str, execution: str):
        self.process, self.nonce, self.execution = process, nonce, execution
        self.base = f"http://127.0.0.1:{port}"
        self.verified = False
        # Ignore HTTP_PROXY/ALL_PROXY: qualification never leaves loopback.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirects())

    def request(self, path, body=None, *, timeout=30):
        self.process.check()
        if path != "/qwen-conformance/identity" and not self.verified:
            raise DiagnosticError("owned server identity has not been verified")
        if not path.startswith("/") or path.startswith("//"):
            raise DiagnosticError("qualification requests require a local relative route")
        request = urllib.request.Request(
            self.base + path,
            data=json.dumps(body, allow_nan=False).encode() if body is not None else None,
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.nonce},
        )
        return self.opener.open(request, timeout=timeout)

    def json(self, path, body=None, *, timeout=30):
        with self.request(path, body, timeout=timeout) as response:
            value = json.load(response)
        if not isinstance(value, dict) or "error" in value:
            raise ProtocolError("backend returned an error or malformed JSON")
        return value

    def connect(self, *, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.process.check()
            try:
                identity = self.json("/qwen-conformance/identity", timeout=1)
            except (urllib.error.URLError, TimeoutError):
                time.sleep(0.05)
                continue
            if identity.get("nonce") != self.nonce or identity.get("execution") != self.execution:
                raise DiagnosticError("server identity mismatch; no inference sent")
            self.verified = True
            return identity
        raise TimeoutError("owned server did not become ready")

    def completion(
        self, path, body, *, evidence: Path, allow_length=False, cancel_after=None, timeout=600
    ):
        if not math.isfinite(timeout) or timeout <= 0:
            raise DiagnosticError("completion requires a positive finite I/O timeout")
        write_private(evidence.with_suffix(".request.json"), body)
        fd = os.open(evidence.with_suffix(".response"), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        assembled, decoder = Completion(), SSEDecoder()
        started = time.monotonic()
        try:
            with os.fdopen(fd, "wb") as raw:
                try:
                    response = self.request(path, body, timeout=timeout)
                except urllib.error.HTTPError as error:
                    with error:
                        raw.write(error.read())
                    raise BackendResponseError(
                        f"backend returned HTTP {error.code}", http_status=error.code
                    ) from error
                with response:
                    if not body.get("stream", False):
                        data = response.read()
                        raw.write(data)
                        assembled.accept(json.loads(data))
                    else:
                        while chunk := response.read1(4096):
                            self.process.check()
                            raw.write(chunk)
                            raw.flush()
                            for event in decoder.feed(chunk):
                                assembled.accept(event)
                            if cancel_after is not None and assembled.events >= cancel_after:
                                return {"cancelled": True, "observed_events": assembled.events}
                        decoder.feed(b"", final=True)
        except Exception as error:
            write_private(
                evidence.with_suffix(".failure.json"),
                {
                    "error_type": type(error).__name__,
                    "http_status": getattr(error, "http_status", None),
                    "elapsed_seconds": time.monotonic() - started,
                    "observed_events": assembled.events,
                    "response_bytes": evidence.with_suffix(".response").stat().st_size,
                },
            )
            raise
        result = assembled.result(allow_length=allow_length)
        result["elapsed_seconds"] = time.monotonic() - started
        write_private(evidence.with_suffix(".result.json"), result)
        return result
