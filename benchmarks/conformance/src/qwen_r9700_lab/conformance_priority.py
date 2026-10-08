"""Synthetic Pi-style priority leases with durable, content-free IPC receipts.

Each chat has its own ordered control queue. Failed heartbeats remain evidence
and do not disable later attempts. This driver does not implement the scheduler.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import urllib.error
from concurrent.futures import Future, ThreadPoolExecutor

from qwen_r9700_lab.diagnostic_contract import DiagnosticError, write_private


class PriorityLease:
    def __init__(self, client, chat, abi, root, *, automatic=True):
        self.client, self.chat, self.abi, self.root = client, chat, abi, root
        root.mkdir(mode=0o700)
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="priority-control")
        self.stop = threading.Event()
        self.sequence = self.pending = self.priority = 0
        self.active = self.touched = self.closed = False
        self.failures = []
        self.confirmations = []
        self.heart = threading.Thread(target=self._heartbeats, daemon=True) if automatic else None
        if self.heart:
            self.heart.start()

    def _heartbeats(self):
        while not self.stop.wait(10):
            self.heartbeat()

    @staticmethod
    def _done():
        future = Future()
        future.set_result(None)
        return future

    def _submit(self, purpose):
        # Caller holds the chat's lock; snapshot state at enqueue time just as
        # Pi does. A delayed heartbeat must never overwrite a later release.
        if self.closed:
            raise DiagnosticError("priority lease is closed")
        self.sequence += 1
        self.pending += 1
        self.touched = True
        update = {
            "chat_id": self.chat["id"],
            "abi": self.abi,
            "client": self.chat["id"][:32],
            "answer": self.chat["id"][:32],
            "sequence": self.sequence,
            "priority": self.priority,
            "active": self.active,
        }
        return self.pool.submit(self._send, update, purpose, time.monotonic_ns())

    def start(self, priority):
        with self.lock:
            if type(priority) is not int or priority not in (0, 1, 2):
                raise ValueError("invalid priority")
            self.priority, self.active = priority, True
            # Default-priority Pi chats do not send startup or heartbeat IPC.
            return self._submit("start") if priority else self._done()

    def change(self, priority):
        with self.lock:
            if type(priority) is not int or priority not in (0, 1, 2):
                raise ValueError("invalid priority")
            self.priority = priority
            return self._submit("change")

    def heartbeat(self):
        with self.lock:
            if self.closed or not self.active or not self.priority or self.pending:
                return self._done()
            return self._submit("heartbeat")

    def release(self):
        with self.lock:
            was_active, self.active = self.active, False
            if was_active and self.touched:
                return self._submit("release")
            return self._done()

    def _send(self, update, purpose, queued_ns):
        name = f"{update['sequence']:05d}"
        receipt = {
            "schema": "urn:qwen:priority-control-receipt:v1",
            "purpose": purpose,
            "update": update,
            "queued_ns": queued_ns,
            "started_ns": time.monotonic_ns(),
            "timeout_seconds": 6.5,
            "http_status": None,
        }
        try:
            write_private(self.root / (name + ".begin.json"), receipt)
            with self.client.request("/qwen-radiance/priority", update, timeout=6.5) as response:
                receipt["http_status"] = response.status
                result = json.load(response)
            if (
                not isinstance(result, dict)
                or result.get("applied") is not True
                or result.get("priority") != update["priority"]
            ):
                raise DiagnosticError("priority update not applied by real scheduler")
            receipt["status"] = "confirmed"
            receipt["confirmation"] = {
                key: result[key]
                for key in ("applied", "priority", "sequence", "active", "lease_seconds")
                if key in result
            }
            return result
        except Exception as error:
            receipt["status"] = "failed"
            receipt["error_type"] = type(error).__name__
            if isinstance(error, urllib.error.HTTPError):
                receipt["http_status"] = error.code
                with error:
                    raw = error.read(4097)
                receipt["error_body_sha256"] = hashlib.sha256(raw).hexdigest()
                receipt["error_body_bytes_read"] = len(raw)
                receipt["error_body_truncated"] = len(raw) > 4096
            # No arbitrary error text, request contents or authorization headers
            # enter the diagnostic receipt.
            raise
        finally:
            receipt["finished_ns"] = time.monotonic_ns()
            receipt["seconds"] = (receipt["finished_ns"] - receipt["started_ns"]) / 1e9
            with self.lock:
                self.pending -= 1
                target = (
                    self.confirmations if receipt.get("status") == "confirmed" else self.failures
                )
                target.append(receipt)
            write_private(self.root / (name + ".result.json"), receipt)

    def close(self):
        with self.lock:
            self.closed, self.active = True, False
        self.stop.set()
        if self.heart:
            self.heart.join()
        self.pool.shutdown(wait=True)
