"""Timing spans for seed's trace (seed.obs.trace's format), written by this process to the run's spans.jsonl.

seed runs PSHuman as a subprocess with SEED_TRACE_FILE / SEED_TRACE_ID / SEED_TRACE_PARENT set; without them every call
here is a no-op. Owner 2026-10-12: "do we have spans for model load times?" -- PSHuman was one opaque span per pass.
"""
import contextlib
import json
import os
import secrets
import socket
import time

#: when this module was first imported: the process start, near enough (it is imported at the top of inference.py)
PROCESS_T0_NS = time.time_ns()


def emit(name, start_ns, end_ns, **attributes):
    path = os.environ.get("SEED_TRACE_FILE")
    if not path:
        return
    line = json.dumps({
        "trace_id": os.environ.get("SEED_TRACE_ID") or "0" * 32, "span_id": secrets.token_hex(8),
        "parent_span_id": os.environ.get("SEED_TRACE_PARENT") or None, "name": name,
        "start_unix_nano": int(start_ns), "end_unix_nano": int(end_ns), "status": "ok",
        "attributes": {"kind": "pshuman_phase", **{k: v if isinstance(v, (int, float, str, bool)) else str(v)
                                                   for k, v in attributes.items()}},
        "resource": {"host": socket.gethostname(), "pid": os.getpid(), "process": "pshuman"},
    }) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, line.encode())
    finally:
        os.close(fd)


@contextlib.contextmanager
def span(name, **attributes):
    t0 = time.time_ns()
    try:
        yield
    finally:
        emit(name, t0, time.time_ns(), **attributes)
