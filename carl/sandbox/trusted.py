"""Opt-in Python execution in restricted containers or trusted host subprocesses.

Host subprocesses are not a security boundary: they retain the invoking user's
filesystem and network permissions.
"""
from __future__ import annotations
import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
import json
import logging
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
from uuid import uuid4

from carl.core.tool import ToolResult

logger = logging.getLogger(__name__)
Search = Callable[[str], Awaitable[str]]


@dataclass(frozen=True)
class ToolLimits:
    wall_seconds: float = 5
    search_seconds: float = 5
    cpu_seconds: int = 5
    memory_bytes: int = 256 * 1024 * 1024
    file_bytes: int = 1024 * 1024
    code_chars: int = 16384
    output_chars: int = 12000
    query_chars: int = 1024
    max_searches: int = 5
    search_result_chars: int = 16000

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            if name not in ("wall_seconds", "search_seconds") and type(value) is not int:
                raise ValueError(f"{name} must be an integer")


def child_environment(directory: str) -> dict[str, str]:
    env = {"HOME": directory, "TMPDIR": directory, "TEMP": directory, "TMP": directory,
           "LANG": "C.UTF-8"}
    if os.name == "nt" and "SystemRoot" in os.environ:
        env["SystemRoot"] = os.environ["SystemRoot"]
    return env


async def stop_worker(process):
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.returncode is None:
        process.kill()
    await process.wait()


class TrustedExecutor:
    def __init__(self, *, allow_trusted_execution: bool = False,
                 search: Search | None = None, limits: ToolLimits = ToolLimits(),
                 container_image: str | None = None):
        if not allow_trusted_execution and not container_image:
            raise ValueError("Trusted execution requires explicit allow_trusted_execution=True")
        if container_image is not None and (not container_image or container_image.startswith("-")):
            raise ValueError("Container image must be a nonempty image reference")
        self.container_image = container_image
        self.search = search
        self.limits = limits
        self._worker_path = str(Path(__file__).with_name("worker.py").resolve())
        self._active_pids: set[int] = set()
        if not container_image:
            logger.warning(
                "Trusted Python execution is enabled, NOT a security sandbox. "
                "Job-user filesystem/network access remains; use controlled inputs only.")
        if os.name != "posix":
            logger.warning("Windows host: no POSIX resource limits or descendant process-group cleanup")

    def __call__(self, code: str) -> ToolResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.execute(code))
        raise RuntimeError("Use await executor.execute(...) inside an asyncio event loop")

    async def execute(self, code: str, sample=None) -> ToolResult:
        if not isinstance(code, str) or not code.strip() or len(code) > self.limits.code_chars:
            raise ValueError(f"Tool code must be nonempty and at most {self.limits.code_chars} characters")
        execution_id, started = uuid4().hex, time.monotonic()
        searches = []
        with tempfile.TemporaryDirectory(prefix="carl_tool_") as directory:
            container_name = "carl-" + execution_id
            command = (
                ["docker", "run", "--rm", "-i", "--name", container_name,
                 "--network", "none", "--read-only", "--cap-drop", "ALL",
                 "--security-opt", "no-new-privileges", "--user", "65534:65534",
                 "--pids-limit", "32", "--memory", str(self.limits.memory_bytes),
                 "--cpus", "1", "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m",
                 "--mount", f"type=bind,source={self._worker_path},target=/worker.py,readonly",
                 self.container_image, "python", "-I", "-S", "/worker.py"]
                if self.container_image else [sys.executable, "-I", "-S", self._worker_path])
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, cwd=directory,
                env=child_environment(directory), start_new_session=os.name == "posix", close_fds=True,
                limit=(self.limits.output_chars * 12 + self.limits.query_chars * 6 + 8192),
            )
            self._active_pids.add(process.pid)
            result = None

            async def communicate():
                nonlocal result
                process.stdin.write((json.dumps({"code": code, "limits": asdict(self.limits)}) + "\n").encode())
                await process.stdin.drain()
                while True:
                    line = await process.stdout.readline()
                    if not line:
                        await process.wait()
                        return
                    try:
                        message = json.loads(line)
                    except (ValueError, UnicodeError) as error:
                        raise RuntimeError("Worker protocol was corrupted by tool output") from error
                    if message.get("kind") == "result":
                        result = message
                        return
                    if message.get("kind") != "search":
                        raise RuntimeError("Unexpected worker message")
                    query = message.get("query")
                    if (not isinstance(query, str) or not query.strip()
                            or len(query) > self.limits.query_chars or len(searches) >= self.limits.max_searches):
                        raise RuntimeError("Worker sent an invalid or excessive search request")
                    event = {"query": query, "status": "pending"}
                    searches.append(event)
                    if self.search is None:
                        response = {"error": "Search is not enabled for this executor"}
                        event["status"] = "unavailable"
                    else:
                        try:
                            text = await asyncio.wait_for(self.search(query), self.limits.search_seconds)
                        except asyncio.TimeoutError:
                            event["status"] = "timeout"
                            response = {"error": "Search timed out"}
                        else:
                            if not isinstance(text, str):
                                raise TypeError("Search provider must return a string")
                            if len(text) > self.limits.search_result_chars:
                                response = {"error": "Search result exceeds the configured limit"}
                                event["status"] = "output_limit"
                            else:
                                response = {"result": text}
                                event.update(status="success", result=text)
                    process.stdin.write((json.dumps(response) + "\n").encode())
                    await process.stdin.drain()

            async def bounded_stderr():
                chunks, size = [], 0
                while True:
                    chunk = await process.stderr.read(4096)
                    if not chunk:
                        return b"".join(chunks).decode("utf-8", errors="replace")
                    size += len(chunk)
                    if size > self.limits.output_chars * 4:
                        raise RuntimeError("Worker stderr exceeded transport limit")
                    chunks.append(chunk)

            io_task = asyncio.create_task(communicate())
            error_task = asyncio.create_task(bounded_stderr())
            stopped = False
            try:
                await asyncio.wait_for(io_task, self.limits.wall_seconds)
                # Tear down the invocation group even on success; user subprocesses
                # must not keep the pipe or temporary directory alive.
                await stop_worker(process)
                stopped = True
                stderr = await asyncio.wait_for(error_task, 2)
            except asyncio.TimeoutError:
                result = {"status": "timeout", "error": f"Tool exceeded {self.limits.wall_seconds}s"}
                stderr = ""
            finally:
                if not stopped:
                    await stop_worker(process)
                for task in (io_task, error_task):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(io_task, error_task, return_exceptions=True)
                self._active_pids.discard(process.pid)
                if self.container_image:
                    cleanup = await asyncio.create_subprocess_exec(
                        "docker", "rm", "-f", container_name,
                        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                    _, errors = await cleanup.communicate()
                    if cleanup.returncode and b"No such container" not in errors:
                        raise RuntimeError(f"Container cleanup failed: {errors.decode(errors='replace')}")
            if result is None:
                if os.name == "posix" and process.returncode in (
                    -signal.SIGKILL, -signal.SIGXCPU, -signal.SIGXFSZ,
                ):
                    result = {"status": "resource_limit", "error": f"Worker terminated by signal {-process.returncode}"}
                else:
                    raise RuntimeError(f"Worker exited without a result ({process.returncode}): {stderr}")
            status = result.get("status")
            if status not in ("success", "python_error", "search_error", "output_limit", "resource_limit", "timeout"):
                raise RuntimeError(f"Invalid worker result status: {status!r}")
            output = result.get("stdout", "")
            captured_error = result.get("stderr", "")
            if not isinstance(output, str) or not isinstance(captured_error, str):
                raise RuntimeError("Malformed worker output")
            if len(output) > self.limits.output_chars or len(captured_error) > self.limits.output_chars:
                raise RuntimeError("Worker output violated the capture limit")
            tool_result = ToolResult(
                execution_id, status, output, captured_error, result.get("error"),
                time.monotonic() - started, searches,
            )
            if self.container_image:
                tool_result.execution_mode = "restricted_container"
            if status != "success":
                logger.warning("Tool %s ended with %s: %s", execution_id, status, tool_result.error)
            return tool_result
