"""Resource-bounded research worker, launched with Python -I -S.

This is NOT a security sandbox. It retains the job user's filesystem/network
permissions. Resource limits and capture protect against ordinary mistakes,
not deliberately hostile Python. Only the parent enables this mode.
"""
import ast
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
import sys
import traceback


class OutputLimit(RuntimeError):
    pass


class SearchFailure(RuntimeError):
    pass


class LimitedOutput(io.StringIO):
    def __init__(self, limit):
        super().__init__()
        self.limit = limit
        self.used = 0
        self.exceeded = False

    def write(self, text):
        if self.used + len(text) > self.limit:
            self.exceeded = True
            raise OutputLimit(f"Output exceeded {self.limit} characters")
        self.used += len(text)
        return super().write(text)


def compile_auto_display(code):
    tree = ast.parse(code, filename="<tool>", mode="exec")
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = tree.body[-1]
        is_print = (isinstance(last.value, ast.Call) and isinstance(last.value.func, ast.Name)
                    and last.value.func.id == "print")
        if not is_print:
            last.value = ast.Call(func=ast.Name(id="print", ctx=ast.Load()),
                                  args=[last.value], keywords=[])
            ast.fix_missing_locations(tree)
    return compile(tree, "<tool>", "exec")


def apply_limits(limits):
    if os.name == "posix":
        import resource
        for key, value in (
            (resource.RLIMIT_CPU, limits["cpu_seconds"]),
            (resource.RLIMIT_AS, limits["memory_bytes"]),
            (resource.RLIMIT_FSIZE, limits["file_bytes"]),
            (resource.RLIMIT_NOFILE, 64),
            (resource.RLIMIT_CORE, 0),
        ):
            _, hard = resource.getrlimit(key)
            target = value if hard == resource.RLIM_INFINITY else min(value, hard)
            resource.setrlimit(key, (target, target))


def main():
    request = json.loads(sys.stdin.readline())
    limits = request["limits"]
    apply_limits(limits)
    channel_in, channel_out = sys.stdin, sys.stdout

    def send(message):
        channel_out.write(json.dumps(message, ensure_ascii=True) + "\n")
        channel_out.flush()

    count = 0

    def search(query):
        nonlocal count
        if not isinstance(query, str) or not query.strip():
            raise SearchFailure("search requires a nonempty string")
        if len(query) > limits["query_chars"] or count >= limits["max_searches"]:
            raise SearchFailure("Search query/count limit exceeded")
        count += 1
        send({"kind": "search", "query": query})
        response = json.loads(channel_in.readline())
        if response.get("error") is not None:
            raise SearchFailure(response["error"])
        return response["result"]

    stdout, stderr = LimitedOutput(limits["output_chars"]), LimitedOutput(limits["output_chars"])
    status, error = "success", None
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            program = compile_auto_display(request["code"])
            scope = {"__name__": "__main__", "search": search}
            exec(program, scope)
        except OutputLimit as caught:
            status, error = "output_limit", str(caught)
        except SearchFailure as caught:
            status, error = "search_error", str(caught)
        except MemoryError:
            status, error = "resource_limit", "Python memory allocation exceeded the limit"
        except BaseException as caught:
            # These are user-code errors returned to the controller, not suppressed host failures.
            status = "python_error"
            error = "".join(traceback.format_exception_only(type(caught), caught)).strip()[:limits["output_chars"]]
    if stdout.exceeded or stderr.exceeded:
        status, error = "output_limit", "Captured stdout/stderr exceeded its limit"
    send({"kind": "result", "status": status, "stdout": stdout.getvalue(),
          "stderr": stderr.getvalue(), "error": error})


if __name__ == "__main__":
    main()
