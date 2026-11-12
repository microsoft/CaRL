import asyncio
import os
from pathlib import Path

import pytest

from carl.sandbox.trusted import ToolLimits, TrustedExecutor


def executor(**kwargs):
    return TrustedExecutor(allow_trusted_execution=True, **kwargs)


def test_explicit_opt_in():
    with pytest.raises(ValueError):
        TrustedExecutor()
    with pytest.raises(ValueError):
        ToolLimits(wall_seconds=True)


@pytest.mark.parametrize("code,stdout", [
    ("347 * 28", "9716\n"),
    ("x=2; x+2", "4\n"),
    ("sum([\n1,\n2,\n3\n])", "6\n"),
    ("x=1", ""),
])
def test_actual_execution_and_last_expression(code, stdout):
    r = executor()(code)
    assert r.status == "success" and r.stdout == stdout
    assert r.execution_mode == "trusted_subprocess"


def test_computed_and_multiple_queries_are_resolved_by_parent():
    calls = []

    async def search(query):
        calls.append(query)
        return {"first lookup": "second", "second lookup": "found"}[query]

    r = executor(search=search)("name='first'\nnext_name=search(name+' lookup')\nprint(search(next_name+' lookup'))")
    assert r.status == "success" and r.stdout == "found\n"
    assert calls == ["first lookup", "second lookup"]
    assert len(r.searches) == 2


@pytest.mark.parametrize("code,status", [
    ("print(", "python_error"), ("1/0", "python_error"),
    ("import nonexistent_package", "python_error"), ("search('missing')", "search_error"),
    ("print('x'*200)", "output_limit"),
])
def test_tool_failures_are_explicit(code, status):
    result = executor(limits=ToolLimits(output_chars=80))(code)
    assert result.status == status and result.error
    assert "ERROR [" in result.observation


def test_fresh_variables_and_temporary_directory(monkeypatch):
    monkeypatch.setenv("CARL_TEST_PRIVATE_VALUE", "not-for-child")
    tool = executor()
    assert tool("prior=123").status == "success"
    assert tool("prior").status == "python_error"
    result = tool("import os\nprint(os.getcwd())\nprint('CARL_TEST_PRIVATE_VALUE' in os.environ)")
    directory, inherited = result.stdout.splitlines()
    assert inherited == "False" and not Path(directory).exists()


def test_timeout_and_cancel_reap_workers():
    tool = executor(limits=ToolLimits(wall_seconds=0.25))
    assert tool("import time;time.sleep(30)").status == "timeout"
    assert not tool._active_pids

    async def check():
        task = asyncio.create_task(tool.execute("import time;time.sleep(30)"))
        for _ in range(1000):
            if tool._active_pids:
                break
            await asyncio.sleep(0.001)
        assert tool._active_pids
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not tool._active_pids
    asyncio.run(check())


def test_search_infrastructure_errors_propagate():
    async def search(query):
        raise RuntimeError("Search provider failed")

    tool = executor(search=search)
    with pytest.raises(RuntimeError, match="Search provider"):
        tool("search('question')")
    assert not tool._active_pids


def test_query_count_is_bounded():
    async def search(query):
        return query

    r = executor(search=search, limits=ToolLimits(max_searches=1))("search('a')\nsearch('b')")
    assert r.status == "search_error" and len(r.searches) == 1


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")
def test_descendants_are_killed_after_success():
    import time

    r = executor()("import subprocess,sys\np=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])\nprint(p.pid)")
    assert r.status == "success"
    pid = int(r.stdout)
    for _ in range(100):
        state = Path(f"/proc/{pid}/stat")
        if not state.exists() or state.read_text().split()[2] == "Z":
            return
        time.sleep(0.01)
    pytest.fail("Tool descendant is still running")
