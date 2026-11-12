import asyncio
from unittest.mock import patch

from carl.sandbox.trusted import TrustedExecutor


def test_container_launch_is_restricted_and_does_not_fall_back():
    commands = []

    async def unavailable(*command, **kwargs):
        commands.append(command)
        raise FileNotFoundError("docker daemon unavailable")

    async def check():
        with patch("asyncio.create_subprocess_exec", unavailable):
            try:
                await TrustedExecutor(container_image="worker-image").execute("print(4)")
            except FileNotFoundError:
                pass
            else:
                raise AssertionError("Container launch must fail explicitly")

    asyncio.run(check())
    assert len(commands) == 1
    command = commands[0]
    for argument in ("--network", "none", "--read-only", "--cap-drop", "ALL",
                     "no-new-privileges", "--user", "65534:65534", "--pids-limit", "--memory"):
        assert argument in command
    assert "worker-image" in command
    assert not any("docker.sock" in x for x in command)
