"""Bounded asynchronous trajectories and tool calls."""
import asyncio

from carl.core.rollout import RolloutController, ContextOverflow


async def run_batch(rows, backend, execute, persist, concurrency=8, tool_concurrency=4,
                    max_segments=15, decision_tokens=2048, assimilation_tokens=256,
                    generation_tokens=4096, timeout=600):
    if min(concurrency, tool_concurrency, max_segments, decision_tokens, assimilation_tokens, generation_tokens) < 1:
        raise ValueError("Batch and generation limits must be positive")
    pending = iter(rows)
    tools = asyncio.Semaphore(tool_concurrency)

    async def one(row):
        controller = RolloutController(
            row["question"], max_segments=max_segments, invoke_max_tokens=decision_tokens,
            assimilate_max_tokens=assimilation_tokens, initial_context=backend.render_prompt(row["question"]))
        used = 0
        try:
            async with asyncio.timeout(timeout):
                while not controller.finished:
                    if controller.state == "tool":
                        async with tools:
                            result = await execute(controller.tool_request())
                        controller.accept_tool_result(result)
                    else:
                        if used >= generation_tokens:
                            controller.terminate("token_limit", "Episode generation budget exhausted")
                            break
                        request = controller.generation_request()
                        request["max_new_tokens"] = min(request["max_new_tokens"], generation_tokens-used)
                        output = await backend(**request)
                        used += len(output["ids"])
                        controller.accept_generation(output)
        except ContextOverflow as error:
            controller.terminate("context_limit", str(error))
        except TimeoutError:
            if not controller.finished:
                controller.terminate("timeout", "Episode deadline exceeded")
        await persist(row, controller.trajectory)

    async def worker():
        for row in pending:
            await one(row)

    tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
