import asyncio

from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain_core.outputs import LLMResult, Generation


async def test_resets_between_runs() -> None:
    handler = AsyncIteratorCallbackHandler()

    # first run
    await handler.on_llm_start({}, ["foo"])
    await handler.on_llm_new_token("1")
    await handler.on_llm_end(LLMResult(generations=[[Generation(text="")]]))

    # second run
    await handler.on_llm_start({}, ["bar"])
    await handler.on_llm_new_token("2")
    await handler.on_llm_end(LLMResult(generations=[[Generation(text="")]]))

    tokens = [token async for token in handler.aiter()]
    assert tokens == ["2"]
