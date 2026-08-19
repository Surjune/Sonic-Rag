"""Tests for tool definitions, execution and multi-turn tool calling."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from app.harness import MAX_TOOL_ROUNDS, GenerationRequest, GroqHarness
from app.tools import Tool, ToolCall, ToolRegistry, parse_tool_calls


def make_registry() -> ToolRegistry:
    async def lookup(query: str, top_k: int = 3) -> dict[str, object]:
        return {"query": query, "results": [f"result {i}" for i in range(top_k)]}

    def sync_tool(value: int) -> int:
        return value * 2

    async def boom() -> None:
        raise RuntimeError("tool exploded")

    async def slow() -> str:
        await asyncio.sleep(10)
        return "never"

    return ToolRegistry(
        [
            Tool("lookup", "Search", {"type": "object", "properties": {}}, lookup),
            Tool("sync_tool", "Double", {"type": "object", "properties": {}}, sync_tool),
            Tool("boom", "Fails", {"type": "object", "properties": {}}, boom),
            Tool("slow", "Hangs", {"type": "object", "properties": {}}, slow),
        ]
    )


def tool_call_message(*names: str) -> dict[str, object]:
    return {
        "role": "assistant",
        "tool_calls": [
            {"id": f"call_{i}", "type": "function",
             "function": {"name": name, "arguments": json.dumps({"query": "q"})}}
            for i, name in enumerate(names)
        ],
    }


def scripted_harness(messages: list[dict[str, object]]) -> GroqHarness:
    """Return each scripted assistant message in turn."""
    remaining = list(messages)

    def handler(request: httpx.Request) -> httpx.Response:
        message = remaining.pop(0) if remaining else {"role": "assistant", "content": "done"}
        return httpx.Response(200, json={"choices": [{"message": message}]})

    return GroqHarness(
        api_key="k",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        tools=make_registry(),
    )


class TestSchema:
    def test_schema_matches_groq_function_format(self) -> None:
        tool = Tool("t", "desc", {"type": "object", "properties": {}}, lambda: None)
        schema = tool.schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "t"
        assert schema["function"]["description"] == "desc"
        assert "parameters" in schema["function"]

    def test_registry_exposes_all_schemas(self) -> None:
        registry = make_registry()
        assert len(registry.schemas()) == len(registry)
        assert "lookup" in registry


class TestParsing:
    def test_parses_calls_and_arguments(self) -> None:
        calls = parse_tool_calls(tool_call_message("lookup"))
        assert len(calls) == 1
        assert calls[0].name == "lookup"
        assert calls[0].arguments == {"query": "q"}

    def test_message_without_tool_calls(self) -> None:
        assert parse_tool_calls({"role": "assistant", "content": "hi"}) == []

    def test_malformed_arguments_become_empty_dict(self) -> None:
        """Models do emit broken JSON; the registry reports it, parsing must not raise."""
        message = {
            "tool_calls": [
                {"id": "1", "function": {"name": "lookup", "arguments": "{not json"}}
            ]
        }
        assert parse_tool_calls(message)[0].arguments == {}

    def test_non_object_arguments_become_empty_dict(self) -> None:
        message = {"tool_calls": [{"id": "1", "function": {"name": "lookup", "arguments": "[1,2]"}}]}
        assert parse_tool_calls(message)[0].arguments == {}

    def test_call_without_name_is_skipped(self) -> None:
        assert parse_tool_calls({"tool_calls": [{"id": "1", "function": {}}]}) == []


class TestExecution:
    @pytest.mark.asyncio
    async def test_executes_async_tool(self) -> None:
        result = await make_registry().execute(ToolCall("1", "lookup", {"query": "x", "top_k": 2}))
        assert result.ok
        assert json.loads(result.content)["results"] == ["result 0", "result 1"]

    @pytest.mark.asyncio
    async def test_executes_sync_tool(self) -> None:
        result = await make_registry().execute(ToolCall("1", "sync_tool", {"value": 21}))
        assert result.ok
        assert result.content == "42"

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error_not_raise(self) -> None:
        result = await make_registry().execute(ToolCall("1", "missing", {}))
        assert not result.ok
        assert "unknown tool" in result.content

    @pytest.mark.asyncio
    async def test_raising_tool_is_reported_to_the_model(self) -> None:
        """A failing tool must not abort a request the model could still answer."""
        result = await make_registry().execute(ToolCall("1", "boom", {}))
        assert not result.ok
        assert "tool exploded" in result.content

    @pytest.mark.asyncio
    async def test_bad_arguments_reported(self) -> None:
        result = await make_registry().execute(ToolCall("1", "sync_tool", {"wrong": 1}))
        assert not result.ok
        assert "invalid arguments" in result.content

    @pytest.mark.asyncio
    async def test_hanging_tool_times_out(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("app.tools.TOOL_TIMEOUT_S", 0.05)
        result = await make_registry().execute(ToolCall("1", "slow", {}))
        assert not result.ok
        assert "timed out" in result.content

    @pytest.mark.asyncio
    async def test_result_converts_to_tool_message(self) -> None:
        result = await make_registry().execute(ToolCall("abc", "sync_tool", {"value": 1}))
        message = result.to_message()
        assert message["role"] == "tool"
        assert message["tool_call_id"] == "abc"
        assert message["name"] == "sync_tool"


class TestMultiTurn:
    @pytest.mark.asyncio
    async def test_executes_tools_then_answers(self) -> None:
        harness = scripted_harness(
            [tool_call_message("lookup"), {"role": "assistant", "content": "The answer."}]
        )
        result = await harness.generate_with_tools(GenerationRequest(query="q", contexts=["c"]))
        assert result.text == "The answer."
        assert result.tool_rounds == 1
        assert [call.name for call in result.tool_results] == ["lookup"]
        await harness.aclose()

    @pytest.mark.asyncio
    async def test_runs_parallel_calls_in_one_round(self) -> None:
        harness = scripted_harness(
            [tool_call_message("lookup", "sync_tool"), {"role": "assistant", "content": "ok"}]
        )
        result = await harness.generate_with_tools(GenerationRequest(query="q", contexts=["c"]))
        assert result.tool_rounds == 1
        assert len(result.tool_results) == 2
        await harness.aclose()

    @pytest.mark.asyncio
    async def test_no_tool_calls_answers_directly(self) -> None:
        harness = scripted_harness([{"role": "assistant", "content": "direct"}])
        result = await harness.generate_with_tools(GenerationRequest(query="q", contexts=["c"]))
        assert result.text == "direct"
        assert result.tool_rounds == 0
        await harness.aclose()

    @pytest.mark.asyncio
    async def test_round_cap_stops_a_looping_model(self) -> None:
        """Without the cap a model that always requests tools never terminates."""
        harness = scripted_harness([tool_call_message("lookup") for _ in range(10)])
        result = await harness.generate_with_tools(GenerationRequest(query="q", contexts=["c"]))
        assert result.tool_rounds == MAX_TOOL_ROUNDS
        await harness.aclose()

    @pytest.mark.asyncio
    async def test_failing_tool_still_produces_an_answer(self) -> None:
        harness = scripted_harness(
            [tool_call_message("boom"), {"role": "assistant", "content": "answered anyway"}]
        )
        result = await harness.generate_with_tools(GenerationRequest(query="q", contexts=["c"]))
        assert result.text == "answered anyway"
        assert result.tool_results[0].ok is False
        await harness.aclose()
