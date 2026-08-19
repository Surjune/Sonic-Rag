"""Tool definitions and registry for the model harness.

Kept separate from harness.py so the harness stays generic: it knows how to
advertise tools, parse a call and feed the result back, but nothing about what
any particular tool does. Concrete tools are registered by the API layer, which
is the only place allowed to reach into retrieval.

Handlers are async and every one of them is wrapped: a tool that raises must
return an error to the model rather than killing the request, because the model
can often recover by answering without it.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

ToolHandler = Callable[..., Awaitable[Any]] | Callable[..., Any]

# A tool that hangs would hold the whole request open; the model is better
# served by a prompt error than by waiting.
TOOL_TIMEOUT_S = 5.0


@dataclass
class Tool:
    """One callable exposed to the model, in Groq/OpenAI function schema."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolCall:
    """A call requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """The outcome of executing one call, fed back as a `tool` message."""

    call_id: str
    name: str
    content: str
    latency_ms: float
    ok: bool = True

    def to_message(self) -> dict[str, str]:
        return {
            "role": "tool",
            "tool_call_id": self.call_id,
            "name": self.name,
            "content": self.content,
        }


class ToolRegistry:
    """Holds the tools available to a harness instance."""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {tool.name: tool for tool in (tools or [])}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    async def execute(self, call: ToolCall) -> ToolResult:
        """Run one tool call, converting every failure into a tool message.

        An unknown name, bad arguments, a timeout and a raising handler all come
        back as content the model can read. Raising instead would abort a
        request the model might well have completed without the tool.
        """
        started = time.perf_counter()

        def finish(content: str, ok: bool) -> ToolResult:
            return ToolResult(
                call_id=call.id,
                name=call.name,
                content=content,
                latency_ms=(time.perf_counter() - started) * 1000,
                ok=ok,
            )

        tool = self._tools.get(call.name)
        if tool is None:
            return finish(json.dumps({"error": f"unknown tool {call.name!r}"}), False)

        try:
            result = tool.handler(**call.arguments)
            if inspect.isawaitable(result):
                result = await asyncio.wait_for(result, timeout=TOOL_TIMEOUT_S)
        except asyncio.TimeoutError:
            return finish(json.dumps({"error": f"{call.name} timed out"}), False)
        except TypeError as error:
            # Almost always the model inventing or omitting an argument.
            return finish(json.dumps({"error": f"invalid arguments: {error}"}), False)
        except Exception as error:  # noqa: BLE001 - surfaced to the model, not swallowed
            return finish(json.dumps({"error": f"{type(error).__name__}: {error}"}), False)

        if isinstance(result, str):
            return finish(result, True)
        try:
            return finish(json.dumps(result, ensure_ascii=False, default=str), True)
        except (TypeError, ValueError) as error:
            return finish(json.dumps({"error": f"unserializable result: {error}"}), False)


def parse_tool_calls(message: dict[str, Any]) -> list[ToolCall]:
    """Extract tool calls from an assistant message.

    Arguments arrive as a JSON *string*, and models do sometimes emit malformed
    JSON. A bad payload becomes an empty argument dict so the registry can
    report the failure to the model, rather than raising here.
    """
    calls: list[ToolCall] = []
    for raw in message.get("tool_calls") or []:
        function = raw.get("function") or {}
        name = function.get("name")
        if not name:
            continue
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except (TypeError, ValueError):
            arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        calls.append(ToolCall(id=str(raw.get("id") or name), name=str(name), arguments=arguments))
    return calls
