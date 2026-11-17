"""Thin async wrapper around an MCP client: tool discovery and normalised tool results."""

from __future__ import annotations

import json
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from mcp import Client, StdioServerParameters


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict


@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error: str | None = None


class McpToolbox:
    """Connects to the research MCP server, either in-process or as a stdio subprocess.

    ``McpToolbox.stdio()`` launches ``python -m research_agent.mcp_server`` and talks MCP over
    its stdin/stdout, which is exactly how an external MCP host would use the server.
    """

    def __init__(self, target: Any) -> None:
        self._target = target
        self._stack: AsyncExitStack | None = None
        self._client: Client | None = None
        self.specs: list[ToolSpec] = []

    @classmethod
    def stdio(cls, env: dict[str, str] | None = None) -> "McpToolbox":
        return cls(StdioServerParameters(command=sys.executable, args=["-m", "research_agent.mcp_server"], env=env))

    async def __aenter__(self) -> "McpToolbox":
        self._stack = AsyncExitStack()
        self._client = await self._stack.enter_async_context(Client(self._target))
        listed = await self._client.list_tools()
        self.specs = [ToolSpec(t.name, t.description or "", t.input_schema) for t in listed.tools]
        return self

    async def __aexit__(self, *exc) -> None:
        if self._stack:
            await self._stack.aclose()

    @property
    def tool_names(self) -> set[str]:
        return {s.name for s in self.specs}

    async def call(self, name: str, args: dict) -> ToolResult:
        if self._client is None:
            raise RuntimeError("toolbox is not connected; use 'async with'")
        if name not in self.tool_names:
            return ToolResult(False, error=f"unknown tool '{name}'")
        result = await self._client.call_tool(name, args)
        text = "\n".join(getattr(c, "text", "") for c in result.content)
        if result.is_error:
            return ToolResult(False, error=text.removeprefix(f"Error executing tool {name}: ") or "tool failed")
        structured = result.structured_content
        if isinstance(structured, dict) and set(structured) == {"result"}:
            return ToolResult(True, structured["result"])
        if structured is not None:
            return ToolResult(True, structured)
        try:
            return ToolResult(True, json.loads(text))
        except json.JSONDecodeError:
            return ToolResult(True, text)
