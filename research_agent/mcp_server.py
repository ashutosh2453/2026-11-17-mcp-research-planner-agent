"""MCP server exposing research tools over stdio.

Run standalone (e.g. for Claude Desktop or any MCP client):

    python -m research_agent.mcp_server

Tools: search_arxiv, search_wikipedia, read_wikipedia, fetch_url, save_note, search_notes.
"""

from __future__ import annotations

import os

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from .notebook import Notebook
from .sources import SourceError, Sources

INSTRUCTIONS = (
    "Research tools. Use search_wikipedia/read_wikipedia for background, search_arxiv for recent papers, "
    "fetch_url to read a specific page, and save_note/search_notes to keep findings between sessions."
)


def build_server(sources: Sources | None = None, notebook: Notebook | None = None) -> MCPServer:
    src = sources or Sources()
    nb = notebook or Notebook(os.getenv("RESEARCH_NOTEBOOK", "data/notebook.db"))
    server = MCPServer("research-tools", instructions=INSTRUCTIONS, version="0.1.0")

    def _bounded(n: int, lo: int, hi: int, name: str) -> int:
        if not lo <= n <= hi:
            raise ToolError(f"{name} must be between {lo} and {hi}")
        return n

    @server.tool()
    async def search_arxiv(query: str, max_results: int = 5) -> list[dict]:
        """Search arXiv for research papers. Returns title, url, abstract, published date and authors."""
        try:
            return [d.to_dict() for d in await src.search_arxiv(query, _bounded(max_results, 1, 20, "max_results"))]
        except SourceError as exc:
            raise ToolError(str(exc)) from exc

    @server.tool()
    async def search_wikipedia(query: str, max_results: int = 5) -> list[dict]:
        """Search English Wikipedia. Returns matching article titles, urls and snippets."""
        try:
            return [d.to_dict() for d in await src.search_wikipedia(query, _bounded(max_results, 1, 20, "max_results"))]
        except SourceError as exc:
            raise ToolError(str(exc)) from exc

    @server.tool()
    async def read_wikipedia(title: str, max_chars: int = 6000) -> dict:
        """Read the plain-text body of a Wikipedia article by exact title."""
        try:
            return (await src.read_wikipedia(title, _bounded(max_chars, 200, 50_000, "max_chars"))).to_dict()
        except SourceError as exc:
            raise ToolError(str(exc)) from exc

    @server.tool()
    async def fetch_url(url: str, max_chars: int = 8000) -> dict:
        """Fetch a public web page and return its readable text. Private/internal addresses are refused."""
        try:
            return (await src.fetch_url(url, _bounded(max_chars, 200, 50_000, "max_chars"))).to_dict()
        except SourceError as exc:
            raise ToolError(str(exc)) from exc

    @server.tool()
    def save_note(topic: str, content: str, source_url: str | None = None) -> dict:
        """Save a research note to the persistent notebook."""
        try:
            return nb.add(topic, content, source_url).to_dict()
        except ValueError as exc:
            raise ToolError(str(exc)) from exc

    @server.tool()
    def search_notes(query: str = "", limit: int = 10) -> list[dict]:
        """Full-text search over saved notes (most recent notes if query is empty)."""
        return [n.to_dict() for n in nb.search(query, _bounded(limit, 1, 100, "limit"))]

    return server


def main() -> None:
    build_server().run("stdio")


if __name__ == "__main__":
    main()
