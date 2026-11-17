"""Command line: ``research-agent "your question" [--brain ollama] [--out report.md]``."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from .agent import run_research
from .brains import HeuristicBrain, OllamaBrain
from .toolbox import McpToolbox


def _print_trace(trace: list[dict]) -> None:
    for t in trace:
        if t["node"] == "act":
            status = "ok " if t["ok"] else "ERR"
            print(f"  [{status}] {t['tool']}({json.dumps(t['args'])}) {t['ms']} ms" + (f" - {t['error']}" if t["error"] else ""), file=sys.stderr)
        elif t["node"] in ("plan", "reflect"):
            print(f"  {t['node']}: {len(t['steps'])} step(s) planned", file=sys.stderr)


async def _main(args: argparse.Namespace) -> int:
    brain = OllamaBrain(args.model, args.ollama_url) if args.brain == "ollama" else HeuristicBrain()
    env = {**os.environ, "RESEARCH_NOTEBOOK": args.notebook}
    async with McpToolbox.stdio(env=env) as toolbox:
        print(f"Connected to MCP server with tools: {', '.join(sorted(toolbox.tool_names))}", file=sys.stderr)
        result = await run_research(args.question, brain, toolbox, max_steps=args.max_steps)
    if args.verbose:
        _print_trace(result["trace"])
    for w in result["warnings"]:
        print(f"warning: {w}", file=sys.stderr)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(result["report"])
    if args.out:
        Path(args.out).write_text(result["report"], encoding="utf-8")
        print(f"Report saved to {args.out}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="research-agent", description="Agentic research over MCP tools (arXiv, Wikipedia, web, notebook).")
    p.add_argument("question")
    p.add_argument("--brain", choices=["heuristic", "ollama"], default=os.getenv("RESEARCH_BRAIN", "heuristic"))
    p.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "llama3.2"))
    p.add_argument("--ollama-url", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--notebook", default=os.getenv("RESEARCH_NOTEBOOK", "data/notebook.db"))
    p.add_argument("--out", help="write the markdown report to this file")
    p.add_argument("--json", action="store_true", help="print the full result (report, evidence, trace) as JSON")
    p.add_argument("-v", "--verbose", action="store_true", help="print the tool-call trace")
    args = p.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to a legacy code page
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        return asyncio.run(_main(args))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
