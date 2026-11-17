"""LangGraph research agent: plan -> act (MCP tools) -> reflect -> ... -> write -> remember."""

from __future__ import annotations

import logging
import operator
import time
from dataclasses import asdict
from typing import Annotated, Any, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from .brains import Brain, Evidence, Observation, Step, select_evidence
from .toolbox import ToolResult, ToolSpec

log = logging.getLogger("research_agent")


class Toolbox(Protocol):
    specs: list[ToolSpec]

    async def call(self, name: str, args: dict) -> ToolResult: ...


class AgentState(TypedDict, total=False):
    question: str
    max_steps: int
    pending: list[Step]
    observations: Annotated[list[Observation], operator.add]
    evidence: list[Evidence]
    notes: list[dict]
    report: str
    trace: Annotated[list[dict], operator.add]
    reflections: int


def _merge_evidence(evidence: list[Evidence], docs: list[dict]) -> list[Evidence]:
    """Number new documents; a fuller text for a known URL (article vs. search snippet) replaces the snippet."""
    by_url = {e.url: e for e in evidence}
    out = list(evidence)
    for d in docs:
        if not isinstance(d, dict) or not d.get("url") or not d.get("title"):
            continue
        text = d.get("text") or ""
        if d["url"] in by_url:
            if len(text) > len(by_url[d["url"]].text):
                by_url[d["url"]].text = text
            continue
        ev = Evidence(len(out) + 1, d.get("kind", "web"), d["title"], d["url"], text, d.get("published"), d.get("authors") or [])
        out.append(ev)
        by_url[ev.url] = ev
    return out


def build_graph(brain: Brain, toolbox: Toolbox, max_reflections: int = 3):
    tools = [{"name": s.name, "description": s.description, "input_schema": s.input_schema} for s in toolbox.specs]

    async def plan(state: AgentState) -> dict:
        steps = await brain.plan(state["question"], tools)
        return {"pending": steps, "trace": [{"node": "plan", "steps": [s.to_dict() for s in steps]}]}

    async def act(state: AgentState) -> dict:
        pending = list(state.get("pending", []))
        step = pending.pop(0)
        started = time.perf_counter()
        res = await toolbox.call(step.tool, step.args)
        ms = round((time.perf_counter() - started) * 1000)
        obs = Observation(step, res.ok, res.data, res.error)
        update: dict[str, Any] = {"pending": pending, "observations": [obs],
                                  "trace": [{"node": "act", "tool": step.tool, "args": step.args, "purpose": step.purpose,
                                             "ok": res.ok, "error": res.error, "ms": ms}]}
        if res.ok:
            if step.tool == "search_notes":
                update["notes"] = (state.get("notes") or []) + list(res.data or [])
            else:
                docs = res.data if isinstance(res.data, list) else [res.data]
                update["evidence"] = _merge_evidence(state.get("evidence") or [], docs)
        else:
            log.warning("tool %s failed: %s", step.tool, res.error)
        return update

    def after_act(state: AgentState) -> str:
        if state.get("pending") and len(state["observations"]) < state["max_steps"]:
            return "act"
        return "reflect"

    async def reflect(state: AgentState) -> dict:
        budget = state["max_steps"] - len(state["observations"])
        rounds = state.get("reflections", 0) + 1
        steps: list[Step] = []
        if budget > 0 and rounds <= max_reflections:
            done = {o.step.key() for o in state["observations"]}
            proposed = await brain.reflect(state["question"], state["observations"], state.get("evidence") or [], tools, budget)
            steps = [s for s in proposed if s.key() not in done][:budget]
        return {"pending": steps, "reflections": rounds,
                "trace": [{"node": "reflect", "budget": budget, "steps": [s.to_dict() for s in steps]}]}

    def after_reflect(state: AgentState) -> str:
        return "act" if state.get("pending") else "write"

    async def write(state: AgentState) -> dict:
        gathered = state.get("evidence") or []
        selected = select_evidence(state["question"], gathered)
        report = await brain.write(state["question"], selected, state.get("notes") or [])
        return {"report": report, "evidence": selected,
                "trace": [{"node": "write", "chars": len(report), "sources_gathered": len(gathered), "sources_used": len(selected)}]}

    async def remember(state: AgentState) -> dict:
        if not any(s.name == "save_note" for s in toolbox.specs) or not state.get("evidence"):
            return {}
        summary = state["report"].split("## Summary", 1)[-1].split("\n## ", 1)[0].strip() or state["report"][:800]
        res = await toolbox.call("save_note", {"topic": state["question"][:200], "content": summary[:2000]})
        return {"trace": [{"node": "remember", "ok": res.ok, "error": res.error}]}

    g = StateGraph(AgentState)
    for name, fn in [("plan", plan), ("act", act), ("reflect", reflect), ("write", write), ("remember", remember)]:
        g.add_node(name, fn)
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", lambda s: "act" if s.get("pending") else "write", {"act": "act", "write": "write"})
    g.add_conditional_edges("act", after_act, {"act": "act", "reflect": "reflect"})
    g.add_conditional_edges("reflect", after_reflect, {"act": "act", "write": "write"})
    g.add_edge("write", "remember")
    g.add_edge("remember", END)
    return g.compile()


async def run_research(question: str, brain: Brain, toolbox: Toolbox, max_steps: int = 8) -> dict:
    if not question.strip():
        raise ValueError("question must not be empty")
    if not 1 <= max_steps <= 30:
        raise ValueError("max_steps must be between 1 and 30")
    graph = build_graph(brain, toolbox)
    state = await graph.ainvoke(
        {"question": question, "max_steps": max_steps, "observations": [], "trace": [], "evidence": [], "notes": []},
        config={"recursion_limit": max_steps * 3 + 20},
    )
    return {
        "question": question,
        "brain": brain.name,
        "report": state["report"],
        "evidence": [asdict(e) for e in state.get("evidence", [])],
        "trace": state["trace"],
        "tool_calls": len(state["observations"]),
        "warnings": list(getattr(brain, "warnings", [])),
    }
