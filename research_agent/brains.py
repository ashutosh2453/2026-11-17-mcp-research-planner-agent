"""Agent "brains": the decision-making policy behind plan / reflect / write.

* ``HeuristicBrain`` - deterministic, offline, zero-cost. Decomposes the question
  into topics, chains tools based on what earlier observations returned, and
  writes an extractive, fully cited report.
* ``OllamaBrain`` - a local LLM (free, via Ollama) chooses tools from the MCP
  tool list and writes the report. Every LLM decision is schema-validated and
  falls back to the heuristic policy if the model returns something unusable.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import httpx

log = logging.getLogger("research_agent")

STOP = set("""a an the and or but of to in on at by for with from as is are was were be been being this that these
those it its into about over under than so such can could should would will shall may might must do does did have
has had i you he she we they them his her our their your my me us what which who whom whose when where why how all
any both each few more most other some own same very just also there here not no nor only too s t vs versus between
tell explain describe give show find summarise summarize overview introduction intro please latest recent recently
current state art research papers paper study studies work works know learn understand compare comparison difference
differences new trends trend top best main key important things use used using""".split())


@dataclass
class Step:
    tool: str
    args: dict
    purpose: str = ""

    def key(self) -> str:
        return f"{self.tool}:{json.dumps(self.args, sort_keys=True)}"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Observation:
    step: Step
    ok: bool
    data: Any = None
    error: str | None = None


@dataclass
class Evidence:
    n: int
    kind: str
    title: str
    url: str
    text: str
    published: str | None = None
    authors: list[str] = field(default_factory=list)


class Brain(Protocol):
    name: str

    async def plan(self, question: str, tools: list[dict]) -> list[Step]: ...
    async def reflect(self, question: str, observations: list[Observation], evidence: list[Evidence],
                      tools: list[dict], budget: int) -> list[Step]: ...
    async def write(self, question: str, evidence: list[Evidence], notes: list[dict]) -> str: ...


# ---------------------------------------------------------------- helpers

def keywords(question: str) -> list[str]:
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9+\-.#]*", question)
    return [w.strip(".") for w in words if w.lower().strip(".") not in STOP and len(w.strip(".")) > 1]


def topics(question: str) -> list[str]:
    """Split comparison questions ("X vs Y", "compare X and Y") into separate topics."""
    quoted = re.findall(r'"([^"]+)"', question)
    if quoted:
        return quoted[:3]
    m = re.search(r"(?:compare|comparing|difference between|differences between)\s+(.+?)\s+(?:and|with|to|vs\.?|versus)\s+(.+?)(?:[?.!]|\s+(?:for|in|on|when)\s|$)", question, re.I) \
        or re.search(r"(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?:[?.!]|\s+(?:for|in|on|when)\s|$)", question, re.I)
    if m:
        parts = [" ".join(keywords(p)) for p in m.groups()]
        if all(parts):
            # Carry the shared context ("... for adding knowledge to LLMs") into each topic so an
            # ambiguous term like "fine-tuning" isn't searched in isolation.
            context = [w for w in keywords(question[m.end():]) if w.lower() not in " ".join(parts).lower()][-2:]
            return [" ".join([p, *context]) for p in parts]
    kw = keywords(question)
    return [" ".join(kw[:6])] if kw else []


def sentences(text: str) -> list[str]:
    """Complete sentences only: starts with a capital/digit, ends with punctuation (drops snippet fragments)."""
    text = re.sub(r"\s+", " ", text).strip()
    out = []
    for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\"])", text):
        s = s.strip()
        if 40 <= len(s) <= 400 and re.match(r"[A-Z0-9(\"]", s) and s[-1] in ".!?":
            out.append(s)
    return out


def select_evidence(question: str, evidence: list[Evidence], min_overlap: int = 2) -> list[Evidence]:
    """Keep sources that share enough vocabulary with the question, renumbered 1..k for citation."""
    q = _tokens(question)
    need = min(min_overlap, len(q)) or 1
    kept = []
    for e in evidence:
        overlap = len(q & _tokens(f"{e.title} {e.text[:1500]}"))
        if overlap >= need:
            kept.append(e)
    return [Evidence(i, e.kind, e.title, e.url, e.text, e.published, e.authors) for i, e in enumerate(kept, start=1)]


def _tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in STOP and len(t) > 2}


def _has_tool(tools: list[dict], name: str) -> bool:
    return any(t["name"] == name for t in tools)


# ---------------------------------------------------------------- heuristic brain

class HeuristicBrain:
    name = "heuristic"

    async def plan(self, question: str, tools: list[dict]) -> list[Step]:
        steps: list[Step] = []
        tps = topics(question)
        if not tps:
            return steps
        if _has_tool(tools, "search_notes"):
            steps.append(Step("search_notes", {"query": " ".join(keywords(question)[:6]), "limit": 5}, "recall earlier findings"))
        for t in tps:
            if _has_tool(tools, "search_wikipedia"):
                steps.append(Step("search_wikipedia", {"query": t, "max_results": 3}, f"background on {t}"))
        if _has_tool(tools, "search_arxiv"):
            for t in tps:
                steps.append(Step("search_arxiv", {"query": t, "max_results": 5}, f"recent papers on {t}"))
        return steps

    async def reflect(self, question, observations, evidence, tools, budget) -> list[Step]:
        if budget <= 0:
            return []
        done = {o.step.key() for o in observations}
        follow: list[Step] = []
        q_tokens = _tokens(question)

        # Chain: read the article that best matches each search's own topic (not one already read/planned).
        read_titles = {o.step.args.get("title") for o in observations if o.step.tool == "read_wikipedia"}
        for o in observations:
            if o.step.tool == "search_wikipedia" and o.ok and o.data and _has_tool(tools, "read_wikipedia"):
                topic = _tokens(o.step.args["query"])
                candidates = [d for d in o.data if d["title"] not in read_titles]
                if not candidates:
                    continue
                best = max(candidates, key=lambda d: (2 * len(_tokens(d["title"]) & topic) + len(_tokens(d["title"]) & q_tokens),
                                                      -o.data.index(d)))
                if not _tokens(best["title"]) & (topic | q_tokens):
                    continue
                read_titles.add(best["title"])
                step = Step("read_wikipedia", {"title": best["title"], "max_chars": 5000}, f"read '{best['title']}'")
                if step.key() not in done:
                    follow.append(step)

        # Broaden an arXiv search that came back thin (too many AND-ed terms).
        for o in observations:
            if o.step.tool == "search_arxiv" and (not o.ok or len(o.data or []) < 2):
                words = o.step.args["query"].split()
                if len(words) > 2:
                    step = Step("search_arxiv", {"query": " ".join(words[:2]), "max_results": 5}, "broaden thin arXiv search")
                    if step.key() not in done:
                        follow.append(step)

        # Read a URL the user explicitly mentioned.
        for url in re.findall(r"https?://\S+", question):
            step = Step("fetch_url", {"url": url.rstrip(").,"), "max_chars": 6000}, "read the page the question cites")
            if _has_tool(tools, "fetch_url") and step.key() not in done:
                follow.append(step)
        return follow[:budget]

    async def write(self, question: str, evidence: list[Evidence], notes: list[dict]) -> str:
        return extractive_report(question, evidence, notes)


def extractive_report(question: str, evidence: list[Evidence], notes: list[dict]) -> str:
    q = _tokens(question)
    lines = [f"# Research report: {question.strip()}", ""]
    if not evidence:
        lines += ["No sources could be retrieved for this question. Try rephrasing it or check network access.", ""]
        return "\n".join(lines)

    # Summary: Maximal Marginal Relevance over all evidence sentences.
    cands: list[tuple[float, int, str, set[str]]] = []
    for ev in evidence:
        if len(ev.text) < 200:  # search snippets are fragments, not statements
            continue
        for i, s in enumerate(sentences(ev.text)[:40]):
            toks = _tokens(s)
            if not toks:
                continue
            rel = len(toks & q) / math.sqrt(len(q) or 1) + (0.3 if i == 0 else 0) + (0.15 if ev.kind == "wikipedia" else 0)
            if rel > 0:
                cands.append((rel, ev.n, s, toks))
    chosen: list[tuple[float, int, str, set[str]]] = []
    while cands and len(chosen) < 5:
        def mmr(c):
            red = max((len(c[3] & s[3]) / len(c[3] | s[3]) for s in chosen), default=0.0)
            same_src = sum(1 for s in chosen if s[1] == c[1])
            return 0.7 * c[0] - 0.3 * red - 0.2 * same_src
        best = max(cands, key=mmr)
        cands.remove(best)
        chosen.append(best)
    lines += ["## Summary", ""]
    lines += [f"- {s} [{n}]" for _, n, s, _ in chosen] or ["- The retrieved sources do not directly address the question."]
    lines.append("")

    wiki = [e for e in evidence if e.kind == "wikipedia" and len(e.text) > 300]
    if wiki:
        lines += ["## Background", ""]
        for e in wiki[:3]:
            intro = " ".join(sentences(e.text)[:2])
            if intro:
                lines.append(f"**{e.title}**: {intro} [{e.n}]")
                lines.append("")

    papers = sorted((e for e in evidence if e.kind == "arxiv"), key=lambda e: e.published or "", reverse=True)
    if papers:
        lines += ["## Recent research", ""]
        for e in papers[:8]:
            first = (sentences(e.text) or [e.text[:200]])[0]
            who = (e.authors[0] + (" et al." if len(e.authors) > 1 else "")) if e.authors else "unknown authors"
            lines.append(f"- **{e.title}** ({(e.published or '')[:4]}, {who}): {first} [{e.n}]")
        lines.append("")

    web = [e for e in evidence if e.kind == "web"]
    if web:
        lines += ["## From the cited page", ""]
        for e in web:
            lines.append(f"- {' '.join(sentences(e.text)[:3])} [{e.n}]")
        lines.append("")

    related = [n for n in notes if n.get("topic", "").strip().lower() != question.strip().lower()
               and len(_tokens(n.get("topic", "")) & q) >= min(2, len(q))]
    if related:
        lines += ["## From earlier research sessions", ""]
        for n in related[:3]:
            first = re.sub(r"\s*\[\d+\]", "", n["content"].lstrip("- ").split("\n")[0])[:240]
            lines.append(f"- *{n['topic']}* ({n.get('created_at', '')[:10]}): {first}")
        lines.append("")

    lines += ["## Sources", ""]
    lines += [f"{e.n}. [{e.title}]({e.url}){' (' + e.published[:4] + ')' if e.published else ''}" for e in evidence]
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------- Ollama brain

class OllamaBrain:
    """Local LLM policy via Ollama's /api/chat with JSON-constrained output."""

    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434", timeout: float = 180.0,
                 client: httpx.AsyncClient | None = None) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.name = f"ollama:{model}"
        self.fallback = HeuristicBrain()
        self.warnings: list[str] = []
        self._client = client or httpx.AsyncClient(timeout=timeout)

    async def _chat(self, system: str, user: str, json_mode: bool) -> str:
        payload: dict[str, Any] = {"model": self.model, "stream": False, "options": {"temperature": 0},
                                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if json_mode:
            payload["format"] = "json"
        resp = await self._client.post(f"{self.base_url}/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def _parse_steps(self, raw: str, tools: list[dict], limit: int) -> list[Step]:
        data = json.loads(raw)
        items = data.get("steps", data) if isinstance(data, dict) else data
        if not isinstance(items, list):
            raise ValueError("expected a list of steps")
        schemas = {t["name"]: t.get("input_schema", {}) for t in tools}
        steps = []
        for it in items[:limit]:
            tool, args = it.get("tool"), it.get("args", {})
            if tool not in schemas or not isinstance(args, dict):
                raise ValueError(f"invalid tool call {it!r}")
            missing = [r for r in schemas[tool].get("required", []) if r not in args]
            if missing:
                raise ValueError(f"{tool} missing required args {missing}")
            allowed = set(schemas[tool].get("properties", {}))
            steps.append(Step(tool, {k: v for k, v in args.items() if k in allowed}, str(it.get("purpose", ""))))
        return steps

    def _tool_menu(self, tools: list[dict]) -> str:
        return "\n".join(f"- {t['name']}({', '.join(t.get('input_schema', {}).get('properties', {}))}): {t['description']}" for t in tools)

    async def plan(self, question, tools):
        system = ("You are a research planner. Choose tool calls that gather evidence to answer the question. "
                  'Reply with JSON: {"steps": [{"tool": name, "args": {...}, "purpose": text}]}. Use at most 5 steps.\n'
                  f"Tools:\n{self._tool_menu(tools)}")
        try:
            return self._parse_steps(await self._chat(system, f"Question: {question}", True), tools, 5)
        except (httpx.HTTPError, ValueError, KeyError, json.JSONDecodeError) as exc:
            self.warnings.append(f"plan: LLM output unusable ({exc}); used heuristic plan")
            log.warning(self.warnings[-1])
            return await self.fallback.plan(question, tools)

    async def reflect(self, question, observations, evidence, tools, budget):
        if budget <= 0:
            return []
        seen = "\n".join(f"- {o.step.tool}({json.dumps(o.step.args)}): " + ("ok, " + _brief(o.data) if o.ok else f"error: {o.error}")
                         for o in observations)
        system = ("You review research progress. If the evidence is enough to answer, reply {\"steps\": []}. "
                  f"Otherwise propose up to {budget} new tool calls (never repeat one already made) as "
                  '{"steps": [{"tool": name, "args": {...}, "purpose": text}]}.\n' f"Tools:\n{self._tool_menu(tools)}")
        try:
            return self._parse_steps(await self._chat(system, f"Question: {question}\nTool calls so far:\n{seen}", True), tools, budget)
        except (httpx.HTTPError, ValueError, KeyError, json.JSONDecodeError) as exc:
            self.warnings.append(f"reflect: LLM output unusable ({exc}); used heuristic reflection")
            log.warning(self.warnings[-1])
            return await self.fallback.reflect(question, observations, evidence, tools, budget)

    async def write(self, question, evidence, notes):
        if not evidence:
            return extractive_report(question, evidence, notes)
        numbered = "\n\n".join(f"[{e.n}] {e.title} ({e.kind}{', ' + e.published[:4] if e.published else ''})\n{e.text[:1500]}" for e in evidence)
        system = ("Write a concise markdown research report answering the question using ONLY the numbered sources. "
                  "Cite every factual sentence with [n]. Sections: Summary, Details, Open questions. Do not invent sources.")
        try:
            body = await self._chat(system, f"Question: {question}\n\nSources:\n{numbered}", False)
        except (httpx.HTTPError, KeyError) as exc:
            self.warnings.append(f"write: LLM unavailable ({exc}); used extractive report")
            return extractive_report(question, evidence, notes)
        valid = {e.n for e in evidence}
        body = re.sub(r"\[(\d+)\]", lambda m: m.group(0) if int(m.group(1)) in valid else "", body)
        cited = Counter(int(n) for n in re.findall(r"\[(\d+)\]", body))
        if not cited:
            self.warnings.append("write: LLM report had no valid citations; used extractive report")
            return extractive_report(question, evidence, notes)
        sources = "\n".join(f"{e.n}. [{e.title}]({e.url})" for e in evidence)
        return f"# Research report: {question.strip()}\n\n{body.strip()}\n\n## Sources\n\n{sources}\n"


def _brief(data: Any) -> str:
    if isinstance(data, list):
        return f"{len(data)} results: " + "; ".join(str(d.get("title", ""))[:60] for d in data[:5] if isinstance(d, dict))
    if isinstance(data, dict):
        return f"'{data.get('title', '')}' ({len(str(data.get('text', '')))} chars)"
    return str(data)[:120]
