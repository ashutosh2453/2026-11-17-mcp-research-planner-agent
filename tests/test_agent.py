import json
import re

import httpx
import pytest
import respx

from research_agent.agent import run_research
from research_agent.brains import Evidence, HeuristicBrain, OllamaBrain, Step, select_evidence, sentences, topics
from research_agent.mcp_server import build_server
from research_agent.notebook import Notebook
from research_agent.toolbox import McpToolbox

from .conftest import FakeSources

Q = "Compare retrieval-augmented generation and fine-tuning for adding knowledge to LLMs"


# ---------------------------------------------------------------- planning helpers

def test_topics_split_comparisons_and_keep_shared_context():
    assert topics(Q) == ["retrieval-augmented generation knowledge LLMs", "fine-tuning knowledge LLMs"]
    assert topics("BM25 vs dense retrieval") == ["BM25", "dense retrieval"]
    assert topics('What is "graph neural network" pooling?') == ["graph neural network"]
    assert topics("What is the Model Context Protocol?") == ["Model Context Protocol"]
    assert topics("what is it?") == []


def test_sentences_drop_fragments():
    text = "detection, and retrieval-augmented generation. RAG grounds answers in retrieved documents for accuracy. Short."
    assert sentences(text) == ["RAG grounds answers in retrieved documents for accuracy."]


def test_select_evidence_filters_and_renumbers():
    ev = [Evidence(1, "wikipedia", "Fine-tuned universe", "u1", "cosmology constants"),
          Evidence(2, "arxiv", "Fine-Tuning or Retrieval? Knowledge Injection in LLMs", "u2", "retrieval and fine-tuning")]
    out = select_evidence(Q, ev)
    assert [(e.n, e.url) for e in out] == [(1, "u2")]


# ---------------------------------------------------------------- full agent over the real MCP server (in-process)

async def run(question, sources=None, max_steps=8, notebook=None, brain=None):
    sources = sources or FakeSources()
    async with McpToolbox(build_server(sources, notebook or Notebook())) as tb:
        result = await run_research(question, brain or HeuristicBrain(), tb, max_steps=max_steps)
    return result, sources


async def test_agent_plans_chains_tools_and_cites_sources():
    result, src = await run(Q)
    acts = [t for t in result["trace"] if t["node"] == "act"]
    tools = [a["tool"] for a in acts]
    assert tools[:5] == ["search_notes", "search_wikipedia", "search_wikipedia", "search_arxiv", "search_arxiv"]
    # Reflection chained a read of the article matching each search, skipping the cosmology page.
    assert ("wiki_read", "Retrieval-augmented generation") in src.calls
    assert ("wiki_read", "Fine-tuning (deep learning)") in src.calls
    assert ("wiki_read", "Fine-tuned universe") not in src.calls

    report = result["report"]
    assert report.startswith(f"# Research report: {Q}")
    for heading in ("## Summary", "## Background", "## Recent research", "## Sources"):
        assert heading in report
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", report)}
    assert cited and cited <= {e["n"] for e in result["evidence"]}
    assert all("Fine-tuned universe" != e["title"] for e in result["evidence"])
    assert result["trace"][-1] == {"node": "remember", "ok": True, "error": None}


async def test_agent_remembers_findings_across_sessions():
    nb = Notebook()
    await run(Q, notebook=nb)
    assert nb.search("retrieval")[0].topic == Q
    result, _ = await run("How does retrieval-augmented generation reduce hallucination in LLMs?", notebook=nb)
    assert "## From earlier research sessions" in result["report"]
    assert Q in result["report"]


async def test_agent_respects_step_budget():
    result, _ = await run(Q, max_steps=3)
    assert result["tool_calls"] == 3
    assert "## Sources" in result["report"]


async def test_agent_survives_tool_failures_and_broadens_thin_searches():
    result, src = await run("graph neural network message passing oversmoothing", sources=FakeSources(fail_arxiv=True))
    failed = [t for t in result["trace"] if t["node"] == "act" and not t["ok"]]
    assert failed and failed[0]["error"] == "arXiv is down"
    assert ("arxiv", "graph neural") in src.calls  # broadened retry
    assert result["report"].startswith("# Research report")


async def test_agent_reads_url_mentioned_in_question():
    result, src = await run("Summarise retrieval-augmented generation pipelines from https://example.org/rag-guide.")
    assert ("fetch", "https://example.org/rag-guide") in src.calls
    assert "## From the cited page" in result["report"]


async def test_empty_question_and_bad_budget_are_rejected():
    with pytest.raises(ValueError):
        await run("   ")
    with pytest.raises(ValueError):
        await run(Q, max_steps=0)


async def test_mcp_tools_validate_arguments():
    async with McpToolbox(build_server(FakeSources(), Notebook())) as tb:
        assert tb.tool_names == {"search_arxiv", "search_wikipedia", "read_wikipedia", "fetch_url", "save_note", "search_notes"}
        bad = await tb.call("search_arxiv", {"query": "x", "max_results": 99})
        assert not bad.ok and "between 1 and 20" in bad.error
        blocked = await tb.call("fetch_url", {"url": "http://internal.corp/"})
        assert not blocked.ok and "non-public" in blocked.error
        assert not (await tb.call("save_note", {"topic": " ", "content": "x"})).ok
        assert (await tb.call("nope", {})).error == "unknown tool 'nope'"
        saved = await tb.call("save_note", {"topic": "MCP", "content": 'quotes " and OR operators AND NOT'})
        assert saved.ok
        found = await tb.call("search_notes", {"query": 'OR "AND NOT'})  # FTS syntax is neutralised
        assert found.ok and found.data[0]["topic"] == "MCP"


# ---------------------------------------------------------------- Ollama brain (HTTP mocked)

def _chat(content):
    return httpx.Response(200, json={"message": {"role": "assistant", "content": content}})


TOOLS = [{"name": "search_arxiv", "description": "papers", "input_schema": {"properties": {"query": {}, "max_results": {}}, "required": ["query"]}},
         {"name": "read_wikipedia", "description": "article", "input_schema": {"properties": {"title": {}}, "required": ["title"]}}]


@respx.mock
async def test_ollama_brain_uses_validated_llm_plan():
    route = respx.post("http://ollama:11434/api/chat").mock(return_value=_chat(json.dumps(
        {"steps": [{"tool": "search_arxiv", "args": {"query": "rag", "bogus": 1}, "purpose": "papers"}]})))
    brain = OllamaBrain("tiny", "http://ollama:11434")
    steps = await brain.plan("what is rag?", TOOLS)
    assert steps == [Step("search_arxiv", {"query": "rag"}, "papers")]  # unknown arg stripped
    body = json.loads(route.calls[0].request.content)
    assert body["format"] == "json" and body["model"] == "tiny" and "search_arxiv(query, max_results)" in body["messages"][0]["content"]
    assert brain.warnings == []


@respx.mock
@pytest.mark.parametrize("content", ["not json", json.dumps({"steps": [{"tool": "delete_everything", "args": {}}]}),
                                     json.dumps({"steps": [{"tool": "read_wikipedia", "args": {}}]})])
async def test_ollama_brain_falls_back_on_unusable_output(content):
    respx.post("http://ollama:11434/api/chat").mock(return_value=_chat(content))
    brain = OllamaBrain("tiny", "http://ollama:11434")
    steps = await brain.plan("What is the Model Context Protocol?", TOOLS)
    assert [s.tool for s in steps] == ["search_arxiv"]  # heuristic plan restricted to available tools
    assert brain.warnings and "heuristic" in brain.warnings[0]


@respx.mock
async def test_ollama_brain_strips_invented_citations_and_falls_back_when_none_remain():
    ev = [Evidence(1, "wikipedia", "RAG", "https://w/rag", "RAG grounds answers in retrieved documents for better accuracy.")]
    respx.post("http://ollama:11434/api/chat").mock(return_value=_chat("## Summary\nRAG grounds answers [1]. It is magic [7]."))
    brain = OllamaBrain("tiny", "http://ollama:11434")
    report = await brain.write("what is rag", ev, [])
    assert "[1]" in report and "[7]" not in report and "## Sources" in report

    respx.post("http://ollama:11434/api/chat").mock(return_value=_chat("No citations at all."))
    report = await brain.write("what is rag", ev, [])
    assert "## Summary" in report and any("no valid citations" in w for w in brain.warnings)


@respx.mock
async def test_ollama_down_does_not_break_the_agent():
    respx.post("http://localhost:11434/api/chat").mock(side_effect=httpx.ConnectError("refused"))
    result, _ = await run(Q, brain=OllamaBrain())
    assert "## Sources" in result["report"]
    assert len(result["warnings"]) >= 2
