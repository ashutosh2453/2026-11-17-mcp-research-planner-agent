import socket
import time

import httpx
import pytest
import respx

from research_agent.sources import ARXIV_API, WIKI_API, RateLimiter, SourceError, Sources, assert_public_http_url, html_to_text

ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2005.11401v4</id>
    <published>2020-05-22T17:29:22Z</published>
    <title>Retrieval-Augmented Generation for
      Knowledge-Intensive NLP Tasks</title>
    <summary>  Large pre-trained language models store knowledge.  </summary>
    <author><name>Patrick Lewis</name></author><author><name>Ethan Perez</name></author>
  </entry>
  <entry><id></id><title>broken entry</title></entry>
</feed>"""


@pytest.fixture
def sources():
    return Sources(arxiv_interval=0)


@respx.mock
async def test_arxiv_parses_atom_and_builds_and_query(sources):
    route = respx.get(ARXIV_API).mock(return_value=httpx.Response(200, text=ATOM))
    docs = await sources.search_arxiv("retrieval augmented generation", 3)
    assert route.calls[0].request.url.params["search_query"] == "all:retrieval AND all:augmented AND all:generation"
    assert len(docs) == 1
    d = docs[0]
    assert d.title == "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
    assert d.url == "https://arxiv.org/abs/2005.11401v4"
    assert (d.published, d.authors, d.text) == ("2020-05-22", ["Patrick Lewis", "Ethan Perez"], "Large pre-trained language models store knowledge.")


@respx.mock
async def test_arxiv_errors_become_source_errors(sources):
    respx.get(ARXIV_API).mock(return_value=httpx.Response(503))
    with pytest.raises(SourceError, match="export.arxiv.org"):
        await sources.search_arxiv("rag")
    respx.get(ARXIV_API).mock(return_value=httpx.Response(200, text="<not xml"))
    with pytest.raises(SourceError, match="malformed"):
        await sources.search_arxiv("rag")
    with pytest.raises(SourceError, match="no searchable terms"):
        await sources.search_arxiv("? !")


@respx.mock
async def test_wikipedia_search_strips_markup(sources):
    respx.get(WIKI_API).mock(return_value=httpx.Response(200, json={"query": {"search": [
        {"title": "Model Context Protocol", "snippet": 'The <span class="searchmatch">Model</span> Context Protocol'}]}}))
    [d] = await sources.search_wikipedia("mcp")
    assert d.text == "The Model Context Protocol"
    assert d.url == "https://en.wikipedia.org/wiki/Model_Context_Protocol"


@respx.mock
async def test_wikipedia_read_drops_headings_and_reference_sections(sources):
    extract = "Intro sentence here.\n\n== History ==\nIt began in 2024.\n\n== See also ==\nOther things\n== References ==\nx"
    respx.get(WIKI_API).mock(return_value=httpx.Response(200, json={"query": {"pages": {"1": {"title": "MCP", "extract": extract}}}}))
    d = await sources.read_wikipedia("MCP")
    assert d.text == "Intro sentence here.\nIt began in 2024."
    respx.get(WIKI_API).mock(return_value=httpx.Response(200, json={"query": {"pages": {"-1": {"title": "Nope", "missing": ""}}}}))
    with pytest.raises(SourceError, match="not found"):
        await sources.read_wikipedia("Nope")


def test_html_to_text_skips_boilerplate():
    title, text = html_to_text("<html><head><title> Doc </title><style>x{}</style></head><body><nav>Menu</nav>"
                               "<h1>Heading</h1><p>First &amp; para.</p><script>evil()</script><footer>foot</footer></body></html>")
    assert title == "Doc"
    assert text == "Heading\nFirst & para."


@pytest.fixture
def resolve(monkeypatch):
    table = {}

    async def fake_getaddrinfo(self, host, port, **kw):
        if host not in table:
            raise socket.gaierror("unknown")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (table[host], port))]

    import asyncio
    monkeypatch.setattr(asyncio.AbstractEventLoop, "getaddrinfo", fake_getaddrinfo, raising=False)
    monkeypatch.setattr(type(asyncio.new_event_loop()), "getaddrinfo", fake_getaddrinfo)
    return table


@pytest.mark.parametrize("url,ip", [
    ("http://localhost/", "127.0.0.1"), ("http://intranet/", "10.0.0.5"), ("http://metadata/", "169.254.169.254"),
    ("http://router/", "192.168.1.1"), ("http://six/", "::1"),
])
async def test_ssrf_guard_blocks_private_addresses(resolve, url, ip):
    resolve[url.split("/")[2]] = ip
    with pytest.raises(SourceError, match="non-public"):
        await assert_public_http_url(url)


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/x", "https://user:pw@example.com/", "http:///nohost"])
async def test_ssrf_guard_rejects_bad_schemes_and_credentials(url):
    with pytest.raises(SourceError):
        await assert_public_http_url(url)


@respx.mock
async def test_fetch_url_follows_redirects_but_revalidates_each_hop(sources, resolve):
    resolve["public.example"] = "93.184.216.34"
    resolve["internal.example"] = "10.1.2.3"
    respx.get("https://public.example/start").mock(return_value=httpx.Response(302, headers={"location": "/page"}))
    respx.get("https://public.example/page").mock(return_value=httpx.Response(
        200, headers={"content-type": "text/html; charset=utf-8"}, text="<title>T</title><p>Readable body text.</p>"))
    d = await sources.fetch_url("https://public.example/start")
    assert (d.title, d.url, d.text) == ("T", "https://public.example/page", "Readable body text.")

    respx.get("https://public.example/evil").mock(return_value=httpx.Response(302, headers={"location": "http://internal.example/admin"}))
    with pytest.raises(SourceError, match="non-public"):
        await sources.fetch_url("https://public.example/evil")


@respx.mock
async def test_fetch_url_rejects_binary_content(sources, resolve):
    resolve["public.example"] = "93.184.216.34"
    respx.get("https://public.example/file.pdf").mock(return_value=httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF"))
    with pytest.raises(SourceError, match="unsupported content type"):
        await sources.fetch_url("https://public.example/file.pdf")


async def test_rate_limiter_spaces_calls():
    limiter = RateLimiter(0.05)
    start = time.monotonic()
    for _ in range(3):
        await limiter.wait()
    assert time.monotonic() - start >= 0.1
