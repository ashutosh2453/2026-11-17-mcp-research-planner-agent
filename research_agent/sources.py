"""Free, key-less research sources: arXiv, Wikipedia and safe web page fetching."""

from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import httpx

USER_AGENT = "mcp-research-planner-agent/0.1 (+https://github.com/ashutosh2453/2026-11-17-mcp-research-planner-agent)"
ARXIV_API = "https://export.arxiv.org/api/query"
WIKI_API = "https://en.wikipedia.org/w/api.php"
ATOM = {"a": "http://www.w3.org/2005/Atom"}


class SourceError(RuntimeError):
    """A source could not be queried (network failure, bad response, blocked URL)."""


@dataclass
class Document:
    kind: str  # "arxiv" | "wikipedia" | "web"
    title: str
    url: str
    text: str
    published: str | None = None
    authors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", text or "")).strip()


class RateLimiter:
    """Enforce a minimum interval between calls (arXiv asks for one request every 3 s)."""

    def __init__(self, min_interval: float) -> None:
        self.min_interval = min_interval
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            delay = self._last + self.min_interval - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            self._last = time.monotonic()


class Sources:
    def __init__(self, client: httpx.AsyncClient | None = None, arxiv_interval: float = 3.0, timeout: float = 20.0) -> None:
        self._client = client or httpx.AsyncClient(timeout=timeout, headers={"User-Agent": USER_AGENT})
        self._arxiv_limit = RateLimiter(arxiv_interval)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _get(self, url: str, **params) -> httpx.Response:
        try:
            resp = await self._client.get(url, params=params or None)
            resp.raise_for_status()
            return resp
        except httpx.HTTPError as exc:
            raise SourceError(f"request to {urlparse(url).netloc} failed: {exc}") from exc

    # ---------------- arXiv ----------------
    async def search_arxiv(self, query: str, max_results: int = 5) -> list[Document]:
        terms = [t for t in re.findall(r"[\w\-]+", query) if len(t) > 1][:8]
        if not terms:
            raise SourceError("query has no searchable terms")
        await self._arxiv_limit.wait()
        resp = await self._get(ARXIV_API, search_query=" AND ".join(f"all:{t}" for t in terms),
                               start=0, max_results=max(1, min(max_results, 20)), sortBy="relevance")
        try:
            root = ET.fromstring(resp.text)
        except ET.ParseError as exc:
            raise SourceError(f"arXiv returned malformed XML: {exc}") from exc
        docs = []
        for entry in root.findall("a:entry", ATOM):
            url = (entry.findtext("a:id", "", ATOM) or "").strip()
            title = _clean(entry.findtext("a:title", "", ATOM))
            if not url or not title:
                continue
            docs.append(Document(
                kind="arxiv",
                title=title,
                url=url.replace("http://", "https://"),
                text=_clean(entry.findtext("a:summary", "", ATOM)),
                published=(entry.findtext("a:published", "", ATOM) or "")[:10] or None,
                authors=[_clean(a.findtext("a:name", "", ATOM)) for a in entry.findall("a:author", ATOM)][:6],
            ))
        return docs

    # ---------------- Wikipedia ----------------
    async def search_wikipedia(self, query: str, max_results: int = 5) -> list[Document]:
        if not query.strip():
            raise SourceError("query must not be empty")
        resp = await self._get(WIKI_API, action="query", list="search", srsearch=query,
                               srlimit=max(1, min(max_results, 20)), format="json", utf8=1)
        hits = resp.json().get("query", {}).get("search", [])
        return [Document(kind="wikipedia", title=h["title"],
                         url="https://en.wikipedia.org/wiki/" + h["title"].replace(" ", "_"),
                         text=_clean(h.get("snippet", ""))) for h in hits]

    async def read_wikipedia(self, title: str, max_chars: int = 6000) -> Document:
        resp = await self._get(WIKI_API, action="query", prop="extracts", explaintext=1, redirects=1,
                               titles=title, format="json", utf8=1)
        pages = resp.json().get("query", {}).get("pages", {})
        page = next(iter(pages.values()), {})
        if "missing" in page or not page.get("extract"):
            raise SourceError(f"Wikipedia article '{title}' not found")
        text = page["extract"]
        # Drop trailing reference sections, then turn "== Heading ==" markers into paragraph breaks.
        text = re.split(r"\n=+\s*(?:See also|References|External links|Further reading|Notes|Bibliography)\s*=+", text)[0]
        text = re.sub(r"\n*=+\s*[^=\n]+?\s*=+\n*", "\n", text)
        text = re.sub(r"\n{2,}", "\n", text).strip()
        return Document(kind="wikipedia", title=page["title"],
                        url="https://en.wikipedia.org/wiki/" + page["title"].replace(" ", "_"), text=text[:max_chars])

    # ---------------- arbitrary web pages ----------------
    async def fetch_url(self, url: str, max_chars: int = 8000, max_bytes: int = 2_000_000) -> Document:
        current = url
        for _ in range(4):  # follow at most 3 redirects, re-validating each hop
            await assert_public_http_url(current)
            try:
                async with self._client.stream("GET", current, follow_redirects=False) as resp:
                    if resp.is_redirect and "location" in resp.headers:
                        current = urljoin(current, resp.headers["location"])
                        continue
                    resp.raise_for_status()
                    ctype = resp.headers.get("content-type", "")
                    if not any(t in ctype for t in ("text/html", "text/plain", "application/xhtml")):
                        raise SourceError(f"unsupported content type '{ctype or 'unknown'}'")
                    body = bytearray()
                    async for chunk in resp.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > max_bytes:
                            break
                    raw = body.decode(resp.encoding or "utf-8", errors="replace")
            except httpx.HTTPError as exc:
                raise SourceError(f"fetching {current} failed: {exc}") from exc
            if "text/plain" in ctype:
                return Document(kind="web", title=current, url=current, text=_clean(raw)[:max_chars])
            title, text = html_to_text(raw)
            return Document(kind="web", title=title or current, url=current, text=text[:max_chars])
        raise SourceError("too many redirects")


async def assert_public_http_url(url: str) -> None:
    """Block SSRF: only http(s) URLs whose host resolves exclusively to public addresses."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise SourceError("only http(s) URLs with a hostname can be fetched")
    if parsed.username or parsed.password:
        raise SourceError("URLs with embedded credentials are not allowed")
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise SourceError(f"cannot resolve host '{parsed.hostname}'") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if not ip.is_global or ip.is_multicast:
            raise SourceError(f"refusing to fetch non-public address {ip}")


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg", "template"}
    BLOCK = {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "br", "tr", "section", "article", "pre", "blockquote"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> tuple[str, str]:
    parser = _TextExtractor()
    parser.feed(html)
    lines = (re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in "".join(parser.parts).split("\n"))
    text = "\n".join(line for line in lines if len(line) > 1)
    return parser.title.strip(), text
