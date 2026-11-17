import pytest

from research_agent.sources import Document, SourceError

RAG_TEXT = ("Retrieval-augmented generation (RAG) is a technique that enables large language models to retrieve "
            "and incorporate new information from external data sources. With RAG, models first refer to a specified "
            "set of documents, then respond to user queries. RAG reduces hallucination because answers are grounded "
            "in retrieved documents rather than only in model parameters.")
FT_TEXT = ("In deep learning, fine-tuning is the process of adapting a model trained for one task to perform a "
           "different, usually more specific, task. Fine-tuning updates model parameters on new training data, which "
           "can inject domain knowledge into large language models but is costly to repeat.")


class FakeSources:
    """Deterministic stand-in for the network-backed Sources class."""

    def __init__(self, fail_arxiv: bool = False) -> None:
        self.fail_arxiv = fail_arxiv
        self.calls: list[tuple] = []

    async def search_arxiv(self, query, max_results=5):
        self.calls.append(("arxiv", query))
        if self.fail_arxiv:
            raise SourceError("arXiv is down")
        if "fine-tuning" in query.lower():
            return [Document("arxiv", "Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs",
                             "https://arxiv.org/abs/2312.05934", "We compare unsupervised fine-tuning and retrieval-augmented "
                             "generation for knowledge injection in large language models and find retrieval consistently better.",
                             "2023-12-10", ["Oded Ovadia", "Menachem Brief"])]
        return [Document("arxiv", "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                         "https://arxiv.org/abs/2005.11401", "Large pre-trained language models store factual knowledge in "
                         "their parameters. We explore retrieval-augmented generation models which combine parametric and "
                         "non-parametric memory for language generation.", "2020-05-22", ["Patrick Lewis", "Ethan Perez"])][:max_results]

    async def search_wikipedia(self, query, max_results=5):
        self.calls.append(("wiki_search", query))
        if "fine-tuning" in query.lower():
            return [Document("wikipedia", "Fine-tuned universe", "https://en.wikipedia.org/wiki/Fine-tuned_universe", "cosmology snippet"),
                    Document("wikipedia", "Fine-tuning (deep learning)", "https://en.wikipedia.org/wiki/Fine-tuning_(deep_learning)", "adapting a model")]
        return [Document("wikipedia", "Retrieval-augmented generation", "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
                         "technique that enables large language models")]

    async def read_wikipedia(self, title, max_chars=6000):
        self.calls.append(("wiki_read", title))
        text = FT_TEXT if "Fine-tuning" in title else RAG_TEXT
        return Document("wikipedia", title, "https://en.wikipedia.org/wiki/" + title.replace(" ", "_"), text[:max_chars])

    async def fetch_url(self, url, max_chars=8000):
        self.calls.append(("fetch", url))
        if "internal" in url:
            raise SourceError("refusing to fetch non-public address 10.0.0.5")
        return Document("web", "Example page", url, "The example page explains retrieval-augmented generation pipelines in detail.")


@pytest.fixture
def fake_sources():
    return FakeSources()
