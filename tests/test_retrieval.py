import asyncio
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from carl.retrieval.search import BM25Search


def test_parent_search_applies_bm25_and_validates_document_schema(tmp_path, monkeypatch):
    class Searcher:
        def __init__(self, path):
            self.path = path

        def set_bm25(self, k1, b):
            self.parameters = (k1, b)

        def search(self, query, k):
            self.request = (query, k)
            return [SimpleNamespace(docid="one")]

        def doc(self, docid):
            return SimpleNamespace(raw=lambda: json.dumps({"title": "Title", "contents": "Evidence"}))

    module = ModuleType("pyserini.search.lucene")
    module.LuceneSearcher = Searcher
    monkeypatch.setitem(sys.modules, "pyserini", ModuleType("pyserini"))
    monkeypatch.setitem(sys.modules, "pyserini.search", ModuleType("pyserini.search"))
    monkeypatch.setitem(sys.modules, "pyserini.search.lucene", module)
    provider = BM25Search(tmp_path)
    assert provider.searcher.parameters == (0.9, 0.4)
    assert asyncio.run(provider("question")) == "[1] Title\nEvidence"
    assert provider.searcher.request == ("question", 3)
    provider.searcher.doc = lambda docid: SimpleNamespace(raw=lambda: '{"title":"No contents"}')
    with pytest.raises(ValueError, match="contents"):
        provider.lookup("question")
    with pytest.raises(ValueError, match="nonempty"):
        provider.lookup("")


def test_missing_index_fails_before_importing_optional_packages(tmp_path):
    with pytest.raises(FileNotFoundError):
        BM25Search(tmp_path / "not-present")
