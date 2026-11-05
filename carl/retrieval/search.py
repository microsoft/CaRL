"""Local Wikipedia BM25; the index must be provisioned before inference."""
import asyncio
import json
import math
from pathlib import Path
import threading


class BM25Search:
    def __init__(self, index, top_k=3, k1=0.9, b=0.4):
        if not Path(index).is_dir():
            raise FileNotFoundError(f"Missing Lucene index: {index}")
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if not math.isfinite(k1) or k1 <= 0 or not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("BM25 requires finite k1 > 0 and b in [0,1]")
        from pyserini.search.lucene import LuceneSearcher
        self.searcher = LuceneSearcher(str(index))
        self.searcher.set_bm25(k1, b)
        self.top_k = top_k
        self._lock = threading.Lock()

    def lookup(self, query):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search requires a nonempty string")
        with self._lock:
            hits = self.searcher.search(query, k=self.top_k)
            passages = []
            for rank, hit in enumerate(hits, 1):
                document = json.loads(self.searcher.doc(hit.docid).raw())
                if not isinstance(document.get("contents"), str):
                    raise ValueError("Lucene document lacks a text contents field")
                passages.append(f"[{rank}] {document.get('title', '')}\n{document['contents']}")
        return "\n\n".join(passages)

    async def __call__(self, query):
        return await asyncio.to_thread(self.lookup, query)
