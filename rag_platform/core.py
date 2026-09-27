"""Deterministic local retrieval and evidence extraction.

The local mode needs no model credentials and makes citation behavior testable.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


@dataclass(frozen=True)
class Hit:
    source: str
    passage: str
    score: float


class DocumentStore:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS passages ("
            "source TEXT NOT NULL, ordinal INTEGER NOT NULL, body TEXT NOT NULL, "
            "PRIMARY KEY (source, ordinal))"
        )
        self.db.commit()

    def ingest(self, source: str, text: str, *, chunk_words: int = 120) -> int:
        if not source.strip() or not text.strip():
            raise ValueError("source and text must be nonempty")
        if chunk_words < 10 or chunk_words > 1000:
            raise ValueError("chunk_words must be between 10 and 1000")
        words = text.split()
        chunks = [" ".join(words[i : i + chunk_words]) for i in range(0, len(words), chunk_words)]
        with self.db:
            self.db.execute("DELETE FROM passages WHERE source = ?", (source,))
            self.db.executemany(
                "INSERT INTO passages (source, ordinal, body) VALUES (?, ?, ?)",
                [(source, i, chunk) for i, chunk in enumerate(chunks)],
            )
        return len(chunks)

    def search(self, query: str, *, limit: int = 5) -> list[Hit]:
        terms = _tokens(query)
        if not terms:
            return []
        rows = self.db.execute("SELECT source, body FROM passages").fetchall()
        hits = []
        for source, body in rows:
            overlap = len(terms & _tokens(body))
            if overlap:
                hits.append(Hit(source, body, overlap / len(terms)))
        return sorted(hits, key=lambda h: (-h.score, h.source, h.passage))[:limit]

    def answer(self, query: str) -> dict:
        hits = self.search(query)
        if not hits:
            return {"answer": "No supporting passage found.", "citations": []}
        # Extractive baseline: every sentence shown comes from a cited passage.
        result = []
        for hit in hits[:3]:
            sentences = re.split(r"(?<=[.!?])\s+", hit.passage)
            best = max(sentences, key=lambda s: len(_tokens(query) & _tokens(s)))
            result.append({"source": hit.source, "excerpt": best, "score": hit.score})
        return {
            "answer": " ".join(f"{r['excerpt']} [{i}]" for i, r in enumerate(result, 1)),
            "citations": result,
        }

    def close(self) -> None:
        self.db.close()
