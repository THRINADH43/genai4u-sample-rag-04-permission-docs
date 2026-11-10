"""Documents and their chunks, in memory, guarded by a lock. Every chunk carries the groups
allowed to read its document."""

import re
import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    n: int
    text: str
    groups: frozenset

    @property
    def cite(self):
        return f"[{self.doc_id}#{self.n}]"


def split_chunks(text, size=40):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, current = [], []
    for s in sentences:
        current.append(s)
        if sum(len(x.split()) for x in current) >= size:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks


class DocumentStore:
    def __init__(self):
        self._chunks: dict[str, list[Chunk]] = {}
        self._lock = threading.Lock()
        self.version = 0  # bumped on every change, so search indexes know to rebuild

    def upsert(self, doc_id, text, groups):
        """Add or replace a document. Old chunks are removed in the same step."""
        chunks = [Chunk(doc_id, i, t, frozenset(groups)) for i, t in enumerate(split_chunks(text), start=1)]
        with self._lock:
            self._chunks[doc_id] = chunks
            self.version += 1

    def delete(self, doc_id):
        with self._lock:
            existed = self._chunks.pop(doc_id, None) is not None
            self.version += 1
        return existed

    def visible_chunks(self, groups):
        """Only chunks of documents shared with at least one of the user's groups. This is the
        permission filter: search never sees anything else."""
        groups = set(groups)
        with self._lock:
            return [c for chunks in self._chunks.values() for c in chunks if c.groups & groups]

    def doc_ids(self):
        with self._lock:
            return sorted(self._chunks)
