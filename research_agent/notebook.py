"""Persistent research notebook (SQLite + FTS5) shared by the MCP server and the agent."""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    topic       TEXT NOT NULL,
    content     TEXT NOT NULL,
    source_url  TEXT,
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(topic, content, content='notes', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS notes_ai AFTER INSERT ON notes BEGIN
    INSERT INTO notes_fts(rowid, topic, content) VALUES (new.id, new.topic, new.content);
END;
CREATE TRIGGER IF NOT EXISTS notes_ad AFTER DELETE ON notes BEGIN
    INSERT INTO notes_fts(notes_fts, rowid, topic, content) VALUES ('delete', old.id, old.topic, old.content);
END;
"""


@dataclass
class Note:
    id: int
    topic: str
    content: str
    source_url: str | None
    created_at: str

    def to_dict(self) -> dict:
        return asdict(self)


class Notebook:
    def __init__(self, path: str | Path = ":memory:") -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(SCHEMA)

    def add(self, topic: str, content: str, source_url: str | None = None) -> Note:
        topic, content = topic.strip(), content.strip()
        if not topic or not content:
            raise ValueError("topic and content must not be empty")
        with self._lock, self._db:
            cur = self._db.execute("INSERT INTO notes (topic, content, source_url) VALUES (?, ?, ?)", (topic, content, source_url))
            row = self._db.execute("SELECT * FROM notes WHERE id = ?", (cur.lastrowid,)).fetchone()
        return Note(**dict(row))

    def search(self, query: str, limit: int = 10) -> list[Note]:
        # Quote each term so user input can't inject FTS5 syntax.
        terms = [t.replace('"', "") for t in query.split() if t.strip('"')]
        if not terms:
            return self.recent(limit)
        match = " OR ".join(f'"{t}"' for t in terms)
        with self._lock:
            rows = self._db.execute(
                "SELECT n.* FROM notes_fts f JOIN notes n ON n.id = f.rowid WHERE notes_fts MATCH ? ORDER BY rank LIMIT ?",
                (match, limit),
            ).fetchall()
        return [Note(**dict(r)) for r in rows]

    def recent(self, limit: int = 10) -> list[Note]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM notes ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [Note(**dict(r)) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._db.close()
