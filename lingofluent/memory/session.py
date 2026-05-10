from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Sequence

from lingofluent.llm.llm_base import ImagePart, Message, TextPart


_DEFAULT_DB = Path("data") / "sessions.db"
_MAX_HISTORY = 20


class SessionStore:
    """SQLite-backed store for conversation sessions.

    Schema
    ------
    sessions  — one row per conversation (chat_id + time window)
    messages  — one row per turn; images stored as BLOBs

    /reset closes the active session; old sessions are kept for audit.
    On the next message a new session is opened automatically.
    """

    def __init__(self, db_path: str | Path = _DEFAULT_DB) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id    TEXT    NOT NULL,
                created_at TEXT    NOT NULL DEFAULT (datetime('now')),
                closed_at  TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_chat
                ON sessions (chat_id, id);

            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL REFERENCES sessions(id),
                role       TEXT    NOT NULL,
                content    TEXT    NOT NULL,
                media      BLOB,
                media_mime TEXT,
                created_at TEXT    NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages (session_id, id);
        """)
        self._conn.commit()
        self._migrate()

    def _migrate(self) -> None:
        """Add media columns to existing tables created before this version."""
        existing = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(messages)").fetchall()
        }
        for col, definition in [("media", "BLOB"), ("media_mime", "TEXT")]:
            if col not in existing:
                self._conn.execute(f"ALTER TABLE messages ADD COLUMN {col} {definition}")
        self._conn.commit()

    # ---- session lifecycle -------------------------------------------------

    def get_or_create_session(self, chat_id: str | int) -> int:
        """Return the active session id for chat_id, creating one if needed."""
        chat_id = str(chat_id)
        row = self._conn.execute(
            "SELECT id FROM sessions WHERE chat_id = ? AND closed_at IS NULL ORDER BY id DESC LIMIT 1",
            (chat_id,),
        ).fetchone()
        if row:
            return row[0]
        cur = self._conn.execute(
            "INSERT INTO sessions (chat_id) VALUES (?)", (chat_id,)
        )
        self._conn.commit()
        return cur.lastrowid

    def close_session(self, chat_id: str | int) -> None:
        """Mark the active session as closed (for /reset). Keeps all messages."""
        self._conn.execute(
            """
            UPDATE sessions SET closed_at = datetime('now')
            WHERE chat_id = ? AND closed_at IS NULL
            """,
            (str(chat_id),),
        )
        self._conn.commit()

    # ---- message CRUD ------------------------------------------------------

    def add(self, session_id: int, role: str, content: str | Sequence) -> None:
        """Append one message to the session, including image bytes if present."""
        if isinstance(content, str):
            text = content
            media: bytes | None = None
            mime: str | None = None
        else:
            text = " ".join(p.text for p in content if isinstance(p, TextPart))
            image = next((p for p in content if isinstance(p, ImagePart)), None)
            if image is not None:
                media = image.data if image.data is not None else None
                mime = image.mime if image.data is not None else None
            else:
                media = None
                mime = None

        self._conn.execute(
            "INSERT INTO messages (session_id, role, content, media, media_mime) VALUES (?, ?, ?, ?, ?)",
            (session_id, role, text, media, mime),
        )
        self._conn.commit()

    def get_history(self, session_id: int, limit: int = _MAX_HISTORY) -> list[Message]:
        """Return the last `limit` messages in chronological order.

        Messages with a stored image are reconstructed as multimodal content.
        """
        rows = self._conn.execute(
            """
            SELECT role, content, media, media_mime FROM (
                SELECT id, role, content, media, media_mime
                FROM   messages
                WHERE  session_id = ?
                ORDER  BY id DESC
                LIMIT  ?
            )
            ORDER BY id ASC
            """,
            (session_id, limit),
        ).fetchall()

        messages = []
        for role, text, media, mime in rows:
            if media is not None:
                content = [TextPart(text=text), ImagePart(data=bytes(media), mime=mime or "image/jpeg")]
            else:
                content = text
            messages.append(Message(role=role, content=content))
        return messages

    # ---- audit helpers -----------------------------------------------------

    def list_sessions(self, chat_id: str | int) -> list[dict]:
        """Return all sessions for a chat_id (open and closed), newest first."""
        rows = self._conn.execute(
            """
            SELECT s.id, s.created_at, s.closed_at, COUNT(m.id) as msg_count
            FROM   sessions s
            LEFT   JOIN messages m ON m.session_id = s.id
            WHERE  s.chat_id = ?
            GROUP  BY s.id
            ORDER  BY s.id DESC
            """,
            (str(chat_id),),
        ).fetchall()
        return [
            {"id": r[0], "created_at": r[1], "closed_at": r[2], "msg_count": r[3]}
            for r in rows
        ]

    def get_session_history(self, session_id: int) -> list[Message]:
        """Return the full history of any session by its id (for audit)."""
        return self.get_history(session_id, limit=10_000)

    # ---- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "SessionStore":
        return self

    def __exit__(self, *_) -> None:
        self.close()


class Session:
    """Scoped view of a SessionStore for one session_id (e.g. a Telegram chat_id)."""

    def __init__(self, chat_id: str | int, store: SessionStore) -> None:
        self.chat_id = str(chat_id)
        self._store = store
        self._session_id: int = store.get_or_create_session(chat_id)

    @property
    def session_id(self) -> int:
        return self._session_id

    def add(self, role: str, content: str | Sequence) -> None:
        self._store.add(self._session_id, role, content)

    def history(self, limit: int = _MAX_HISTORY) -> list[Message]:
        return self._store.get_history(self._session_id, limit)

    def reset(self) -> None:
        """Close this session. The next message will open a new one."""
        self._store.close_session(self.chat_id)
