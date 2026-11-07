"""SQLite persistence layer for VeriTag Face.

Every statement in this module uses parameter binding (``?`` placeholders), so
user supplied values are never interpolated into SQL text.  Connections are
short lived, which keeps the module safe to use from the server's per-client
threads without sharing a connection object between threads.

Typical use::

    db = DatabaseHandler(Path("data/veritag.db"))
    db.add_user(60, "Natan", "data/known_faces/60.jpg", embedding_bytes)
    db.add_log(60, "login", "Face and AprilTag matched")
    for entry in db.get_logs(limit=20):
        print(entry["event"], entry["created_at"])
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "veritag.db"
DEFAULT_TIMEOUT = 5.0


class DatabaseHandler:
    """Thread-safe access to the VeriTag SQLite database.

    Args:
        db_path: Location of the database file.  Parent directories are
            created automatically.
        timeout: SQLite busy timeout in seconds.  When another thread is
            writing, readers wait up to this long before failing.

    Attributes:
        db_path: Resolved path of the SQLite file.
    """

    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB_PATH,
        *,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """Create the database file and tables when they do not exist."""
        self.db_path = Path(db_path).expanduser().resolve()
        self._timeout = timeout
        self._lock = threading.RLock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        """Open a configured connection.

        Returns:
            A connection with ``sqlite3.Row`` rows, WAL journaling and a
            busy timeout suitable for multi-threaded access.
        """
        connection = sqlite3.connect(self.db_path, timeout=self._timeout)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        """Create the ``users`` and ``logs`` tables when they do not exist."""
        with self._lock, closing(self._connect()) as connection, connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    photo_path TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    registered_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    event TEXT NOT NULL,
                    details TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    # ------------------------------------------------------------------
    # Embedding conversion
    # ------------------------------------------------------------------
    @staticmethod
    def encode_embedding(embedding: np.ndarray | bytes | bytearray) -> bytes:
        """Convert a face embedding to the ``BLOB`` representation.

        Args:
            embedding: A NumPy array (typically ``float32``) or raw bytes.

        Returns:
            Little-endian ``float32`` bytes when given an array, or the bytes
            unchanged.

        Raises:
            TypeError: If *embedding* is not an array or bytes-like object.
        """
        if isinstance(embedding, np.ndarray):
            return np.ascontiguousarray(embedding, dtype=np.float32).tobytes()
        if isinstance(embedding, (bytes, bytearray)):
            return bytes(embedding)
        raise TypeError("Embedding must be a numpy array or bytes")

    @staticmethod
    def decode_embedding(blob: bytes | bytearray | memoryview) -> np.ndarray:
        """Convert a stored embedding blob to a ``float32`` row vector.

        Args:
            blob: Raw bytes read from the database.

        Returns:
            An array shaped ``(1, n)``, ready for ``FaceRecognizerSF.match``.
        """
        return np.frombuffer(bytes(blob), dtype=np.float32).reshape(1, -1).copy()

    # ------------------------------------------------------------------
    # Users
    # ------------------------------------------------------------------
    def user_exists(self, user_id: int) -> bool:
        """Return whether a user with *user_id* is registered.

        Args:
            user_id: Integer identifier shared with the AprilTag.
        """
        with self._lock, closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT 1 FROM users WHERE id = ? LIMIT 1", (int(user_id),)
            ).fetchone()
        return row is not None

    def add_user(
        self,
        user_id: int,
        name: str,
        photo_path: str | Path,
        embedding: np.ndarray | bytes | bytearray,
    ) -> bool:
        """Insert a new user and their face embedding.

        Args:
            user_id: Positive integer identifier.  It must be unique.
            name: Display name.
            photo_path: Path to the saved registration photo.
            embedding: NumPy feature vector or raw ``float32`` bytes.

        Returns:
            ``True`` when the user was inserted, ``False`` when the identifier
            already exists.

        Raises:
            ValueError: If *user_id* is not positive or *name* is empty.
        """
        user_id = int(user_id)
        name = str(name).strip()
        if user_id <= 0:
            raise ValueError("user_id must be a positive integer")
        if not name:
            raise ValueError("name must not be empty")

        blob = self.encode_embedding(embedding)
        try:
            with self._lock, closing(self._connect()) as connection, connection:
                connection.execute(
                    "INSERT INTO users (id, name, photo_path, embedding) "
                    "VALUES (?, ?, ?, ?)",
                    (user_id, name, str(photo_path), blob),
                )
        except sqlite3.IntegrityError:
            return False
        return True

    def add_embedding(
        self,
        user_id: int,
        embedding: np.ndarray | bytes | bytearray,
    ) -> bool:
        """Replace the stored face embedding of an existing user.

        Args:
            user_id: Identifier of the user to update.
            embedding: New NumPy feature vector or raw ``float32`` bytes.

        Returns:
            ``True`` if a row was updated, ``False`` if the user is unknown.
        """
        blob = self.encode_embedding(embedding)
        with self._lock, closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                "UPDATE users SET embedding = ? WHERE id = ?",
                (blob, int(user_id)),
            )
        return cursor.rowcount > 0

    def get_user(self, user_id: int) -> dict[str, Any] | None:
        """Fetch one user by identifier.

        Args:
            user_id: Identifier to look up.

        Returns:
            A dictionary with ``id``, ``name``, ``photo_path``, ``embedding``
            (raw bytes) and ``registered_at``, or ``None`` when not found.
        """
        with self._lock, closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT id, name, photo_path, embedding, registered_at "
                "FROM users WHERE id = ?",
                (int(user_id),),
            ).fetchone()
        return dict(row) if row is not None else None

    def get_users(self) -> list[dict[str, Any]]:
        """Return every registered user ordered by identifier.

        Returns:
            A list of dictionaries in the same shape as :meth:`get_user`.
        """
        with self._lock, closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT id, name, photo_path, embedding, registered_at "
                "FROM users ORDER BY id"
            ).fetchall()
        return [dict(row) for row in rows]

    def verify_credentials(
        self,
        user_id: int,
        name: str | None = None,
    ) -> dict[str, Any] | None:
        """Check that a user exists and optionally matches a name.

        In this project the "credential" is the numeric identifier shared by
        the face recognition result and the AprilTag, optionally confirmed by
        the registered name.

        Args:
            user_id: Identifier to verify.
            name: When given, the registered name must match
                case-insensitively.

        Returns:
            The matching user record, or ``None`` when verification fails.
        """
        user = self.get_user(user_id)
        if user is None:
            return None
        if name is not None and user["name"].casefold() != str(name).strip().casefold():
            return None
        return user

    def delete_user(self, user_id: int, *, purge_logs: bool = False) -> bool:
        """Remove a user account.

        Args:
            user_id: Identifier of the user to delete.
            purge_logs: When ``True`` also erase that user's audit history.
                The default keeps it for accountability and appends a
                ``user_deleted`` event instead.

        Returns:
            ``True`` if a user row was deleted, otherwise ``False``.
        """
        user_id = int(user_id)
        with self._lock, closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                "DELETE FROM users WHERE id = ?", (user_id,)
            )
            deleted = cursor.rowcount > 0
            if deleted:
                if purge_logs:
                    connection.execute(
                        "DELETE FROM logs WHERE user_id = ?", (user_id,)
                    )
                else:
                    connection.execute(
                        "INSERT INTO logs (user_id, event, details) "
                        "VALUES (?, ?, ?)",
                        (user_id, "user_deleted", "User account deleted"),
                    )
        return deleted

    # ------------------------------------------------------------------
    # Authentication logs
    # ------------------------------------------------------------------
    def add_log(
        self,
        user_id: int | None,
        event: str,
        details: str = "",
    ) -> int:
        """Append an authentication event to the audit log.

        Args:
            user_id: Related user identifier, or ``None`` for anonymous
                events such as a failed tag match.
            event: Short event name, for example ``"login"`` or ``"denied"``.
            details: Human readable context.

        Returns:
            The row identifier of the inserted log entry.

        Raises:
            ValueError: If *event* is empty.
        """
        event = str(event).strip()
        if not event:
            raise ValueError("event must not be empty")

        with self._lock, closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                "INSERT INTO logs (user_id, event, details) VALUES (?, ?, ?)",
                (
                    None if user_id is None else int(user_id),
                    event,
                    str(details),
                ),
            )
            log_id = int(cursor.lastrowid or 0)
        return log_id

    def get_logs(
        self,
        limit: int = 100,
        *,
        user_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return recent audit log entries, newest first.

        Args:
            limit: Maximum number of rows to return, clamped to ``1..1000``.
            user_id: When given, only return entries for that user.

        Returns:
            A list of dictionaries with ``id``, ``user_id``, ``event``,
            ``details`` and ``created_at``.
        """
        limit = max(1, min(int(limit), 1000))
        with self._lock, closing(self._connect()) as connection:
            if user_id is None:
                rows = connection.execute(
                    "SELECT id, user_id, event, details, created_at "
                    "FROM logs ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT id, user_id, event, details, created_at "
                    "FROM logs WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                    (int(user_id), limit),
                ).fetchall()
        return [dict(row) for row in rows]
