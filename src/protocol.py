"""Length-prefixed, optionally AES-GCM encrypted message protocol.

Wire format (integers are big-endian)::

    +----------------+-------------+-----------------------------+
    | 4-byte length  | 1-byte flag | payload (length - 1 bytes)  |
    +----------------+-------------+-----------------------------+

``flag == 0`` means the payload is UTF-8 JSON.  ``flag == 1`` means the
payload is ``12-byte nonce || AES-GCM ciphertext + tag``.

Encryption is optional: it is enabled automatically when the ``VERITAG_KEY``
environment variable contains a base64 or hex encoded 16/24/32-byte key.  If
the variable is unset, traffic stays plaintext so the local demo keeps
working.  When a handler has a key it rejects plaintext frames, so both ends
must be configured with the same key.

All parsing errors are converted to :class:`ProtocolError`; callers (for
example the server worker threads) can catch that single exception and close
only the affected connection.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import socket
import struct
import threading
from collections.abc import Mapping
from typing import Any, Final

MAX_MESSAGE_BYTES: Final = 64 * 1024 * 1024
_HANDLER_VERSION: Final = 1
_FLAG_PLAINTEXT: Final = 0
_FLAG_AESGCM: Final = 1
_AAD: Final = b"veritag-protocol-v1"
_NONCE_BYTES: Final = 12
_TAG_BYTES: Final = 16
_VALID_KEY_LENGTHS: Final = (16, 24, 32)


class ProtocolError(Exception):
    """Raised when a frame is malformed, too large, or cannot be decrypted."""


def _decode_key(raw: str) -> bytes:
    """Decode a base64/URL-safe-base64/hex key from ``VERITAG_KEY``.

    Args:
        raw: Encoded key material.

    Returns:
        A 16, 24 or 32 byte AES key.

    Raises:
        ProtocolError: If the value cannot be decoded to a valid AES key.
    """
    padded = raw + "=" * (-len(raw) % 4)
    candidates: list[bytes] = []

    # Hex keys and base64 keys can look alike (hex uses only base64 alphabet
    # characters).  Treat a 32/48/64 character all-hex value as hex first so a
    # 16-byte key written as hex is not silently decoded as 24 bytes of base64.
    if len(raw) in (32, 48, 64) and all(
        character in "0123456789abcdefABCDEF" for character in raw
    ):
        candidates.append(bytes.fromhex(raw))

    for decoder in (
        lambda value: base64.b64decode(value, validate=True),
        lambda value: base64.b64decode(value, altchars=b"-_", validate=True),
    ):
        try:
            candidates.append(decoder(padded))
        except (binascii.Error, ValueError):
            continue

    try:
        candidates.append(bytes.fromhex(raw))
    except ValueError:
        pass

    for key in candidates:
        if len(key) in _VALID_KEY_LENGTHS:
            return key

    raise ProtocolError(
        "VERITAG_KEY must be a base64, URL-safe base64 or hex encoded "
        "16, 24 or 32 byte key"
    )


class ProtocolHandler:
    """Encode and decode VeriTag frames over a stream socket.

    Args:
        key: Optional raw AES key.  ``None`` disables encryption.
        timeout: Optional socket timeout applied by :meth:`request` only.
    """

    def __init__(self, key: bytes | None = None, *, timeout: float | None = None) -> None:
        """Create a handler, optionally enabling AES-GCM with *key*."""
        self._aead: Any | None = None
        self._timeout = timeout

        if key is not None:
            if len(key) not in _VALID_KEY_LENGTHS:
                raise ValueError("AES key must be 16, 24 or 32 bytes")
            try:
                from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            except ImportError as exc:  # pragma: no cover - depends on environment
                raise ProtocolError(
                    "The 'cryptography' package is required when VERITAG_KEY is set"
                ) from exc
            self._aead = AESGCM(key)

    @classmethod
    def from_env(cls, variable: str = "VERITAG_KEY") -> "ProtocolHandler":
        """Build a handler from an environment variable.

        Args:
            variable: Name of the environment variable to read.

        Returns:
            An encrypted handler when the variable is set, otherwise a
            plaintext handler.
        """
        raw = os.environ.get(variable, "").strip()
        if not raw:
            return cls()
        return cls(_decode_key(raw))

    @property
    def encrypted(self) -> bool:
        """Whether outgoing frames are encrypted."""
        return self._aead is not None

    def send(self, sock: socket.socket, message: Mapping[str, Any]) -> None:
        """Serialise *message* and write one frame to *sock*.

        Args:
            sock: Connected stream socket.
            message: JSON-serialisable mapping.

        Raises:
            ProtocolError: If the message cannot be serialised or is too big.
            OSError: If the socket write fails.
        """
        try:
            document = dict(message)
            plaintext = json.dumps(document, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ProtocolError("Message is not JSON serialisable") from exc

        if self._aead is not None:
            nonce = os.urandom(_NONCE_BYTES)
            payload = nonce + self._aead.encrypt(nonce, plaintext, _AAD)
            flag = _FLAG_AESGCM
        else:
            payload = plaintext
            flag = _FLAG_PLAINTEXT

        frame_length = len(payload) + 1
        if frame_length > MAX_MESSAGE_BYTES:
            raise ProtocolError("Message too large")

        sock.sendall(struct.pack("!I", frame_length) + bytes((flag,)) + payload)

    def recv(self, sock: socket.socket) -> dict[str, Any]:
        """Read and decode one frame from *sock*.

        Args:
            sock: Connected stream socket.

        Returns:
            The decoded JSON object.

        Raises:
            ProtocolError: If the frame is malformed, oversized, encrypted
                without a configured key, or cannot be decrypted.
            ConnectionError: If the peer closes the connection mid-frame.
            OSError: If the socket read fails.
        """
        header = self._recv_exact(sock, 4)
        try:
            (frame_length,) = struct.unpack("!I", header)
        except struct.error as exc:  # pragma: no cover - defensive
            raise ProtocolError("Invalid frame header") from exc

        if frame_length < 1 or frame_length > MAX_MESSAGE_BYTES:
            raise ProtocolError(f"Invalid frame length: {frame_length}")

        body = self._recv_exact(sock, frame_length)
        flag, payload = body[0], body[1:]

        if flag == _FLAG_PLAINTEXT:
            if self._aead is not None:
                raise ProtocolError("Plaintext frame rejected: encryption is required")
            plaintext = payload

        elif flag == _FLAG_AESGCM:
            if self._aead is None:
                raise ProtocolError("Encrypted frame received but no key is configured")
            if len(payload) < _NONCE_BYTES + _TAG_BYTES:
                raise ProtocolError("Encrypted payload is too short")
            nonce = payload[:_NONCE_BYTES]
            ciphertext = payload[_NONCE_BYTES:]
            try:
                plaintext = self._aead.decrypt(nonce, ciphertext, _AAD)
            except Exception as exc:  # InvalidTag and backend errors
                raise ProtocolError(
                    "Could not decrypt frame (wrong key or corrupted data)"
                ) from exc

        else:
            raise ProtocolError(f"Unknown frame flag: {flag}")

        try:
            message = json.loads(plaintext.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProtocolError("Frame does not contain valid JSON") from exc

        if not isinstance(message, dict):
            raise ProtocolError("Frame payload must be a JSON object")
        return message

    @staticmethod
    def _recv_exact(sock: socket.socket, size: int) -> bytes:
        """Read exactly *size* bytes or raise a connection error."""
        chunks: list[bytes] = []
        remaining = size
        while remaining > 0:
            try:
                chunk = sock.recv(remaining)
            except TimeoutError as exc:
                raise ConnectionError("Connection timed out while receiving a frame") from exc
            if not chunk:
                raise ConnectionError("Connection closed while receiving a frame")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def request(
        self,
        host: str,
        port: int,
        message: Mapping[str, Any],
        timeout: float = 15.0,
    ) -> dict[str, Any]:
        """Open a short-lived connection, send one request and read the reply.

        Args:
            host: Server hostname or IP address.
            port: Server TCP port.
            message: Request payload.
            timeout: Connection and read timeout in seconds.

        Returns:
            The server's decoded response.
        """
        with socket.create_connection((host, port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            self.send(sock, message)
            return self.recv(sock)


_default_handler: ProtocolHandler | None = None
_default_handler_lock = threading.Lock()


def get_default_handler() -> ProtocolHandler:
    """Return the process-wide handler configured from ``VERITAG_KEY``."""
    global _default_handler
    if _default_handler is None:
        with _default_handler_lock:
            if _default_handler is None:
                _default_handler = ProtocolHandler.from_env()
    return _default_handler


def reset_default_handler() -> None:
    """Forget the cached handler (used by tests after changing ``VERITAG_KEY``)."""
    global _default_handler
    with _default_handler_lock:
        _default_handler = None


def send_message(sock: socket.socket, message: Mapping[str, Any]) -> None:
    """Backward-compatible wrapper around :meth:`ProtocolHandler.send`."""
    get_default_handler().send(sock, message)


def recv_message(sock: socket.socket) -> dict[str, Any]:
    """Backward-compatible wrapper around :meth:`ProtocolHandler.recv`."""
    return get_default_handler().recv(sock)


def request(
    host: str,
    port: int,
    message: Mapping[str, Any],
    timeout: float = 15.0,
) -> dict[str, Any]:
    """Backward-compatible one-shot request helper."""
    return get_default_handler().request(host, port, message, timeout=timeout)
