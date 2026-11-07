"""VeriTag Face TCP server.

The server owns the SQLite database and the face-recognition models.  Each
client is handled in its own daemon thread; a client that disconnects
abruptly only terminates its own thread, never the accept loop.

Protocol frames are decoded by :class:`protocol.ProtocolHandler`, which is
optionally AES-GCM encrypted when ``VERITAG_KEY`` is set.  Malformed frames
raise :class:`protocol.ProtocolError` and are answered with a best-effort
error response before the connection is closed.
"""

from __future__ import annotations

import base64
import binascii
import logging
import os
import socket
import threading
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np

from database import DatabaseHandler
from protocol import ProtocolError, ProtocolHandler, get_default_handler
from vision import FaceDetector, FaceRecognizer

HOST: Final = os.environ.get("VERITAG_HOST", "0.0.0.0")
PORT: Final = int(os.environ.get("VERITAG_PORT", "5000"))
CLIENT_TIMEOUT: Final = float(os.environ.get("VERITAG_CLIENT_TIMEOUT", "120"))

ROOT: Final = Path(__file__).resolve().parent.parent
DATA_DIR: Final = ROOT / "data"
DB_PATH: Final = Path(os.environ.get("VERITAG_DB", DATA_DIR / "veritag.db"))
PHOTO_DIR: Final = DATA_DIR / "known_faces"
MODELS_DIR: Final = ROOT / "models"
ALLOWED_SUFFIXES: Final = {".jpg", ".jpeg", ".png"}
MAX_PHOTO_BYTES: Final = 8 * 1024 * 1024
MAX_NAME_LENGTH: Final = 100

logger = logging.getLogger("veritag.server")

_detector: FaceDetector | None = None
_recognizer: FaceRecognizer | None = None
_models_lock = threading.Lock()

_database: DatabaseHandler | None = None
_database_lock = threading.Lock()


def get_database() -> DatabaseHandler:
    """Return the process-wide database handler.

    The handler is created on first use so importing this module in tests does
    not immediately touch the real database file.
    """
    global _database
    if _database is None:
        with _database_lock:
            if _database is None:
                _database = DatabaseHandler(DB_PATH)
    return _database


def get_models() -> tuple[FaceDetector, FaceRecognizer]:
    """Load the YuNet and SFace models once and return them.

    Returns:
        ``(detector, recognizer)`` instances shared by all client threads.
        Each instance is internally locked, so concurrent inference is safe.
    """
    global _detector, _recognizer
    if _detector is None or _recognizer is None:
        with _models_lock:
            if _detector is None:
                _detector = FaceDetector(MODELS_DIR / "yunet.onnx")
            if _recognizer is None:
                _recognizer = FaceRecognizer(MODELS_DIR / "sface.onnx")
    return _detector, _recognizer


def init_db() -> DatabaseHandler:
    """Create the data directories/schema and return the database handler."""
    PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    return get_database()


def extract_embedding(image: np.ndarray) -> np.ndarray | None:
    """Return the embedding of the largest face in *image*.

    Args:
        image: BGR image decoded from the uploaded registration photo.

    Returns:
        A ``float32`` feature row, or ``None`` when no face is detected.
    """
    detector, recognizer = get_models()
    face = detector.largest_face(detector.detect(image))
    if face is None:
        return None
    return recognizer.embedding(image, face)


def _fail(message: str) -> dict[str, Any]:
    """Build a standard failure response."""
    return {"ok": False, "message": message}


def handle_register(message: dict[str, Any]) -> dict[str, Any]:
    """Validate and store a new user registration.

    Args:
        message: Request containing ``id``, ``name``, ``filename`` and a
            base64 encoded ``photo``.

    Returns:
        A protocol response dictionary.
    """
    try:
        user_id = int(message.get("id"))
    except (TypeError, ValueError):
        return _fail("ID must be an integer.")

    name = str(message.get("name", "")).strip()
    filename = str(message.get("filename", "photo.jpg"))
    suffix = Path(filename).suffix.lower()

    if user_id <= 0:
        return _fail("ID must be a positive integer.")
    if not name:
        return _fail("Name is required.")
    if len(name) > MAX_NAME_LENGTH:
        return _fail(f"Name must be at most {MAX_NAME_LENGTH} characters.")
    if suffix not in ALLOWED_SUFFIXES:
        return _fail("Photo must be JPG or PNG.")

    encoded_photo = message.get("photo", "")
    if not isinstance(encoded_photo, str) or not encoded_photo:
        return _fail("Photo data is invalid.")
    try:
        raw = base64.b64decode(encoded_photo, validate=True)
    except (binascii.Error, ValueError, TypeError):
        return _fail("Photo data is invalid.")

    if not raw or len(raw) > MAX_PHOTO_BYTES:
        return _fail("Photo is empty or too large.")

    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return _fail("Could not read the photo.")

    try:
        embedding = extract_embedding(image)
    except Exception:
        logger.exception("Face extraction failed for user %s", user_id)
        return _fail("Could not process the photo.")
    if embedding is None:
        return _fail("No face detected in that photo.")

    database = get_database()
    if database.user_exists(user_id):
        return _fail(f"ID {user_id} is already registered.")

    destination = PHOTO_DIR / f"{user_id}{suffix}"
    try:
        for old in PHOTO_DIR.glob(f"{user_id}.*"):
            old.unlink(missing_ok=True)

        PHOTO_DIR.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), image):
            return _fail("Could not save the photo.")

        try:
            photo_path = destination.relative_to(ROOT)
        except ValueError:
            # Keeps the server usable when VERITAG_DB/photo paths are moved
            # outside the project root (tests, custom deployments).
            photo_path = destination
        if not database.add_user(user_id, name, photo_path, embedding):
            destination.unlink(missing_ok=True)
            return _fail(f"ID {user_id} is already registered.")
    except Exception:
        try:
            destination.unlink(missing_ok=True)
        except OSError:
            pass
        logger.exception("Registration failed for user %s", user_id)
        return _fail("Could not register the user.")

    database.add_log(user_id, "register", f"Registered {name}")
    logger.info("Registered user %s (%s)", user_id, name)
    return {"ok": True, "message": f"Registered {name} (ID {user_id})."}


def handle_get_users() -> dict[str, Any]:
    """Return every registered user and their base64 encoded embedding."""
    users = [
        {
            "id": int(user["id"]),
            "name": user["name"],
            "embedding": base64.b64encode(user["embedding"]).decode("ascii"),
        }
        for user in get_database().get_users()
    ]
    return {"ok": True, "users": users}


def handle_verify(message: dict[str, Any]) -> dict[str, Any]:
    """Verify that a recognized face ID matches the scanned AprilTag ID.

    Args:
        message: Request containing integer ``face_id`` and ``tag_id``.

    Returns:
        On success ``{"ok": True, "name": ..., "message": ...}``; otherwise a
        failure response describing why access was denied.
    """
    try:
        face_id = int(message.get("face_id"))
        tag_id = int(message.get("tag_id"))
    except (TypeError, ValueError):
        return _fail("Invalid verification request.")

    database = get_database()
    if face_id != tag_id:
        database.add_log(
            face_id,
            "denied",
            f"Face ID {face_id} did not match AprilTag ID {tag_id}",
        )
        return _fail("Face and AprilTag ID do not match.")

    user = database.verify_credentials(face_id)
    if user is None:
        database.add_log(face_id, "denied", "Unknown user identifier")
        return _fail("No registered user has that ID.")

    database.add_log(face_id, "login", "Face and AprilTag matched")
    logger.info("Access granted to %s (ID %s)", user["name"], face_id)
    return {
        "ok": True,
        "name": user["name"],
        "message": f"Successfully logged in as {user['name']}.",
    }


def _dispatch(message: dict[str, Any]) -> dict[str, Any]:
    """Route one decoded request to the matching handler.

    Args:
        message: Request object with an ``action`` key.

    Returns:
        A response dictionary.  Unknown actions and unexpected errors are
        converted to a failure response so the connection stays usable.
    """
    action = message.get("action")

    try:
        if action == "hello":
            return {"ok": True, "command": "open_interface"}
        if action == "register":
            return handle_register(message)
        if action == "get_users":
            return handle_get_users()
        if action == "verify":
            return handle_verify(message)
    except Exception:
        # A bug in one request must not take down the client thread silently.
        logger.exception("Unhandled error while processing action %r", action)
        return _fail("Internal server error.")

    return _fail("Unknown action.")


# Public alias used by tests and external scripts.
dispatch = _dispatch


def handle_client(
    connection: socket.socket,
    address: tuple[str, int],
    handler: ProtocolHandler | None = None,
) -> None:
    """Serve requests for one client until it disconnects.

    The function catches protocol errors, resets, broken pipes and generic
    socket errors.  In every failure case only this client's thread ends; the
    server's accept loop continues running.

    Args:
        connection: Accepted client socket.
        address: Peer address used for logging.
        handler: Optional protocol handler.  Defaults to the process handler,
            which honours ``VERITAG_KEY``.
    """
    try:
        handler = handler or get_default_handler()
    except ProtocolError as exc:
        logger.error("Cannot initialise the protocol handler: %s", exc)
        connection.close()
        return

    with connection:
        if CLIENT_TIMEOUT > 0:
            connection.settimeout(CLIENT_TIMEOUT)
        logger.info("Client connected from %s:%s", *address)

        try:
            while True:
                try:
                    message = handler.recv(connection)
                except ProtocolError as exc:
                    logger.warning("Protocol error from %s: %s", address, exc)
                    try:
                        handler.send(connection, _fail("Malformed or unauthorised request."))
                    except (OSError, ProtocolError):
                        pass
                    break
                except ConnectionResetError:
                    logger.info("Client %s reset the connection", address)
                    break
                except BrokenPipeError:
                    logger.info("Client %s closed the pipe", address)
                    break
                except (ConnectionError, TimeoutError, socket.timeout):
                    logger.info("Client %s disconnected", address)
                    break
                except OSError as exc:
                    logger.warning("Socket error while reading from %s: %s", address, exc)
                    break

                response = _dispatch(message)

                try:
                    handler.send(connection, response)
                except ProtocolError as exc:
                    logger.warning("Could not encode response to %s: %s", address, exc)
                    break
                except ConnectionResetError:
                    logger.info("Client %s reset the connection while replying", address)
                    break
                except BrokenPipeError:
                    logger.info("Client %s closed the pipe while replying", address)
                    break
                except (ConnectionError, TimeoutError, socket.timeout):
                    logger.info("Client %s disconnected while replying", address)
                    break
                except OSError as exc:
                    logger.warning("Socket error while writing to %s: %s", address, exc)
                    break

        finally:
            logger.info("Client disconnected: %s", address)


def main() -> None:
    """Initialise the database/models and run the TCP accept loop."""
    logging.basicConfig(
        level=os.environ.get("VERITAG_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s [%(threadName)s] %(message)s",
    )

    database = init_db()
    get_models()

    try:
        protocol_handler = get_default_handler()
    except ProtocolError as exc:
        logger.error("Invalid VERITAG_KEY: %s", exc)
        raise SystemExit(2) from exc

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(128)

        logger.info("Socket server listening on %s:%s", HOST, PORT)
        logger.info("Database: %s", database.db_path)
        logger.info(
            "Transport: %s",
            "AES-GCM" if protocol_handler.encrypted else "plaintext",
        )

        try:
            while True:
                try:
                    connection, address = server.accept()
                except OSError as exc:
                    logger.warning("accept() failed: %s", exc)
                    continue

                client_thread = threading.Thread(
                    target=handle_client,
                    args=(connection, address),
                    name=f"client-{address[0]}:{address[1]}",
                    daemon=True,
                )
                client_thread.start()
                logger.info("Active threads: %d", threading.active_count())
        except KeyboardInterrupt:
            logger.info("Server interrupted, shutting down.")


if __name__ == "__main__":
    main()
