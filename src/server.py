import base64
import socket
import sqlite3
import threading
from pathlib import Path

import cv2
import numpy as np

from protocol import recv_message, send_message

HOST = "0.0.0.0"
PORT = 5000

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "limelight.db"
PHOTO_DIR = DATA_DIR / "known_faces"
MODELS_DIR = ROOT / "models"
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png"}

_detector = None
_recognizer = None


def get_models():
    global _detector, _recognizer
    if _detector is None:
        _detector = cv2.FaceDetectorYN.create(
            str(MODELS_DIR / "yunet.onnx"), "", (320, 320), 0.6, 0.3, 5000
        )
        _recognizer = cv2.FaceRecognizerSF.create(
            str(MODELS_DIR / "sface.onnx"), ""
        )
    return _detector, _recognizer


def init_db():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
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
        conn.commit()


def extract_embedding(image):
    detector, recognizer = get_models()
    height, width = image.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(image)
    if faces is None or len(faces) == 0:
        return None
    face = max(faces, key=lambda f: f[2] * f[3])
    aligned = recognizer.alignCrop(image, face)
    return recognizer.feature(aligned)


def handle_register(message):
    try:
        user_id = int(message.get("id"))
    except (TypeError, ValueError):
        return {"ok": False, "message": "ID must be an integer."}

    name = str(message.get("name", "")).strip()
    filename = str(message.get("filename", "photo.jpg"))
    suffix = Path(filename).suffix.lower()

    if user_id <= 0:
        return {"ok": False, "message": "ID must be a positive integer."}
    if not name:
        return {"ok": False, "message": "Name is required."}
    if suffix not in ALLOWED_SUFFIXES:
        return {"ok": False, "message": "Photo must be JPG or PNG."}

    try:
        raw = base64.b64decode(message.get("photo", ""), validate=True)
    except Exception:
        return {"ok": False, "message": "Photo data is invalid."}

    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return {"ok": False, "message": "Could not read the photo."}

    embedding = extract_embedding(image)
    if embedding is None:
        return {"ok": False, "message": "No face detected in that photo."}

    destination = PHOTO_DIR / f"{user_id}{suffix}"

    try:
        with sqlite3.connect(DB_PATH) as conn:
            exists = conn.execute(
                "SELECT 1 FROM users WHERE id = ?", (user_id,)
            ).fetchone()
            if exists:
                return {"ok": False, "message": f"ID {user_id} is already registered."}

            for old in PHOTO_DIR.glob(f"{user_id}.*"):
                old.unlink(missing_ok=True)

            if not cv2.imwrite(str(destination), image):
                return {"ok": False, "message": "Could not save the photo."}

            try:
                conn.execute(
                    "INSERT INTO users (id, name, photo_path, embedding) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        user_id,
                        name,
                        str(destination.relative_to(ROOT)),
                        embedding.astype(np.float32).tobytes(),
                    ),
                )
                conn.commit()
            except Exception:
                destination.unlink(missing_ok=True)
                raise
    except Exception as exc:
        return {"ok": False, "message": f"Could not register: {exc}"}

    return {"ok": True, "message": f"Registered {name} (ID {user_id})."}


def handle_get_users():
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT id, name, embedding FROM users ORDER BY id"
        ).fetchall()

    users = [
        {
            "id": int(user_id),
            "name": name,
            "embedding": base64.b64encode(blob).decode("ascii"),
        }
        for user_id, name, blob in rows
    ]
    return {"ok": True, "users": users}


def handle_verify(message):
    try:
        face_id = int(message.get("face_id"))
        tag_id = int(message.get("tag_id"))
    except (TypeError, ValueError):
        return {"ok": False, "message": "Invalid verification request."}

    if face_id != tag_id:
        return {"ok": False, "message": "Face and AprilTag ID do not match."}

    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT name FROM users WHERE id = ?", (face_id,)
        ).fetchone()

    if row is None:
        return {"ok": False, "message": "No registered user has that ID."}

    return {
        "ok": True,
        "name": row[0],
        "message": f"Successfully logged in as {row[0]}.",
    }


def handle_client(connection, address):
    """Handles one connection; supports multiple requests per connection."""
    with connection:
        print(f"Client connected from {address}")
        try:
            while True:
                message = recv_message(connection)
                action = message.get("action")

                if action == "hello":
                    response = {"ok": True, "command": "open_interface"}
                elif action == "register":
                    response = handle_register(message)
                elif action == "get_users":
                    response = handle_get_users()
                elif action == "verify":
                    response = handle_verify(message)
                else:
                    response = {"ok": False, "message": "Unknown action."}

                send_message(connection, response)
        except (ConnectionError, OSError):
            pass
        except Exception as exc:
            print(f"Connection error from {address}: {exc}")
        finally:
            print(f"Client disconnected: {address}")


def main():
    init_db()
    # Load models eagerly so the first registration/verification is fast.
    get_models()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen()
        print(f"Socket server listening on {HOST}:{PORT}")
        print(f"Database: {DB_PATH}")

        while True:
            connection, address = server.accept()
            client_thread = threading.Thread(
                target=handle_client,
                args=(connection, address),
                daemon=True,
            )
            client_thread.start()
            print(f"Active threads: {threading.active_count()}")


if __name__ == "__main__":
    main()
