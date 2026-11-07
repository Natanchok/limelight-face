"""Interactive face + AprilTag login camera for VeriTag Face.

This script is launched by ``interface.py`` after the user presses *Login*.
It recognises registered faces, detects AprilTags, and asks the server to
verify that the face ID and tag ID match.  A successful verification opens
the success window and exits with code ``0``; cancelling with ``Q`` exits with
code ``1`` so the launcher can be shown again.

All heavy lifting is delegated to :mod:`vision`, which keeps this module
focused on the login workflow.
"""

from __future__ import annotations

import base64
import os
import sys
import time
from pathlib import Path
from typing import Final

import cv2
import numpy as np
import tkinter as tk

from protocol import ProtocolError, ProtocolHandler
from vision import ArUcoDetector, CameraHandler, FaceDetector, FaceRecognizer

if tuple(int(part) for part in cv2.__version__.split(".")[:2]) < (4, 8):
    raise RuntimeError(
        f"OpenCV 4.8 or newer is required; found {cv2.__version__}. "
        "Run this project with its virtual environment."
    )

ROOT: Final = Path(__file__).resolve().parent.parent
DATA_DIR: Final = ROOT / "data"
CAMERA_CACHE: Final = DATA_DIR / ".camera_index"
MODELS: Final = ROOT / "models"

RECOGNITION_THRESHOLD: Final = 0.363
REAPPEAR_SECONDS: Final = 2.0
UNKNOWN_COOLDOWN: Final = 3.0
VERIFY_INTERVAL: Final = 2.0

HOST: Final = os.environ.get("VERITAG_HOST", "127.0.0.1")
PORT: Final = int(os.environ.get("VERITAG_PORT", "5000"))

GREEN: Final = (0, 220, 0)
RED: Final = (0, 0, 255)

YUNET_MODEL: Final = MODELS / "yunet.onnx"
SFACE_MODEL: Final = MODELS / "sface.onnx"


def fetch_users(protocol: ProtocolHandler) -> dict[int, dict[str, object]]:
    """Fetch registered users and their face embeddings from the server.

    Args:
        protocol: Protocol handler configured with the same key as the server.

    Returns:
        Mapping of user ID to ``{"name": str, "features": ndarray}``.

    Raises:
        RuntimeError: If the server replies with an error.
    """
    response = protocol.request(HOST, PORT, {"action": "get_users"}, timeout=15)
    if not response.get("ok"):
        raise RuntimeError(response.get("message", "Could not load users."))

    users: dict[int, dict[str, object]] = {}
    for user in response.get("users", []):
        embedding = base64.b64decode(user["embedding"])
        users[int(user["id"])] = {
            "name": str(user["name"]),
            "features": np.frombuffer(embedding, dtype=np.float32)
            .reshape(1, -1)
            .copy(),
        }
    return users


def show_success_window(name: str) -> None:
    """Show the fullscreen 'successfully logged in' window.

    Args:
        name: Display name of the authenticated user.
    """
    root = tk.Tk()
    root.title("Login successful")
    root.configure(bg="#0d0f14")
    try:
        root.attributes("-type", "dialog")
        root.attributes("-fullscreen", True)
    except Exception:
        pass

    canvas = tk.Canvas(root, bg="#0d0f14", highlightthickness=0)
    canvas.pack(fill="both", expand=True)

    frame = tk.Frame(
        canvas,
        bg="#161a22",
        highlightthickness=1,
        highlightbackground="#242a36",
    )
    frame.place(relx=0.5, rely=0.5, anchor="center")

    tk.Label(
        frame,
        text="Successfully logged in",
        font=("Helvetica", 30, "bold"),
        bg="#161a22",
        fg="#3ddc84",
    ).pack(padx=90, pady=(70, 12))

    tk.Label(
        frame,
        text=f"Welcome, {name}",
        font=("Helvetica", 16),
        bg="#161a22",
        fg="#f3f5f8",
    ).pack(pady=(0, 40))

    tk.Button(
        frame,
        text="Close",
        command=root.destroy,
        font=("Helvetica", 12, "bold"),
        bg="#e63946",
        fg="#ffffff",
        activebackground="#f04d5a",
        activeforeground="#ffffff",
        relief="flat",
        bd=0,
        padx=40,
        pady=12,
        cursor="hand2",
    ).pack(pady=(0, 70))

    root.bind("<Escape>", lambda _: root.destroy())
    root.bind("<Control-q>", lambda _: root.destroy())
    root.mainloop()


def _screen_size() -> tuple[int, int]:
    """Return the primary screen size using a hidden Tk window."""
    root = tk.Tk()
    root.withdraw()
    try:
        return root.winfo_screenwidth(), root.winfo_screenheight()
    finally:
        root.destroy()


def _center_window(window_name: str, width: int, height: int) -> None:
    """Place a named OpenCV window in the centre of the screen."""
    screen_width, screen_height = _screen_size()
    x = max(0, (screen_width - width) // 2)
    y = max(0, (screen_height - height) // 2)
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(window_name, x, y)


def main() -> int:
    """Run the recognition/verification loop.

    Returns:
        ``0`` after a successful login, ``1`` when the camera is closed or
        the user cancels.
    """
    for path in (YUNET_MODEL, SFACE_MODEL):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Model not found: {path}")

    detector = FaceDetector(YUNET_MODEL)
    recognizer = FaceRecognizer(SFACE_MODEL)
    try:
        protocol = ProtocolHandler.from_env()
    except ProtocolError as exc:
        raise RuntimeError(f"Invalid VERITAG_KEY: {exc}") from exc

    try:
        known_users = fetch_users(protocol)
    except Exception as exc:
        raise RuntimeError(f"Could not load registered users from server: {exc}") from exc

    if not known_users:
        raise RuntimeError("No registered users on the server. Register a face first.")

    print(f"\nLoaded {len(known_users)} registered user(s) from server.")

    tag_detector = ArUcoDetector()
    print("AprilTag detector ready: DICT_APRILTAG_36h11")

    camera = CameraHandler(CAMERA_CACHE).open()
    print("Camera started. Press Q to quit.\n")

    window_name = "Face Recognition + AprilTag Detection"
    _center_window(window_name, camera.width, camera.height)

    last_seen: dict[int, float] = {}
    last_reported: dict[int | str, float] = {}
    previous_tag_ids: set[int] = set()
    last_verify_time = 0.0
    success_name: str | None = None

    try:
        while True:
            # Read the webcam exactly once per loop and keep a clean copy for
            # face and AprilTag processing.
            ok, frame = camera.read()
            if not ok or frame is None:
                print("Could not read camera frame.")
                break

            raw_frame = frame.copy()

            recognized_id: int | None = None
            recognized_name: str | None = None
            unknown_visible = False

            # ---------------- Face recognition ----------------
            faces = detector.detect(raw_frame)
            now = time.monotonic()

            if faces is not None:
                for face in faces:
                    x_box, y_box, w_box, h_box = map(int, face[:4])
                    features = recognizer.embedding(raw_frame, face)
                    best_id, best_name, best_score = recognizer.identify(
                        features,
                        known_users,
                        RECOGNITION_THRESHOLD,
                    )

                    if best_id is not None and best_name is not None:
                        label = f"{best_name} (#{best_id}) ({best_score:.2f})"
                        color = GREEN
                        recognized_id = best_id
                        recognized_name = best_name

                        if (
                            best_id not in last_reported
                            or now - last_seen.get(best_id, 0) > REAPPEAR_SECONDS
                        ):
                            print(
                                f"[RECOGNIZED] {best_name} (ID {best_id}, "
                                f"similarity {best_score:.2f})"
                            )
                            last_reported[best_id] = now

                        last_seen[best_id] = now
                    else:
                        label = "NOT IN SYSTEM"
                        color = RED
                        unknown_visible = True

                    cv2.rectangle(
                        frame,
                        (x_box, y_box),
                        (x_box + w_box, y_box + h_box),
                        color,
                        2,
                    )
                    cv2.putText(
                        frame,
                        label,
                        (x_box, max(y_box - 10, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

            if unknown_visible and (
                now - last_reported.get("UNKNOWN", 0) >= UNKNOWN_COOLDOWN
            ):
                print("[UNKNOWN] An unrecognized face is visible.")
                last_reported["UNKNOWN"] = now

            face_count = 0 if faces is None else len(faces)
            cv2.putText(
                frame,
                f"Faces: {face_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                GREEN,
                2,
            )

            # ---------------- AprilTag detection ----------------
            gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
            corners, ids = tag_detector.detect(gray)
            tag_detector.draw(frame, corners, ids)

            current_tag_ids: set[int] = set()
            if ids is not None:
                for tag_id in np.asarray(ids).flatten():
                    tag_value = int(tag_id)
                    current_tag_ids.add(tag_value)

                    # Print only when a tag first appears or reappears.
                    if tag_value not in previous_tag_ids:
                        print(f"[APRILTAG] Detected ID: {tag_value}")

            previous_tag_ids = current_tag_ids

            # ---------------- Verify face ID against AprilTag ----------------
            matched = recognized_id is not None and recognized_id in current_tag_ids

            if matched and now - last_verify_time >= VERIFY_INTERVAL:
                last_verify_time = now
                try:
                    response = protocol.request(
                        HOST,
                        PORT,
                        {
                            "action": "verify",
                            "face_id": recognized_id,
                            "tag_id": recognized_id,
                        },
                        timeout=5,
                    )
                except Exception as exc:
                    print(f"[AUTH] Could not reach server: {exc}")
                else:
                    if response.get("ok"):
                        success_name = str(
                            response.get("name", recognized_name or "")
                        )
                        print(f"[AUTH] {response.get('message')}")
                        break
                    print(
                        f"[AUTH] {response.get('message', 'Verification failed.')}"
                    )

            status_color = GREEN if matched else RED
            cv2.putText(
                frame,
                f"MATCH: {matched}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2,
                cv2.LINE_AA,
            )

            # ---------------- Display ----------------
            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        # Let HighGUI process the destroy event before opening the next Tk
        # window, otherwise a stray camera window can linger on some WMs.
        cv2.waitKey(1)
        print("Camera closed.")

    if success_name:
        show_success_window(success_name)
        return 0

    # Camera was closed without a successful login: let the launcher reopen.
    return 1


if __name__ == "__main__":
    sys.exit(main())
