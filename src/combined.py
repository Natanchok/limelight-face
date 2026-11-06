import base64
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tkinter as tk

from protocol import request

if tuple(int(part) for part in cv2.__version__.split(".")[:2]) < (4, 8):
    raise RuntimeError(
        f"OpenCV 4.8 or newer is required; found {cv2.__version__}. "
        "Run this project with its virtual environment."
    )

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CAMERA_CACHE = DATA_DIR / ".camera_index"
CAMERA_BACKEND = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY


def open_camera(max_tested=5, startup_timeout=2.5):
    """Open the first working camera, reusing a cached index when possible."""
    cached_index = None
    try:
        cached_index = int(CAMERA_CACHE.read_text().strip())
    except Exception:
        pass

    candidates = []
    if cached_index is not None:
        candidates.append(cached_index)
    candidates.extend(i for i in range(max_tested) if i != cached_index)

    for index in candidates:
        cap = cv2.VideoCapture(index, CAMERA_BACKEND)
        if not cap.isOpened():
            cap.release()
            continue

        deadline = time.monotonic() + startup_timeout
        while time.monotonic() < deadline:
            ok, frame = cap.read()
            if ok and frame is not None:
                if index != cached_index:
                    try:
                        CAMERA_CACHE.write_text(str(index))
                        print(f"[SYSTEM] Cached camera index {index}")
                    except OSError:
                        pass
                print(f"[SYSTEM] Camera ready at index {index}")
                return cap
            time.sleep(0.02)

        cap.release()
        print(
            f"[SYSTEM] Camera index {index} returned no frames "
            f"within {startup_timeout:g} seconds. Skipping..."
        )

    raise RuntimeError("No camera returned a frame.")


# ---------------- CONFIG ----------------

MODELS = ROOT / "models"

RECOGNITION_THRESHOLD = 0.363
REAPPEAR_SECONDS = 2.0
UNKNOWN_COOLDOWN = 3.0
VERIFY_INTERVAL = 2.0

HOST = os.environ.get("VERITAG_HOST", "127.0.0.1")
PORT = int(os.environ.get("VERITAG_PORT", "5000"))

GREEN = (0, 220, 0)
RED = (0, 0, 255)
BLUE = (255, 150, 0)

YUNET_MODEL = str(MODELS / "yunet.onnx")
SFACE_MODEL = str(MODELS / "sface.onnx")

# ---------------- LOAD FACE MODELS ----------------

for path in (YUNET_MODEL, SFACE_MODEL):
    if not Path(path).is_file():
        raise FileNotFoundError(f"Model not found: {path}")

face_detector = cv2.FaceDetectorYN.create(
    YUNET_MODEL, "", (320, 320), 0.6, 0.3, 5000
)

face_recognizer = cv2.FaceRecognizerSF.create(
    SFACE_MODEL, ""
)

# ---------------- LOAD REGISTERED USERS FROM SERVER ----------------


def fetch_users():
    """Fetch user IDs, names, and face embeddings from the server."""
    response = request(HOST, PORT, {"action": "get_users"}, timeout=15)
    if not response.get("ok"):
        raise RuntimeError(response.get("message", "Could not load users."))

    users = {}
    for user in response.get("users", []):
        embedding = base64.b64decode(user["embedding"])
        users[int(user["id"])] = {
            "name": user["name"],
            "features": np.frombuffer(
                embedding, dtype=np.float32
            ).reshape(1, -1).copy(),
        }
    return users


try:
    known_users = fetch_users()
except Exception as exc:
    raise RuntimeError(f"Could not load registered users from server: {exc}")

if not known_users:
    raise RuntimeError("No registered users on the server. Register a face first.")

print(f"\nLoaded {len(known_users)} registered user(s) from server.")

# ---------------- LOAD APRILTAG DETECTOR ----------------

if not hasattr(cv2, "aruco"):
    raise RuntimeError(
        "cv2.aruco is missing. Install opencv-contrib-python "
        "in your virtual environment."
    )

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_APRILTAG_36h11
)

tag_parameters = cv2.aruco.DetectorParameters()
tag_detector = cv2.aruco.ArucoDetector(
    dictionary, tag_parameters
)

print("AprilTag detector ready: DICT_APRILTAG_36h11")


def show_success_window(name):
    """Show the 'successfully logged in' interface."""
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
        canvas, bg="#161a22", highlightthickness=1,
        highlightbackground="#242a36",
    )
    frame.place(relx=0.5, rely=0.5, anchor="center")

    tk.Label(
        frame, text="Successfully logged in",
        font=("Helvetica", 30, "bold"), bg="#161a22", fg="#3ddc84",
    ).pack(padx=90, pady=(70, 12))

    tk.Label(
        frame, text=f"Welcome, {name}",
        font=("Helvetica", 16), bg="#161a22", fg="#f3f5f8",
    ).pack(pady=(0, 40))

    tk.Button(
        frame, text="Close", command=root.destroy,
        font=("Helvetica", 12, "bold"), bg="#e63946", fg="#ffffff",
        activebackground="#f04d5a", activeforeground="#ffffff",
        relief="flat", bd=0, padx=40, pady=12, cursor="hand2",
    ).pack(pady=(0, 70))

    root.bind("<Escape>", lambda _: root.destroy())
    root.bind("<Control-q>", lambda _: root.destroy())
    root.mainloop()

# ---------------- CAMERA & WINDOW SETUP ----------------

# Camera is opened once and reused for the whole session.
camera = open_camera()

print("Camera started. Press Q to quit.\n")

# Get screen resolution using tkinter to calculate center coordinates.
root = tk.Tk()
root.withdraw()  # Hide the main tkinter window
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.destroy()

# Get camera frame resolution
cam_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
cam_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calculate center position
x = (screen_width - cam_width) // 2
y = (screen_height - cam_height) // 2

# Create and position the window
window_name = "Face Recognition + AprilTag Detection"
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
cv2.moveWindow(window_name, x, y)

last_seen = {}
last_reported = {}
previous_tag_ids = set()
last_verify_time = 0.0
success_name = None

try:
    while True:
        # OpenCV reads the webcam exactly once per loop.
        ok, frame = camera.read()

        if not ok:
            print("Could not read camera frame.")
            break

        # Keep a clean copy for AprilTag and Face processing.
        raw_frame = frame.copy()

        # Track state for the current frame
        recognized_id = None
        recognized_name = None
        unknown_visible = False

        # ================ FACE RECOGNITION ================

        height, width = raw_frame.shape[:2]
        face_detector.setInputSize((width, height))
        _, faces = face_detector.detect(raw_frame)

        now = time.monotonic()

        if faces is not None:
            for face in faces:
                x_box, y_box, w_box, h_box = map(int, face[:4])

                aligned = face_recognizer.alignCrop(
                    raw_frame, face
                )
                features = face_recognizer.feature(aligned)

                best_id = None
                best_name = None
                best_score = -1.0

                for user_id, user in known_users.items():
                    score = float(face_recognizer.match(
                        user["features"],
                        features,
                        cv2.FaceRecognizerSF_FR_COSINE
                    ))

                    if score > best_score:
                        best_score = score
                        best_id = user_id
                        best_name = user["name"]

                if (
                    best_id is not None
                    and best_score >= RECOGNITION_THRESHOLD
                ):
                    label = f"{best_name} (#{best_id}) ({best_score:.2f})"
                    color = GREEN
                    recognized_id = best_id
                    recognized_name = best_name

                    if (
                        best_id not in last_reported
                        or now - last_seen.get(best_id, 0)
                        > REAPPEAR_SECONDS
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
                    frame, (x_box, y_box), (x_box + w_box, y_box + h_box), color, 2
                )
                cv2.putText(
                    frame, label, (x_box, max(y_box - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                    cv2.LINE_AA
                )

        if unknown_visible and (
            now - last_reported.get("UNKNOWN", 0)
            >= UNKNOWN_COOLDOWN
        ):
            print("[UNKNOWN] An unrecognized face is visible.")
            last_reported["UNKNOWN"] = now

        face_count = 0 if faces is None else len(faces)
        cv2.putText(
            frame, f"Faces: {face_count}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2
        )

        # ================ APRILTAG DETECTION ================

        gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = tag_detector.detectMarkers(gray)

        current_tag_ids = set()

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(
                frame, corners, ids
            )

            for tag_id in ids.flatten():
                tag_id = int(tag_id)
                current_tag_ids.add(tag_id)

                # Print only when a tag first appears or reappears.
                if tag_id not in previous_tag_ids:
                    print(f"[APRILTAG] Detected ID: {tag_id}")

        previous_tag_ids = current_tag_ids

        # ================ VERIFY FACE ID AGAINST APRILTAG ================

        matched = recognized_id is not None and recognized_id in current_tag_ids

        if matched and now - last_verify_time >= VERIFY_INTERVAL:
            last_verify_time = now
            try:
                response = request(HOST, PORT, {
                    "action": "verify",
                    "face_id": recognized_id,
                    "tag_id": recognized_id,
                }, timeout=5)
            except Exception as exc:
                print(f"[AUTH] Could not reach server: {exc}")
            else:
                if response.get("ok"):
                    success_name = response.get("name", recognized_name)
                    print(f"[AUTH] {response.get('message')}")
                    break
                print(f"[AUTH] {response.get('message', 'Verification failed.')}")

        status_color = GREEN if matched else RED
        cv2.putText(
            frame, f"MATCH: {matched}", (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA
        )

        # ================ DISPLAY ================

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("Camera closed.")

if success_name:
    show_success_window(success_name)