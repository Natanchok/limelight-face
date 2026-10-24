
from pathlib import Path
import time
import cv2

# ---------------- CONFIG ----------------

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
FACES_DIR = ROOT / "data" / "known_faces"

CAMERA_INDEX = 1
RECOGNITION_THRESHOLD = 0.363
REAPPEAR_SECONDS = 2.0
UNKNOWN_COOLDOWN = 3.0

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

# ---------------- LOAD REGISTERED FACES ----------------

if not FACES_DIR.is_dir():
    raise FileNotFoundError(f"Faces directory not found: {FACES_DIR}")

known_faces = {}
extensions = {".jpg", ".jpeg", ".png"}

for image_path in sorted(FACES_DIR.iterdir()):
    if image_path.suffix.lower() not in extensions:
        continue

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[SKIP] Cannot read {image_path.name}")
        continue

    height, width = image.shape[:2]
    face_detector.setInputSize((width, height))
    _, faces = face_detector.detect(image)

    if faces is None or len(faces) == 0:
        print(f"[SKIP] No face detected in {image_path.name}")
        continue

    face = max(faces, key=lambda f: f[2] * f[3])
    aligned = face_recognizer.alignCrop(image, face)
    features = face_recognizer.feature(aligned)

    name = image_path.stem.upper()
    known_faces[name] = features
    print(f"[REGISTERED] {name} from {image_path.name}")

if not known_faces:
    raise RuntimeError(f"No usable registered faces in {FACES_DIR}")

print(f"\nLoaded {len(known_faces)} registered face(s).")

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

# ---------------- CAMERA ----------------

camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)

if not camera.isOpened():
    raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}")

print("Camera started. Press Q to quit.\n")

last_seen = {}
last_reported = {}
previous_tag_ids = set()

try:
    while True:
        # OpenCV reads the webcam exactly once per loop.
        ok, frame = camera.read()

        if not ok:
            print("Could not read camera frame.")
            break

        # Keep a clean copy for AprilTag processing.
        # The displayed frame will receive both sets of annotations.
        raw_frame = frame.copy()

        # ================ FACE RECOGNITION ================

        height, width = raw_frame.shape[:2]
        face_detector.setInputSize((width, height))
        _, faces = face_detector.detect(raw_frame)

        now = time.monotonic()
        unknown_visible = False

        if faces is not None:
            for face in faces:
                x, y, w, h = map(int, face[:4])

                aligned = face_recognizer.alignCrop(
                    raw_frame, face
                )
                features = face_recognizer.feature(aligned)

                best_name = None
                best_score = -1.0

                for name, reference in known_faces.items():
                    score = float(face_recognizer.match(
                        reference,
                        features,
                        cv2.FaceRecognizerSF_FR_COSINE
                    ))

                    if score > best_score:
                        best_score = score
                        best_name = name

                if (
                    best_name is not None
                    and best_score >= RECOGNITION_THRESHOLD
                ):
                    label = f"{best_name} ({best_score:.2f})"
                    color = GREEN

                    if (
                        best_name not in last_reported
                        or now - last_seen.get(best_name, 0)
                        > REAPPEAR_SECONDS
                    ):
                        print(
                            f"[RECOGNIZED] {best_name} "
                            f"(similarity {best_score:.2f})"
                        )
                        last_reported[best_name] = now

                    last_seen[best_name] = now

                else:
                    label = "The System does not recognize this person"
                    color = RED
                    unknown_visible = True

                cv2.rectangle(
                    frame, (x, y), (x + w, y + h), color, 2
                )
                cv2.putText(
                    frame, label, (x, max(y - 10, 25)),
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

        # ================ DISPLAY ================

        cv2.imshow("Face Recognition + AprilTag Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("Camera closed.")
