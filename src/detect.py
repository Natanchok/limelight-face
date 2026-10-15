from pathlib import Path
import time

import cv2

# ---------------- CONFIG ----------------

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
FACES_DIR = ROOT / "data" / "known_faces"

CAMERA_INDEX = 3

# Cosine similarity threshold; tune with real test images.
RECOGNITION_THRESHOLD = 0.363

# A person must disappear this long before another terminal message
# is printed for the same person appearing again.
REAPPEAR_SECONDS = 2.0

GREEN = (0, 220, 0)
RED = (0, 0, 255)
BLUE = (255, 150, 0)

YUNET_MODEL = str(MODELS / "yunet.onnx")
SFACE_MODEL = str(MODELS / "sface.onnx")

# ---------------- LOAD MODELS ----------------

for file in (Path(YUNET_MODEL), Path(SFACE_MODEL)):
    if not file.is_file():
        raise FileNotFoundError(f"Model not found: {file}")

detector = cv2.FaceDetectorYN.create(
    YUNET_MODEL, "", (320, 320), 0.6, 0.3, 5000
)
recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL, "")

# ---------------- REGISTER KNOWN FACES ----------------

if not FACES_DIR.is_dir():
    raise FileNotFoundError(f"Faces directory not found: {FACES_DIR}")

extensions = {".jpg", ".jpeg", ".png"}
known_faces = {}

for image_path in sorted(FACES_DIR.iterdir()):
    if image_path.suffix.lower() not in extensions:
        continue

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[SKIP] Cannot read image: {image_path.name}")
        continue

    ih, iw = image.shape[:2]
    detector.setInputSize((iw, ih))
    _, faces = detector.detect(image)

    if faces is None or len(faces) == 0:
        print(f"[SKIP] No face detected in {image_path.name}")
        continue

    # Use the largest face in each registration image.
    face = max(faces, key=lambda f: f[2] * f[3])
    aligned = recognizer.alignCrop(image, face)
    features = recognizer.feature(aligned)

    # Filename without extension becomes the person's display name.
    name = image_path.stem.upper()
    known_faces[name] = features
    print(f"[REGISTERED] {name} from {image_path.name}")

if not known_faces:
    raise RuntimeError(
        f"No usable registered faces found in {FACES_DIR}"
    )

print(f"\nLoaded {len(known_faces)} registered face(s).")
print("Starting camera. Press Q to quit.\n")

# ---------------- CAMERA ----------------

camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
if not camera.isOpened():
    raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}")

# Track last-seen times to avoid repeated terminal spam.
last_seen = {}
last_reported = {}
UNKNOWN_COOLDOWN = 3.0

try:
    while True:
        ok, frame = camera.read()
        if not ok:
            print("Could not read camera frame.")
            break

        height, width = frame.shape[:2]
        detector.setInputSize((width, height))
        _, faces = detector.detect(frame)

        now = time.monotonic()
        visible_names = set()
        unknown_visible = False

        if faces is not None:
            for face in faces:
                x, y, w, h = map(int, face[:4])
                aligned = recognizer.alignCrop(frame, face)
                features = recognizer.feature(aligned)

                best_name = None
                best_score = -1.0

                for name, reference_features in known_faces.items():
                    score = float(recognizer.match(
                        reference_features,
                        features,
                        cv2.FaceRecognizerSF_FR_COSINE
                    ))

                    if score > best_score:
                        best_score = score
                        best_name = name

                if best_name is not None and best_score >= RECOGNITION_THRESHOLD:
                    label = f"{best_name} ({best_score:.2f})"
                    color = GREEN
                    visible_names.add(best_name)

                    if (
                        best_name not in last_reported
                        or now - last_seen.get(best_name, 0) > REAPPEAR_SECONDS
                    ):
                        print(f"[RECOGNIZED] {best_name} (similarity {best_score:.2f})")
                        last_reported[best_name] = now

                    last_seen[best_name] = now
                else:
                    label = "NOT RECOGNIZED"
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
            now - last_reported.get("UNKNOWN", 0) >= UNKNOWN_COOLDOWN
        ):
            print("[UNKNOWN] An unregistered face is visible.")
            last_reported["UNKNOWN"] = now

        cv2.putText(
            frame, f"Faces: {0 if faces is None else len(faces)}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2
        )
        cv2.imshow("Registered Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
