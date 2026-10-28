# Import Path for filesystem paths relative to this script.
from pathlib import Path
# Import time for monotonic timing of recognition messages.
import time

# Import OpenCV for camera capture, face detection, recognition, and display.
import cv2

# ---------------- CONFIG ----------------  # Mark the configuration section.

# Resolve the project root as the parent of this script's directory.
ROOT = Path(__file__).resolve().parent.parent
# Store model files under the project's models directory.
MODELS = ROOT / "models"
# Store registered face images under the project's data directory.
FACES_DIR = ROOT / "data" / "known_faces"

# Select the camera device index to open.
CAMERA_INDEX = 1

# Require this minimum cosine similarity to accept a recognized identity.
RECOGNITION_THRESHOLD = 0.363

# Require this absence duration before reporting the same person again.
REAPPEAR_SECONDS = 2.0

# Define the drawing color for recognized faces in BGR order.
GREEN = (0, 220, 0)
# Define the drawing color for unrecognized faces in BGR order.
RED = (0, 0, 255)
# Define an additional blue drawing color in BGR order.
BLUE = (255, 150, 0)

# Build the path to the YuNet face-detection model.
YUNET_MODEL = str(MODELS / "yunet.onnx")
# Build the path to the SFace face-recognition model.
SFACE_MODEL = str(MODELS / "sface.onnx")

# ---------------- LOAD MODELS ----------------  # Mark the model setup section.

# Check that both required model files exist before loading them.
for file in (Path(YUNET_MODEL), Path(SFACE_MODEL)):
    # Stop immediately with the missing path if a model is unavailable.
    if not file.is_file():
        raise FileNotFoundError(f"Model not found: {file}")

# Create the YuNet detector with its initial input size and thresholds.
detector = cv2.FaceDetectorYN.create(
    YUNET_MODEL, "", (320, 320), 0.6, 0.3, 5000
)
# Create the SFace recognizer from its model file.
recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL, "")

# ---------------- REGISTER KNOWN FACES ----------------  # Mark enrollment.

# Require a directory containing known face images.
if not FACES_DIR.is_dir():
    raise FileNotFoundError(f"Faces directory not found: {FACES_DIR}")

# Limit registration inputs to common image filename extensions.
extensions = {".jpg", ".jpeg", ".png"}
# Map each registered display name to its extracted feature vector.
known_faces = {}

# Process face image files in a stable, sorted order.
for image_path in sorted(FACES_DIR.iterdir()):
    # Ignore files whose extensions are not supported.
    if image_path.suffix.lower() not in extensions:
        continue

    # Read the current registration image from disk.
    image = cv2.imread(str(image_path))
    # Skip files OpenCV could not decode and explain why.
    if image is None:
        print(f"[SKIP] Cannot read image: {image_path.name}")
        continue

    # Get the image dimensions and configure detector input accordingly.
    ih, iw = image.shape[:2]
    detector.setInputSize((iw, ih))
    # Detect faces; the first returned value is unused here.
    _, faces = detector.detect(image)

    # Skip images in which the detector found no faces.
    if faces is None or len(faces) == 0:
        print(f"[SKIP] No face detected in {image_path.name}")
        continue

    # Choose the face with the greatest bounding-box area.
    face = max(faces, key=lambda f: f[2] * f[3])
    # Align and crop the chosen face for consistent feature extraction.
    aligned = recognizer.alignCrop(image, face)
    # Extract the face's recognition feature vector.
    features = recognizer.feature(aligned)

    # Use the uppercase filename stem as the person's display name.
    name = image_path.stem.upper()
    # Save this person's feature vector for comparisons at runtime.
    known_faces[name] = features
    # Confirm successful registration in the terminal.
    print(f"[REGISTERED] {name} from {image_path.name}")

# Require at least one image to produce a usable registered face.
if not known_faces:
    # Explain that no registration could be completed.
    raise RuntimeError(
        f"No usable registered faces found in {FACES_DIR}"
    )

# Report the number of registered identities.
print(f"\nLoaded {len(known_faces)} registered face(s).")
# Tell the user how to start and stop the live camera loop.
print("Starting camera. Press Q to quit.\n")

# ---------------- CAMERA ----------------  # Mark the live camera section.

# Open the selected camera using the V4L2 backend.
camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
# Stop if the camera could not be opened.
if not camera.isOpened():
    raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}")

# Track recent visibility and report times to limit repeated messages.
last_seen = {}
last_reported = {}
# Set the minimum interval between unknown-face terminal messages.
UNKNOWN_COOLDOWN = 3.0

# Ensure camera and display resources are released on exit or error.
try:
    # Continuously process frames until the user quits or capture fails.
    while True:
        # Read one frame from the camera.
        ok, frame = camera.read()
        # End the loop if a frame could not be read.
        if not ok:
            print("Could not read camera frame.")
            break

        # Get frame dimensions and update the detector's expected input size.
        height, width = frame.shape[:2]
        detector.setInputSize((width, height))
        # Detect all faces present in this frame.
        _, faces = detector.detect(frame)

        # Use monotonic time for reliable elapsed-time comparisons.
        now = time.monotonic()
        # Collect identities visible in this frame.
        visible_names = set()
        # Track whether any face failed identity matching in this frame.
        unknown_visible = False

        # Process faces only when the detector returned results.
        if faces is not None:
            # Examine each detected face independently.
            for face in faces:
                # Convert the face bounding-box coordinates to integers.
                x, y, w, h = map(int, face[:4])
                # Align and crop the detected face.
                aligned = recognizer.alignCrop(frame, face)
                # Extract its feature vector for identity matching.
                features = recognizer.feature(aligned)

                # Initialize the best match name and score.
                best_name = None
                best_score = -1.0

                # Compare this face with every registered feature vector.
                for name, reference_features in known_faces.items():
                    # Calculate cosine similarity between the two vectors.
                    score = float(recognizer.match(
                        reference_features,
                        features,
                        cv2.FaceRecognizerSF_FR_COSINE
                    ))

                    # Keep the identity with the highest similarity score.
                    if score > best_score:
                        best_score = score
                        best_name = name

                # Accept the best match only if it reaches the threshold.
                if best_name is not None and best_score >= RECOGNITION_THRESHOLD:
                    # Show the identity and similarity on the video frame.
                    label = f"{best_name} ({best_score:.2f})"
                    # Use green to indicate a recognized identity.
                    color = GREEN
                    # Record the identity as visible in this frame.
                    visible_names.add(best_name)

                    # Report new appearances or reappearances after a gap.
                    if (
                        best_name not in last_reported
                        or now - last_seen.get(best_name, 0) > REAPPEAR_SECONDS
                    ):
                        print(f"[RECOGNIZED] {best_name} (similarity {best_score:.2f})")
                        # Save when this identity was last reported.
                        last_reported[best_name] = now

                    # Update the most recent time this identity was seen.
                    last_seen[best_name] = now
                else:
                    # Mark the face as unrecognized and use red for its box.
                    label = "NOT RECOGNIZED"
                    color = RED
                    # Note that an unknown face is present in this frame.
                    unknown_visible = True

                # Draw a rectangle around the face's bounding box.
                cv2.rectangle(
                    frame, (x, y), (x + w, y + h), color, 2
                )
                # Draw the identity label above the face box.
                cv2.putText(
                    frame, label, (x, max(y - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                    cv2.LINE_AA
                )

        # Report unknown faces no more often than the configured cooldown.
        if unknown_visible and (
            now - last_reported.get("UNKNOWN", 0) >= UNKNOWN_COOLDOWN
        ):
            print("[UNKNOWN] An unregistered face is visible.")
            # Record when the unknown-face message was last printed.
            last_reported["UNKNOWN"] = now

        # Overlay the number of detected faces on the current frame.
        cv2.putText(
            frame, f"Faces: {0 if faces is None else len(faces)}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2
        )
        # Display the annotated frame in a window.
        cv2.imshow("Registered Face Recognition", frame)

        # Quit when the user presses Q.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Always release the camera and close OpenCV windows.
finally:
    camera.release()
    cv2.destroyAllWindows()
