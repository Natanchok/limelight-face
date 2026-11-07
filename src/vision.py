"""Computer-vision building blocks for VeriTag Face.

The module wraps the OpenCV pieces used by the project in small, testable
classes:

* :class:`CameraHandler` opens the first working webcam and caches its index.
* :class:`FaceDetector` wraps the YuNet face detector.
* :class:`FaceRecognizer` wraps the SFace feature extractor and matcher.
* :class:`ArUcoDetector` wraps AprilTag detection.

All classes are annotated and safe to share between threads: every call that
touches a mutable OpenCV object is protected by an internal lock.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from types import TracebackType
from typing import Final

import cv2
import numpy as np

ROOT: Final = Path(__file__).resolve().parent.parent
MODELS_DIR: Final = ROOT / "models"
DEFAULT_CAMERA_CACHE: Final = ROOT / "data" / ".camera_index"


class CameraHandler:
    """Open and read from the first available webcam.

    The index of the camera that produced a frame is written to a cache file,
    so the next run opens the correct device immediately instead of probing
    every index again.

    Args:
        cache_path: File used to remember the working camera index.
        max_tested: Number of camera indexes to probe (``0..max_tested-1``).
        startup_timeout: Seconds to wait for each device to produce a frame.
        backend: OpenCV capture backend.  Defaults to ``CAP_V4L2`` on Linux and
            ``CAP_ANY`` elsewhere.

    Example:
        >>> with CameraHandler() as camera:  # doctest: +SKIP
        ...     ok, frame = camera.read()
    """

    def __init__(
        self,
        cache_path: str | Path | None = DEFAULT_CAMERA_CACHE,
        *,
        max_tested: int = 5,
        startup_timeout: float = 2.5,
        backend: int | None = None,
    ) -> None:
        """Configure camera probing, backend selection and index caching."""
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self.max_tested = max(1, int(max_tested))
        self.startup_timeout = max(0.0, float(startup_timeout))
        if backend is None:
            backend = (
                cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
            )
        self.backend = int(backend)
        self.camera_index: int | None = None
        self._capture: cv2.VideoCapture | None = None
        self._lock = threading.RLock()

    # -- context management ------------------------------------------
    def __enter__(self) -> "CameraHandler":
        """Open the camera and return ``self``."""
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release the camera when leaving a ``with`` block."""
        self.release()

    # -- properties ---------------------------------------------------
    @property
    def is_open(self) -> bool:
        """Whether the underlying ``VideoCapture`` is opened."""
        capture = self._capture
        return capture is not None and bool(capture.isOpened())

    @property
    def width(self) -> int:
        """Frame width reported by the camera, or ``0`` when closed."""
        capture = self._capture
        return 0 if capture is None else int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        """Frame height reported by the camera, or ``0`` when closed."""
        capture = self._capture
        return 0 if capture is None else int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # -- camera lifecycle ---------------------------------------------
    def _read_cached_index(self) -> int | None:
        """Read the cached camera index, ignoring a missing/corrupt file."""
        if self.cache_path is None:
            return None
        try:
            return int(self.cache_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return None

    def _write_cached_index(self, index: int) -> None:
        """Remember *index* as the working camera, ignoring write failures."""
        if self.cache_path is None:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(str(index), encoding="utf-8")
            print(f"[SYSTEM] Cached camera index {index}")
        except OSError:
            pass

    def open(self) -> "CameraHandler":
        """Probe camera indexes and keep the first one that returns a frame.

        Returns:
            ``self``, so the call can be chained.

        Raises:
            RuntimeError: If no camera returns a frame within the timeout.
        """
        with self._lock:
            if self.is_open:
                return self

            cached_index = self._read_cached_index()
            candidates: list[int] = []
            if cached_index is not None:
                candidates.append(cached_index)
            candidates.extend(
                index for index in range(self.max_tested) if index != cached_index
            )

            for index in candidates:
                capture = cv2.VideoCapture(index, self.backend)
                if not capture.isOpened():
                    capture.release()
                    continue

                deadline = time.monotonic() + self.startup_timeout
                while time.monotonic() < deadline:
                    ok, frame = capture.read()
                    if ok and frame is not None:
                        if index != cached_index:
                            self._write_cached_index(index)
                        self._capture = capture
                        self.camera_index = index
                        print(f"[SYSTEM] Camera ready at index {index}")
                        return self

                    time.sleep(0.02)

                capture.release()
                print(
                    f"[SYSTEM] Camera index {index} returned no frames "
                    f"within {self.startup_timeout:g} seconds. Skipping..."
                )

            raise RuntimeError("No camera returned a frame.")

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Read one BGR frame.

        Returns:
            ``(True, frame)`` on success, otherwise ``(False, None)``.
        """
        with self._lock:
            if not self.is_open:
                return False, None
            assert self._capture is not None
            ok, frame = self._capture.read()
            if not ok or frame is None:
                return False, None
            return True, frame

    def release(self) -> None:
        """Release the camera.  Safe to call more than once."""
        with self._lock:
            if self._capture is not None:
                self._capture.release()
                self._capture = None


class FaceDetector:
    """YuNet face detector.

    Args:
        model_path: Path to ``yunet.onnx``.
        input_size: Initial ``(width, height)`` used before the first frame.
        score_threshold: Minimum detection confidence.
        nms_threshold: Non-maximum-suppression threshold.
        top_k: Maximum number of candidate faces kept before NMS.

    Raises:
        FileNotFoundError: If *model_path* does not exist.
    """

    def __init__(
        self,
        model_path: str | Path = MODELS_DIR / "yunet.onnx",
        *,
        input_size: tuple[int, int] = (320, 320),
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        """Load the YuNet model and configure detection thresholds."""
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Face detection model not found: {model_path}")

        self.model_path = model_path
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self._detector = cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            self.input_size,
            float(score_threshold),
            float(nms_threshold),
            int(top_k),
        )
        self._lock = threading.RLock()

    def detect(self, image: np.ndarray) -> np.ndarray | None:
        """Detect faces in a BGR image.

        Args:
            image: Image as returned by :meth:`CameraHandler.read`.

        Returns:
            An ``(N, 15)`` array of detections (``x, y, w, h, landmarks...,
            score``), or ``None`` when no face is found.
        """
        if image is None or image.size == 0:
            return None

        height, width = image.shape[:2]
        with self._lock:
            self.input_size = (width, height)
            self._detector.setInputSize((width, height))
            _, faces = self._detector.detect(image)

        if faces is None or len(faces) == 0:
            return None
        return faces

    @staticmethod
    def largest_face(faces: np.ndarray | None) -> np.ndarray | None:
        """Return the detection with the largest bounding-box area.

        Args:
            faces: Output of :meth:`detect`.

        Returns:
            One detection row, or ``None`` when *faces* is empty.
        """
        if faces is None or len(faces) == 0:
            return None
        return max(faces, key=lambda face: float(face[2]) * float(face[3]))


class FaceRecognizer:
    """SFace feature extractor and cosine matcher.

    Args:
        model_path: Path to ``sface.onnx``.

    Raises:
        FileNotFoundError: If *model_path* does not exist.
    """

    def __init__(
        self,
        model_path: str | Path = MODELS_DIR / "sface.onnx",
    ) -> None:
        """Load the SFace model used for embeddings and matching."""
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Face recognition model not found: {model_path}")

        self.model_path = model_path
        self._recognizer = cv2.FaceRecognizerSF.create(str(model_path), "")
        self._lock = threading.RLock()

    def embedding(self, image: np.ndarray, face: np.ndarray) -> np.ndarray:
        """Align and encode one detected face.

        Args:
            image: BGR frame that produced *face*.
            face: One row from :meth:`FaceDetector.detect`.

        Returns:
            A ``float32`` feature row shaped ``(1, 128)``.
        """
        with self._lock:
            aligned = self._recognizer.alignCrop(image, face)
            features = self._recognizer.feature(aligned)
        return np.asarray(features, dtype=np.float32)

    def match(self, known: np.ndarray, probe: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            known: Stored embedding shaped ``(1, n)``.
            probe: Embedding of the visible face.

        Returns:
            Similarity score where a larger value means a closer match.
        """
        with self._lock:
            score = self._recognizer.match(
                known,
                probe,
                cv2.FaceRecognizerSF_FR_COSINE,
            )
        return float(score)

    def identify(
        self,
        probe: np.ndarray,
        known_users: dict[int, dict[str, object]],
        threshold: float,
    ) -> tuple[int | None, str | None, float]:
        """Find the closest registered user above *threshold*.

        Args:
            probe: Embedding of the visible face.
            known_users: Mapping of ``id`` to a mapping containing
                ``"name"`` and a ``"features"`` array.
            threshold: Minimum cosine similarity for a positive match.

        Returns:
            ``(user_id, name, score)``.  ``user_id`` and ``name`` are ``None``
            when no user reaches the threshold.
        """
        best_id: int | None = None
        best_name: str | None = None
        best_score = float("-inf")

        for user_id, user in known_users.items():
            features = user.get("features")
            if not isinstance(features, np.ndarray):
                continue
            score = self.match(features, probe)
            if score > best_score:
                best_score = score
                best_id = int(user_id)
                best_name = str(user.get("name", ""))

        if best_id is not None and best_score >= threshold:
            return best_id, best_name, best_score
        return None, None, best_score


class ArUcoDetector:
    """AprilTag detector based on OpenCV's ArUco module.

    Args:
        dictionary_id: ArUco/AprilTag dictionary, by default
            ``DICT_APRILTAG_36h11``.

    Raises:
        RuntimeError: If OpenCV was installed without the contrib modules.
    """

    def __init__(
        self,
        dictionary_id: int | None = None,
    ) -> None:
        """Create the AprilTag dictionary and detector parameters."""
        if not hasattr(cv2, "aruco"):
            raise RuntimeError(
                "cv2.aruco is missing. Install opencv-contrib-python "
                "in your virtual environment."
            )

        if dictionary_id is None:
            dictionary_id = cv2.aruco.DICT_APRILTAG_36h11
        self.dictionary_id = int(dictionary_id)
        dictionary = cv2.aruco.getPredefinedDictionary(self.dictionary_id)
        parameters = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        self._lock = threading.RLock()

    def detect(
        self,
        image: np.ndarray,
    ) -> tuple[list[np.ndarray], np.ndarray | None]:
        """Detect AprilTags in a grayscale image.

        Args:
            image: Grayscale image (``cv2.COLOR_BGR2GRAY``).

        Returns:
            ``(corners, ids)`` exactly as returned by OpenCV.  ``ids`` is
            ``None`` when no tag is visible.
        """
        if image is None or image.size == 0:
            return [], None

        with self._lock:
            corners, ids, _ = self._detector.detectMarkers(image)
        return list(corners), ids

    def detect_ids(self, image: np.ndarray) -> set[int]:
        """Return the set of AprilTag identifiers visible in *image*.

        Args:
            image: Grayscale image.
        """
        _, ids = self.detect(image)
        if ids is None:
            return set()
        return {int(tag_id) for tag_id in np.asarray(ids).flatten()}

    @staticmethod
    def draw(
        image: np.ndarray,
        corners: list[np.ndarray],
        ids: np.ndarray | None,
    ) -> None:
        """Draw detected tag outlines on *image* in place.

        Args:
            image: BGR frame to annotate.
            corners: Detected tag corners.
            ids: Detected tag identifiers, or ``None``.
        """
        if ids is not None and corners:
            cv2.aruco.drawDetectedMarkers(image, corners, ids)
