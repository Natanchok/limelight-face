# VeriTag Face

Python computer-vision access-control demo: a webcam recognises a registered
face, an AprilTag supplies a second factor, and a TCP server verifies that the
face ID and tag ID match. The server owns the SQLite database and the face
models; the client machine runs the launcher and camera UI.

Built with OpenCV (YuNet face detection, SFace face recognition, AprilTag
detection via the contrib ArUco module), SQLite3, TCP sockets, threading and
optional AES-GCM authenticated encryption.

## Project structure

| File | Responsibility |
| --- | --- |
| `src/protocol.py` | Length-prefixed frames, JSON validation, optional AES-GCM, `ProtocolHandler` |
| `src/database.py` | Thread-safe, fully parameterised SQLite access via `DatabaseHandler` |
| `src/vision.py` | `CameraHandler`, `FaceDetector`, `FaceRecognizer`, `ArUcoDetector` |
| `src/server.py` | Multi-threaded TCP server and request handlers |
| `src/combined.py` | Camera loop: recognise face, detect AprilTag, verify login |
| `src/interface.py` | Tkinter launcher: login and account registration |
| `src/client.py` | Connects to the server and opens the launcher |

## Install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`opencv-contrib-python` is required because `cv2.aruco` and the AprilTag
dictionaries are not included in the plain `opencv-python` wheel.

The YuNet and SFace models must exist at `models/yunet.onnx` and
`models/sface.onnx`.

## Run

Start the server in one terminal:

```bash
.venv/bin/python src/server.py
```

Start the client in another terminal:

```bash
.venv/bin/python src/client.py
```

Or in VS Code, choose **Run VeriTag (Server + Client)** in the Run and Debug
panel and press Run. Press `q` in the camera window to cancel; a successful
login closes the launcher.

Register a user from the launcher's **Register Account** screen. The ID must
be the same integer encoded in that user's AprilTag.

## Encryption (optional)

By default traffic is plaintext so the local demo works out of the box.
Set `VERITAG_KEY` on **both** the server and every client to enable
AES-GCM authenticated encryption:

```bash
export VERITAG_KEY="$(python -c 'import base64,os; print(base64.urlsafe_b64encode(os.urandom(32)).decode())')"
.venv/bin/python src/server.py
```

The key may be base64, URL-safe base64 or hex and must decode to 16, 24 or
32 bytes. When a handler has a key it rejects plaintext frames; when it has no
key it rejects encrypted frames. A wrong key or a corrupted authentication tag
is rejected safely with a `ProtocolError`.

Environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `VERITAG_HOST` | `0.0.0.0` (server), `127.0.0.1` (client) | Bind/connect address |
| `VERITAG_PORT` | `5000` | TCP port |
| `VERITAG_KEY` | unset | Enables AES-GCM on both ends |
| `VERITAG_DB` | `data/veritag.db` | Server database path |
| `VERITAG_CLIENT_TIMEOUT` | `120` | Idle client timeout in seconds |
| `VERITAG_LOG_LEVEL` | `INFO` | Server log level |

## Security design

- **SQL injection**: every statement in `database.py` uses `?` placeholders;
  no user input is formatted into SQL text. `sqlite3.Row` is used for safe
  column access.
- **Malformed frames**: `ProtocolHandler.recv` validates the length prefix
  (`1..64 MiB`), the frame flag, the nonce/tag length and the JSON type, and
  converts every parsing failure into `ProtocolError`.
- **Broken clients**: `handle_client` catches `ProtocolError`,
  `ConnectionResetError`, `BrokenPipeError`, `ConnectionError`, timeouts and
  generic `OSError`. Only that client's daemon thread ends; the accept loop
  and other clients keep running. `SO_REUSEADDR` is enabled.
- **Audit log**: successful logins, denials and registrations are written to
  the `logs` table and can be read with `DatabaseHandler.get_logs()`.
  Deleting an account keeps its history by default and appends a
  `user_deleted` event; pass `purge_logs=True` to erase it.
- **Upload limits**: registration photos are capped at 8 MiB, names at 100
  characters, and only `.jpg`, `.jpeg` and `.png` filenames are accepted.

This is a school demonstration project, not a production authentication
system. A real deployment would add TLS, rate limiting, secure key storage,
liveness detection and a stronger enrolment policy.

## Database

`DatabaseHandler` creates both tables on first use:

- `users`: `id`, `name`, `photo_path`, `embedding` (raw `float32` BLOB),
  `registered_at`.
- `logs`: `id`, `user_id`, `event`, `details`, `created_at`.

Useful helpers: `add_user`, `add_embedding`, `user_exists`, `get_user`,
`get_users`, `verify_credentials`, `add_log`, `get_logs`, `delete_user`.

## Troubleshooting

- **`No camera returned a frame`**: close other apps using the camera and
  check that `/dev/video*` is accessible. The working index is cached in
  `data/.camera_index`.
- **`cv2.aruco is missing`**: install `opencv-contrib-python`, not only
  `opencv-python`.
- **`Encrypted frame received but no key is configured`**: start both ends
  with the same `VERITAG_KEY`, or unset it on both.
