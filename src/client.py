import os
import subprocess
import sys
import time
from pathlib import Path

from protocol import request

HOST = os.environ.get("VERITAG_HOST", "127.0.0.1")  # Must match the server's IP
PORT = int(os.environ.get("VERITAG_PORT", "5000"))
INTERFACE_SCRIPT = Path(__file__).resolve().with_name("interface.py")
ROOT = INTERFACE_SCRIPT.parent.parent
if sys.platform == "win32":
    VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
else:
    VENV_PYTHON = ROOT / ".venv" / "bin" / "python"


def connect_with_retry(timeout=15.0):
    """Wait briefly for the server to finish starting before giving up."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            return request(HOST, PORT, {"action": "hello", "role": "launcher"})
        except OSError as exc:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.25)


def main():
    try:
        response = connect_with_retry()
    except Exception as exc:
        print(f"Could not reach server at {HOST}:{PORT}: {exc}")
        return

    if response.get("command") != "open_interface":
        print(f"Unexpected server response: {response}")
        return

    print(f"Command received. Opening {INTERFACE_SCRIPT.name} locally...")

    env = os.environ.copy()
    env["VERITAG_HOST"] = HOST
    env["VERITAG_PORT"] = str(PORT)

    process = subprocess.Popen(
        [
            str(VENV_PYTHON) if VENV_PYTHON.is_file() else sys.executable,
            str(INTERFACE_SCRIPT),
        ],
        cwd=str(INTERFACE_SCRIPT.parent),
        env=env,
    )
    process.wait()


if __name__ == "__main__":
    main()
