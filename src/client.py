import os
import subprocess
import sys
from pathlib import Path

from protocol import request

HOST = os.environ.get("LIMELIGHT_HOST", "127.0.0.1")  # Must match the server's IP
PORT = int(os.environ.get("LIMELIGHT_PORT", "5000"))
INTERFACE_SCRIPT = Path(__file__).resolve().with_name("interface.py")
ROOT = INTERFACE_SCRIPT.parent.parent
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"


def main():
    try:
        response = request(HOST, PORT, {"action": "hello", "role": "launcher"})
    except Exception as exc:
        print(f"Could not reach server at {HOST}:{PORT}: {exc}")
        return

    if response.get("command") != "open_interface":
        print(f"Unexpected server response: {response}")
        return

    print(f"Command received. Opening {INTERFACE_SCRIPT.name} locally...")

    env = os.environ.copy()
    env["LIMELIGHT_HOST"] = HOST
    env["LIMELIGHT_PORT"] = str(PORT)

    subprocess.Popen(
        [
            str(VENV_PYTHON) if VENV_PYTHON.is_file() else sys.executable,
            str(INTERFACE_SCRIPT),
        ],
        cwd=str(INTERFACE_SCRIPT.parent),
        env=env,
    )


if __name__ == "__main__":
    main()
