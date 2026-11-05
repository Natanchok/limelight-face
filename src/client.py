import socket
import subprocess
import sys
from pathlib import Path

HOST = "192.168.1.247"  # Must match the server's IP
PORT = 5000
COMBINED_SCRIPT = Path(__file__).resolve().with_name("combined.py")
ROOT = COMBINED_SCRIPT.parent.parent
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.connect((HOST, PORT))
        print("Connected to server. Waiting for commands...")
        
        # Receive data from the server
        data = client.recv(1024)
        
        if data == b"EXECUTE_COMBINED":
            print(f"Command received. Running {COMBINED_SCRIPT.name} locally...")
            subprocess.Popen(
                [
                    str(VENV_PYTHON) if VENV_PYTHON.is_file() else sys.executable,
                    str(COMBINED_SCRIPT),
                ],
                cwd=str(COMBINED_SCRIPT.parent),
            )

if __name__ == "__main__":
    main()