import socket
import subprocess
import sys
from pathlib import Path

# Use "127.0.0.1" for local testing, or the server's LAN IP address
HOST = "172.20.10.2"
PORT = 5000
COMBINED_SCRIPT = Path(__file__).resolve().with_name("combined.py")

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.connect((HOST, PORT))
        print("Connected to server. Waiting for commands...")
        
        # Receive data from the server
        data = client.recv(1024)
        
        if data == b"EXECUTE_COMBINED":
            print(f"Command received. Running {COMBINED_SCRIPT.name} locally...")
            subprocess.Popen(
                [sys.executable, str(COMBINED_SCRIPT)],
                cwd=str(COMBINED_SCRIPT.parent),
            )

if __name__ == "__main__":
    main()