import socket
import subprocess
import sys
from pathlib import Path


HOST = "0.0.0.0"
PORT = 5000
COMBINED_SCRIPT = Path(__file__).resolve().with_name("combined.py")


def main():
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
		server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		server.bind((HOST, PORT))
		server.listen()
		print(f"Socket server listening on {HOST}:{PORT}")

		while True:
			connection, address = server.accept()
			with connection:
				print(f"Client connected from {address}")
				subprocess.Popen(
					[sys.executable, str(COMBINED_SCRIPT)],
					cwd=str(COMBINED_SCRIPT.parent),
				)


if __name__ == "__main__":
	main()
