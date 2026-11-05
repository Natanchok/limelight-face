import socket

HOST = "192.168.1.247"
PORT = 5000

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
                # Send an execution command to the client
                connection.sendall(b"EXECUTE_COMBINED")
                print("Execution command sent to client.")

if __name__ == "__main__":
    main()