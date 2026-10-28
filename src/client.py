import socket

# Use "127.0.0.1" for local testing, or the server's LAN IP address
HOST = "0.0.0.0"
PORT = 5000


def main():
    print(f"Connecting to trigger server at {HOST}:{PORT}...")

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            client.connect((HOST, PORT))
            print("Connected successfully! Server should now trigger combined.py.")

            # Optional: Keep the connection open for a moment if needed, 
            # or let the `with` block close it immediately.
            
    except ConnectionRefusedError:
        print("Error: Connection refused. Ensure the server is running and the IP/port are correct.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()