import json
import socket
import struct

MAX_MESSAGE_BYTES = 64 * 1024 * 1024


def send_message(sock, message):
    """Send a JSON message prefixed with its 4-byte length."""
    data = json.dumps(message).encode("utf-8")
    if len(data) > MAX_MESSAGE_BYTES:
        raise ValueError("Message too large")
    sock.sendall(struct.pack("!I", len(data)) + data)


def _recv_exact(sock, size):
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("Connection closed while receiving data")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_message(sock):
    """Receive a length-prefixed JSON message."""
    (length,) = struct.unpack("!I", _recv_exact(sock, 4))
    if length > MAX_MESSAGE_BYTES:
        raise ValueError("Incoming message too large")
    return json.loads(_recv_exact(sock, length).decode("utf-8"))


def request(host, port, message, timeout=15):
    """Open a short-lived connection, send one request, and return the reply."""
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        send_message(sock, message)
        return recv_message(sock)
