"""Send newly written image frames to a TCP receiver in filename/time order."""

import argparse
import socket
import struct
import time
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images_sorted(watch_dir: Path):
    files = []
    for path in watch_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMG_EXTS:
            try:
                stat = path.stat()
            except OSError:
                continue
            files.append((stat.st_mtime, path.name, path))
    files.sort(key=lambda item: (item[0], item[1]))
    return [path for _, __, path in files]


def wait_until_stable(path: Path, stable_for: float = 0.2, timeout: float = 5.0) -> bool:
    start = time.time()
    last_size = -1
    last_change = time.time()
    while time.time() - start < timeout:
        try:
            size = path.stat().st_size
        except OSError:
            time.sleep(0.05)
            continue
        if size != last_size:
            last_size = size
            last_change = time.time()
        elif time.time() - last_change >= stable_for:
            return True
        time.sleep(0.05)
    return False


def send_one(sock: socket.socket, image_path: Path) -> None:
    if not wait_until_stable(image_path):
        raise RuntimeError(f"File is still changing: {image_path}")

    filename_bytes = image_path.name.encode("utf-8")
    payload = image_path.read_bytes()
    sock.sendall(struct.pack(">HQ", len(filename_bytes), len(payload)))
    sock.sendall(filename_bytes)
    sock.sendall(payload)

    ack = b""
    while not ack.endswith(b"\n"):
        chunk = sock.recv(64)
        if not chunk:
            raise ConnectionError("Receiver disconnected before ACK.")
        ack += chunk
    if ack.strip() != b"OK":
        raise RuntimeError(f"Unexpected ACK: {ack!r}")


def run(host: str, port: int, watch_dir: Path, poll_seconds: float) -> None:
    watch_dir.mkdir(parents=True, exist_ok=True)
    sent = set()
    print(f"[sender] Connecting to {host}:{port} ...")
    with socket.create_connection((host, port), timeout=10) as sock:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"[sender] Connected. Watching: {watch_dir}")
        while True:
            images = list_images_sorted(watch_dir)
            if not images:
                time.sleep(poll_seconds)
                continue

            newest = images[-1]
            queue = ([newest] if newest.name not in sent else []) + [
                p for p in images if p.name not in sent and p != newest
            ]
            if not queue:
                time.sleep(poll_seconds)
                continue

            for image_path in queue:
                try:
                    send_one(sock, image_path)
                    sent.add(image_path.name)
                    print(f"[sender] Sent: {image_path.name} ({image_path.stat().st_size} bytes)")
                except Exception as exc:
                    print(f"[sender] Error sending {image_path.name}: {exc}")
                    time.sleep(0.5)
                    break
            time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("host", help="Receiver hostname or IP address")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--watch-dir", type=Path, default=Path("data/frames"))
    parser.add_argument("--poll", type=float, default=0.25, dest="poll_seconds")
    args = parser.parse_args()
    run(args.host, args.port, args.watch_dir, args.poll_seconds)


if __name__ == "__main__":
    main()
