
import cv2

for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_V4L2)

    if cap.isOpened():
        ok, frame = cap.read()
        print(
            f"Index {i}: opened={cap.isOpened()}, "
            f"read={ok}, "
            f"shape={frame.shape if ok else None}"
        )
    else:
        print(f"Index {i}: unavailable")

    cap.release()

