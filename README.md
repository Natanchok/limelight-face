# VeriTag

Python computer-vision project for a webcam-based access-control demo.

## Components
- OpenCV: webcam access and face detection
- YuNet: face detection model
- SFace: face recognition model
- AprilTag: tag identification for access verification

## Run
Start the server in one terminal:

    .venv/bin/python src/server.py

Start the client in another terminal:

    .venv/bin/python src/client.py

Or in VS Code, choose **Run VeriTag (Server + Client)** in the Run and Debug panel
and press Run. This opens the server and client in separate terminals.

Press q to close the webcam window.
