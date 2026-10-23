
import time
import requests

# Change this if your Limelight has a different IP address.
LIMELIGHT_IP = "172.29.0.1"

RESULTS_URL = f"http://{LIMELIGHT_IP}:5801/results"

def get_apriltag_ids() -> list[int]:
    """Return the IDs of all currently detected AprilTags."""
    try:
        response = requests.get(RESULTS_URL, timeout=2)
        response.raise_for_status()
        data = response.json()

        ids = []
        for tag in data.get("Results", {}).get("Fiducial", []):
            tag_id = tag.get("fID")
            if tag_id is not None:
                ids.append(int(tag_id))

        return ids

    except (requests.RequestException, ValueError) as error:
        print(f"Limelight connection or data error: {error}")
        return []


if __name__ == "__main__":
    previous_ids = None

    while True:
        ids = get_apriltag_ids()

        if ids != previous_ids:
            print("Detected AprilTag IDs:", ids)
            previous_ids = ids

        time.sleep(0.1)
