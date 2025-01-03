import cv2
import shutil
from pathlib import Path


def test_camera_index(index):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 680)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print(f"Failed to open camera at index {index}")
        return

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to capture frame from camera at index {index}")
            break

        idx += 1
        if idx == 10:
            print(
                "Frame default resolution: ("
                + str(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                + "; "
                + str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                + ")"
            )
            cv2.imwrite(f"views_test/camera_{index}.jpg", frame)
            break

    cap.release()


# Test camera indices from 0 to 100
if __name__ == "__main__":
    # make directory to save images
    dir_path = Path("./views_test")
    if dir_path.exists():
        shutil.rmtree(dir_path)
    Path("views_test").mkdir(parents=True, exist_ok=True)

    for camera_id in range(100):
        test_camera_index(camera_id)
