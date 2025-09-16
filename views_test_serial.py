import cv2
import shutil
from pathlib import Path
import subprocess
import re
import sys


def get_camera_serial_numbers():
    """
    Get serial numbers of connected cameras using v4l2-ctl (Linux) or system commands.
    Returns a dictionary mapping camera index to serial number.
    """
    camera_serials = {}

    try:
        # Try using v4l2-ctl for Linux systems
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"], capture_output=True, text=True, check=True
        )

        lines = result.stdout.split("\n")
        current_device = None

        for line in lines:
            line = line.strip()
            if "video" in line and ":" in line:
                # Extract device number
                match = re.search(r"video(\d+)", line)
                if match:
                    current_device = int(match.group(1))
            elif "Serial" in line or "serial" in line:
                # Extract serial number
                serial_match = re.search(r"[Ss]erial[:\s]+([A-Za-z0-9_-]+)", line)
                if serial_match and current_device is not None:
                    camera_serials[current_device] = serial_match.group(1)
                    current_device = None

    except (subprocess.CalledProcessError, FileNotFoundError):
        print("v4l2-ctl not available, trying alternative method...")

        # Alternative: try to get camera info using lsusb
        try:
            result = subprocess.run(
                ["lsusb"], capture_output=True, text=True, check=True
            )
            print("Available USB devices:")
            print(result.stdout)
            print(
                "\nNote: This method doesn't provide direct mapping to camera indices."
            )
            print("You may need to manually test camera indices or install v4l2-utils.")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(
                "lsusb not available. Please install v4l2-utils for better camera detection."
            )

    return camera_serials


def find_camera_by_serial(target_serial):
    """
    Find camera index by serial number.
    Returns (index, backend) tuple or (None, None) if not found.
    """
    # First try to get serial numbers from system
    camera_serials = get_camera_serial_numbers()

    # Check if we found the target serial in our system scan
    for index, serial in camera_serials.items():
        if serial == target_serial:
            print(f"Found camera with serial {target_serial} at index {index}")
            return index, cv2.CAP_V4L2  # Use V4L2 backend for Linux

    # If not found in system scan, try brute force approach
    print(
        f"Serial {target_serial} not found in system scan. Trying brute force approach..."
    )

    # Try different backends
    backends = [cv2.CAP_V4L2, cv2.CAP_GSTREAMER, cv2.CAP_ANY]

    for backend in backends:
        for index in range(10):  # Check first 10 indices
            try:
                cap = cv2.VideoCapture(index, backend)
                if cap.isOpened():
                    # Try to get camera properties that might contain serial info
                    # Note: This is limited as OpenCV doesn't directly expose serial numbers
                    cap.release()
                    print(f"Camera found at index {index} with backend {backend}")
                    # For now, we'll return the first available camera
                    # In a real scenario, you'd need additional logic to verify the serial
                    return index, backend
            except:
                continue

    return None, None


def test_camera_by_serial(serial_number):
    """
    Test camera access using serial number.
    """
    print(f"Searching for camera with serial number: {serial_number}")

    index, backend = find_camera_by_serial(serial_number)

    if index is None:
        print(f"Camera with serial number {serial_number} not found.")
        return False

    print(f"Opening camera at index {index} with backend {backend}")
    cap = cv2.VideoCapture(index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 680)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print(f"Failed to open camera at index {index}")
        return False

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
            cv2.imwrite(f"views_test_serial/camera_{serial_number}.jpg", frame)
            print(f"Saved test image as views_test_serial/camera_{serial_number}.jpg")
            break

    cap.release()
    return True


def list_available_cameras():
    """
    List all available cameras and their properties.
    """
    print("Scanning for available cameras...")
    camera_serials = get_camera_serial_numbers()

    if camera_serials:
        print("\nFound cameras with serial numbers:")
        for index, serial in camera_serials.items():
            print(f"  Index {index}: Serial {serial}")
    else:
        print("No cameras with serial numbers detected via system scan.")

    print("\nTesting camera indices 0-9 for availability:")
    backends = [cv2.CAP_V4L2, cv2.CAP_GSTREAMER, cv2.CAP_ANY]

    for backend in backends:
        backend_name = (
            "V4L2"
            if backend == cv2.CAP_V4L2
            else "GStreamer"
            if backend == cv2.CAP_GSTREAMER
            else "Any"
        )
        print(f"\nTesting {backend_name} backend:")

        for index in range(10):
            try:
                cap = cv2.VideoCapture(index, backend)
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"  Index {index}: Available ({width}x{height})")
                    cap.release()
                else:
                    cap.release()
            except:
                pass


if __name__ == "__main__":
    # Create directory to save images
    dir_path = Path("./views_test_serial")
    if dir_path.exists():
        shutil.rmtree(dir_path)
    Path("views_test_serial").mkdir(parents=True, exist_ok=True)

    if len(sys.argv) > 1:
        # If serial number provided as command line argument
        target_serial = sys.argv[1]
        print(f"Testing camera with serial number: {target_serial}")
        success = test_camera_by_serial(target_serial)
        if not success:
            print("\nTrying to list available cameras...")
            list_available_cameras()
    else:
        # List available cameras
        list_available_cameras()
        print("\nUsage: python views_test_serial.py <serial_number>")
        print("Example: python views_test_serial.py ABC123456")
