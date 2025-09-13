from frankateach.network import ZMQCameraPublisher
from frankateach.utils import FrequencyTimer, notify_component_start
from frankateach.constants import CAM_FPS

import re
import cv2
import time
import subprocess
# from rpl_vision_utils.zed import ZEDInterface


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


class ZedCamera:
    def __init__(
        self,
        host,
        port,
        cam_id,
        cam_config,
    ):
        self.cam_id = cam_id
        self.cam_config = cam_config
        self._cam_serial_num = cam_config.cam_serial_num
        self._depth = cam_config.depth
        assert self._depth is False, "Depth is not supported for ZED camera"
        
        self.cam_id_sys = cam_config.cam_id_sys if hasattr(cam_config, 'cam_id_sys') else None

        # Different publishers to avoid overload
        self.rgb_publisher = ZMQCameraPublisher(host, port)

        self.timer = FrequencyTimer(CAM_FPS)

        # # Starting the realsense pipeline
        # self._start_zed(self._cam_serial_num)

        if self.cam_id_sys is not None:
            index, backend = self.cam_id_sys, cv2.CAP_V4L2
        else:
            index, backend = find_camera_by_serial(self._cam_serial_num)
            if index is None:
                raise ValueError(
                    f"Camera with serial number {self._cam_serial_num} not found"
                )

        self.cap = cv2.VideoCapture(index, backend)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_config.width * 2)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_config.height)

    def get_rgb_images(self):
        frames = None

        while frames is None:
            if not self.cap.isOpened():
                raise ValueError(f"Failed to open camera at index {self.index}")

            ret, frames = self.cap.read()
            if not ret:
                raise ValueError(
                    f"Failed to capture frame from camera at index {self.index}"
                )

        timestamp = time.time()
        return frames, timestamp

    def stream(self):
        # Starting the realsense stream
        notify_component_start("zed")
        print(f"Started the ZED pipeline for camera: {self._cam_serial_num}!")

        try:
            while True:
                self.timer.start_loop()
                color_images, timestamp = self.get_rgb_images()

                self.rgb_publisher.pub_rgb_image(color_images, timestamp)

                self.timer.end_loop()
        except KeyboardInterrupt:
            pass
        finally:
            print("Shutting down realsense pipeline for camera {}.".format(self.cam_id))
            self.rgb_publisher.stop()
            self.cap.release()
