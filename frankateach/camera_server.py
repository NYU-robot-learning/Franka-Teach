import pyrealsense2 as rs
import time
import threading

from frankateach.sensors.realsense import RealsenseCamera
from frankateach.sensors.fisheye_cam import FishEyeCamera
from frankateach.sensors.zed_trt import ZedCamera


class CameraServer:
    def __init__(self, host: str, cam_port: int, cam_configs: list):
        self._host = host
        self._cam_port = cam_port
        self._cam_configs = cam_configs
        self._cam_threads = []

        if "realsense" in cam_configs.keys():
            ctx = rs.context()
            devices = ctx.query_devices()

            for dev in devices:
                dev.hardware_reset()

            print("Waiting for hardware reset on cameras for 15 seconds...")
            time.sleep(10)

    def _start_component(self, cam_idx, cam_config):
        cam_type = cam_config.type
        if cam_type == "realsense":
            component = RealsenseCamera(
                host=self._host,
                port=self._cam_port + cam_idx,
                cam_id=cam_idx,
                cam_config=cam_config,
            )
        elif cam_type == "fisheye":
            component = FishEyeCamera(
                host=self._host,
                port=self._cam_port + cam_idx,
                cam_id=cam_idx,
                cam_config=cam_config,
            )
        elif cam_type == "zed":
            component = ZedCamera(
                host=self._host,
                port=self._cam_port + cam_idx,
                cam_id=cam_idx,
                cam_config=cam_config,
            )
        else:
            raise ValueError(f"Invalid camera type: {cam_type}")
        component.stream()

    def _init_camera_threads(self):
        for cam_type in self._cam_configs:
            for cam_cfg in self._cam_configs[cam_type]:
                cam_thread = threading.Thread(
                    target=self._start_component,
                    args=(cam_cfg.cam_id, cam_cfg),
                    daemon=True,
                )
                cam_thread.start()
                self._cam_threads.append(cam_thread)

        for cam_thread in self._cam_threads:
            cam_thread.join()
