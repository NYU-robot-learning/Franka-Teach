import numpy as np

from frankateach.network import ZMQCameraPublisher
from frankateach.utils import FrequencyTimer, notify_component_start
from frankateach.constants import CAM_FPS, DEPTH_PORT_OFFSET

import time
from rpl_vision_utils.zed import ZEDInterface
from easydict import EasyDict


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

        # Different publishers to avoid overload
        self.rgb_publisher = ZMQCameraPublisher(host, port)

        # if self._depth:
        #     self.depth_publisher = ZMQCameraPublisher(
        #         host, port=port + DEPTH_PORT_OFFSET
        #     )

        self.timer = FrequencyTimer(CAM_FPS)

        # Starting the realsense pipeline
        self._start_zed(self._cam_serial_num)

    def _start_zed(self, cam_serial_num):
        color_cfg = EasyDict(
            enabled=True,
            img_w=self.cam_config.width,
            img_h=self.cam_config.height,
            img_format="bgr8",
            fps=self.cam_config.fps,
        )
        depth_cfg = EasyDict(
            enabled=self._depth,
            img_w=self.cam_config.width,
            img_h=self.cam_config.height,
            img_format="z16",
            fps=self.cam_config.fps,
        )

        # initialize the camera
        self.camera = ZEDInterface(
            device_id=self.cam_id,
            color_cfg=color_cfg,
            depth_cfg=depth_cfg,
            serial_number=cam_serial_num,
            stereo_mode="separate",
        )

        # start the camera
        self.camera.start()

    def get_rgb_depth_images(self):
        frames = None

        while frames is None:
            obs = self.camera.get_last_obs()
            if obs is None:
                continue
            left, right = obs["color_left"], obs["color_right"]
            frames = np.concatenate([left, right], axis=1)

            # if self._depth:
            #     left, right = obs["depth_left"], obs["depth_right"]
            #     depth_image = np.concatenate([left, right], axis=-1)
            # else:
            depth_image = None
            
        timestamp = time.time()

        return frames, depth_image, timestamp

    def stream(self):
        # Starting the realsense stream
        notify_component_start("zed")
        print(f"Started the ZED pipeline for camera: {self._cam_serial_num}!")

        try:
            while True:
                self.timer.start_loop()
                color_image, depth_image, timestamp = self.get_rgb_depth_images()
                
                self.rgb_publisher.pub_rgb_image(color_image, timestamp)
                # if self._depth:
                #     self.depth_publisher.pub_depth_image(depth_image, timestamp)

                self.timer.end_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self.rgb_publisher.stop()
            # if self._depth:
            #     self.depth_publisher.stop()
            self.camera.close()
