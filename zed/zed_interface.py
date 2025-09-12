import cv2
import numpy as np
import time
from easydict import EasyDict

from rpl_vision_utils.threading.threading_utils import Worker

try:
    import pyzed.sl as sl
except ModuleNotFoundError:
    print("WARNING: ZED SDK not found. ZED cameras will not be available.")
    sl = None


def get_zed_intrinsics_param(K_matrix: np.ndarray):
    """
    Args:
       K_matrix (np.ndarray): Numpy matrix of camera intrinsics

    Return:
       intrinsics_params (dict): a dictionary of intrinsics parameters, namely fx, fy, cx, cy
    """
    return {"fx": K_matrix[0][0], "fy": K_matrix[1][1], "cx": K_matrix[0][2], "cy": K_matrix[1][2]}


class ZEDCameraWorker(Worker):
    def __init__(
        self,
        camera_config: EasyDict = {},
        thread_safe: bool = True,
    ):
        if sl is None:
            raise RuntimeError("ZED SDK not available. Please install pyzed.")

        self.camera_config = camera_config
        self.enable_color = camera_config.enable_color
        self.enable_depth = camera_config.enable_depth
        self.serial_number = camera_config.serial_number
        
        # Configuration for stereo images
        self.stereo_mode = getattr(camera_config, 'stereo_mode', 'separate')  # 'separate', 'sbs', or 'left_only'

        # Initialize calibration structure
        self.calibration = {
            "color": {"intrinsics": None, "distortion": None},
            "depth": {"intrinsics": None, "distortion": None},
        }

        # Initialize ZED camera
        self._cam = sl.Camera()
        self._left_img = sl.Mat()
        self._right_img = sl.Mat()
        self._left_depth = sl.Mat()
        self._right_depth = sl.Mat()
        self._sbs_img = sl.Mat()  # Side-by-side image
        self._runtime = sl.RuntimeParameters()
        


        # Configure camera
        self._configure_camera()

        self.last_obs = None
        self.last_time = time.time()

        super().__init__()

    def _configure_camera(self):
        """Configure the ZED camera with the specified parameters."""
        init_params = sl.InitParameters()
        
        # Set camera by serial number if provided
        if self.serial_number:
            init_params.set_from_serial_number(int(self.serial_number))
        
        # Set resolution
        if hasattr(self.camera_config, 'color_cfg'):
            resolution_map = {
                (640, 480): sl.RESOLUTION.VGA,
                (1280, 720): sl.RESOLUTION.HD720,
                (1920, 1080): sl.RESOLUTION.HD1080,
                (2208, 1242): sl.RESOLUTION.HD2K,
            }
            img_w = self.camera_config.color_cfg.img_w
            img_h = self.camera_config.color_cfg.img_h
            init_params.camera_resolution = resolution_map.get((img_w, img_h), sl.RESOLUTION.HD720)
            init_params.camera_fps = self.camera_config.color_cfg.fps
        else:
            init_params.camera_resolution = sl.RESOLUTION.HD720
            init_params.camera_fps = 30

        # Enable depth if requested
        if self.enable_depth:
            # Use PERFORMANCE mode for better speed (deprecated but faster)
            # Users can override this by setting depth_mode in depth_cfg
            if hasattr(self.camera_config, 'depth_cfg') and hasattr(self.camera_config.depth_cfg, 'depth_mode'):
                depth_mode = self.camera_config.depth_cfg.depth_mode
            else:
                depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Faster but deprecated
            
            init_params.depth_mode = depth_mode
            init_params.coordinate_units = sl.UNIT.METER

        # Open camera
        status = self._cam.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera: {status}")

        # Get calibration parameters
        calib_params = self._cam.get_camera_information().camera_configuration.calibration_parameters
        
        # Process intrinsics for left camera (used as color camera)
        left_cam = calib_params.left_cam
        color_K_matrix = np.array([
            [left_cam.fx, 0.0, left_cam.cx],
            [0.0, left_cam.fy, left_cam.cy],
            [0.0, 0.0, 1.0],
        ])
        
        # For depth, we use the same intrinsics as color (left camera)
        depth_K_matrix = color_K_matrix.copy()

        self.calibration["color"]["intrinsics"] = color_K_matrix
        self.calibration["depth"]["intrinsics"] = depth_K_matrix
        self.calibration["color"]["distortion"] = np.array(left_cam.disto)
        self.calibration["depth"]["distortion"] = np.array(left_cam.disto)

        print(f"ZED Camera {self.serial_number} opened successfully")
        print(f"Color intrinsics: {color_K_matrix}")
        print(f"Depth intrinsics: {depth_K_matrix}")

    def get_intrinsics(self, key, mode=None):
        assert key in ["color", "depth"]
        if mode == "dict":
            return get_zed_intrinsics_param(self.calibration[key]["intrinsics"])
        else:
            return self.calibration[key]["intrinsics"]

    def get_distortion(self, key):
        assert key in ["color"]
        return self.calibration[key]["distortion"]

    def run(self) -> None:
        self.last_obs = EasyDict()

        while not self._halt:
            # Grab a new frame (this blocks until a new frame is available)
            err = self._cam.grab(self._runtime)
            if err != sl.ERROR_CODE.SUCCESS:
                continue

            # Get color images based on stereo mode
            if self.enable_color:
                if self.stereo_mode == 'sbs':
                    # Side-by-side mode
                    self._cam.retrieve_image(self._sbs_img, sl.VIEW.SIDE_BY_SIDE)
                    sbs_data = self._sbs_img.get_data().copy()
                    # Convert BGRA to BGR for proper color display
                    if sbs_data.shape[-1] == 4:  # BGRA format
                        self.last_obs["color_sbs"] = sbs_data[:, :, :3]  # Remove alpha channel
                    else:
                        self.last_obs["color_sbs"] = sbs_data
                elif self.stereo_mode == 'separate':
                    # Separate left and right images
                    self._cam.retrieve_image(self._left_img, sl.VIEW.LEFT)
                    self._cam.retrieve_image(self._right_img, sl.VIEW.RIGHT)
                    left_data = self._left_img.get_data().copy()
                    right_data = self._right_img.get_data().copy()
                    # Convert BGRA to BGR for proper color display
                    if left_data.shape[-1] == 4:  # BGRA format
                        self.last_obs["color_left"] = left_data[:, :, :3]  # Remove alpha channel
                        self.last_obs["color_right"] = right_data[:, :, :3]  # Remove alpha channel
                    else:
                        self.last_obs["color_left"] = left_data
                        self.last_obs["color_right"] = right_data
                else:  # left_only (default for compatibility)
                    self._cam.retrieve_image(self._left_img, sl.VIEW.LEFT)
                    left_data = self._left_img.get_data().copy()
                    # Convert BGRA to BGR for proper color display
                    if left_data.shape[-1] == 4:  # BGRA format
                        self.last_obs["color"] = left_data[:, :, :3]  # Remove alpha channel
                    else:
                        self.last_obs["color"] = left_data

            # Get depth image
            if self.enable_depth:
                self._cam.retrieve_measure(self._left_depth, sl.MEASURE.DEPTH)
                depth_data = self._left_depth.get_data().copy()
                # Convert to uint16 format similar to RealSense
                # Handle invalid values (NaN, inf) before conversion
                depth_data = np.nan_to_num(depth_data, nan=0.0, posinf=0.0, neginf=0.0)
                depth_data = (depth_data * 1000).astype(np.uint16)  # Convert meters to millimeters
                self.last_obs["depth"] = depth_data
                self.last_obs["unaligned_depth"] = depth_data.copy()

        # Clean up
        self._cam.close()

    def save_img(self, img_name):
        """Save a single image from the camera."""
        err = self._cam.grab(self._runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            self._cam.retrieve_image(self._left_img, sl.VIEW.LEFT)
            color_image = self._left_img.get_data()
            # Convert BGRA to BGR for proper color display
            if color_image.shape[-1] == 4:  # BGRA format
                color_image = color_image[:, :, :3]  # Remove alpha channel
            cv2.imwrite(img_name, color_image)


class ZEDInterface:
    """
    This is the Python Interface for getting images from ZED cameras.
    """

    def __init__(
        self,
        device_id,
        color_cfg: dict = None,
        depth_cfg: dict = None,
        pc_cfg: dict = None,
        serial_number=None,
        stereo_mode='separate',  # 'separate', 'sbs', or 'left_only'
    ):
        if sl is None:
            raise RuntimeError("ZED SDK not available. Please install pyzed.")

        # Default color configuration
        if color_cfg is not None:
            self.color_cfg = color_cfg
        else:
            self.color_cfg = EasyDict(
                enabled=True, img_w=1280, img_h=720, img_format="bgr8", fps=30
            )

        # Default depth configuration
        if depth_cfg is not None:
            self.depth_cfg = depth_cfg
        else:
            self.depth_cfg = EasyDict(
                enabled=False, img_w=1280, img_h=720, img_format="z16", fps=30
            )

        # Point cloud configuration (not implemented yet)
        if pc_cfg is not None:
            self.pc_cfg = pc_cfg
        else:
            self.pc_cfg = EasyDict(enabled=False)

        self.serial_number = serial_number
        self.stereo_mode = stereo_mode

        # Validate that at least one stream is enabled
        if not (self.color_cfg.enabled or self.depth_cfg.enabled or self.pc_cfg.enabled):
            raise ValueError("At least one stream (color, depth, or point cloud) must be enabled")

        # Create camera configuration
        camera_config = EasyDict(
            enable_color=self.color_cfg.enabled,
            enable_depth=self.depth_cfg.enabled,
            enable_pc=self.pc_cfg.enabled,
            color_cfg=self.color_cfg,
            depth_cfg=self.depth_cfg,
            pc_cfg=self.pc_cfg,
            serial_number=self.serial_number,
            stereo_mode=getattr(self, 'stereo_mode', 'separate'),  # Default to separate mode
        )

        # Create camera worker
        self.camera = ZEDCameraWorker(
            camera_config=camera_config,
            thread_safe=False,
        )

    def start(self):
        """Start the camera worker thread."""
        self.camera.start()

    def get_last_obs(self):
        """
        Get last observation from camera
        """
        if self.camera.last_obs is None or self.camera.last_obs == {}:
            return None
        else:
            self.last_obs = self.camera.last_obs
            return self.last_obs

    def close(self):
        """Stop the camera worker thread."""
        self.camera.halt()

    def get_camera_intrinsics(self):
        """Get camera intrinsics for both color and depth."""
        return {
            "color": self.camera.get_intrinsics("color"),
            "depth": self.camera.get_intrinsics("depth"),
        }

    def get_depth_intrinsics(self, mode=None):
        """Get depth camera intrinsics."""
        intrinsics = self.camera.get_intrinsics("depth", mode=mode)
        return intrinsics

    def get_color_intrinsics(self, mode=None):
        """Get color camera intrinsics."""
        intrinsics = self.camera.get_intrinsics("color", mode=mode)
        return intrinsics

    def get_color_distortion(self, mode=None):
        """Get color camera distortion coefficients."""
        distortion = self.camera.get_distortion("color")
        return distortion

    def get_depth_distortion(self):
        """Get depth camera distortion coefficients."""
        # ZED uses the same distortion for both color and depth
        distortion = self.camera.get_distortion("color")
        return distortion 