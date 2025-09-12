"""
Utility functions for ZED cameras.
"""

try:
    import pyzed.sl as sl
except ModuleNotFoundError:
    print("WARNING: ZED SDK not found. ZED utilities will not be available.")
    sl = None


def list_zed_cameras():
    """
    List all available ZED cameras.
    
    Returns:
        list: List of dictionaries containing camera information
    """
    if sl is None:
        print("ZED SDK not available")
        return []
    
    cameras = []
    try:
        device_list = sl.Camera.get_device_list()
        for device in device_list:
            camera_info = {
                "serial_number": str(device.serial_number),
                "camera_model": str(device.camera_model),
                "camera_state": str(device.camera_state),
                "path": device.path,
            }
            cameras.append(camera_info)
    except Exception as e:
        print(f"Error listing ZED cameras: {e}")
    
    return cameras


def get_zed_camera_info(serial_number):
    """
    Get detailed information about a specific ZED camera.
    
    Args:
        serial_number (str): Camera serial number
        
    Returns:
        dict: Camera information
    """
    if sl is None:
        print("ZED SDK not available")
        return None
    
    try:
        # Create a temporary camera instance to get info
        cam = sl.Camera()
        init_params = sl.InitParameters()
        init_params.set_from_serial_number(int(serial_number))
        
        status = cam.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            print(f"Failed to open ZED camera {serial_number}: {status}")
            return None
        
        # Get camera information
        camera_info = cam.get_camera_information()
        cam.close()
        
        info = {
            "serial_number": serial_number,
            "camera_model": str(camera_info.camera_model),
            "firmware_version": getattr(camera_info, 'camera_firmware_version', 'Unknown'),
            "sensors_firmware_version": getattr(camera_info, 'sensors_firmware_version', 'Unknown'),
            "camera_resolution": str(getattr(camera_info, 'camera_resolution', 'Unknown')),
            "camera_fps": getattr(camera_info, 'camera_fps', 'Unknown'),
            "calibration_parameters": {
                "left_cam": {
                    "fx": camera_info.camera_configuration.calibration_parameters.left_cam.fx,
                    "fy": camera_info.camera_configuration.calibration_parameters.left_cam.fy,
                    "cx": camera_info.camera_configuration.calibration_parameters.left_cam.cx,
                    "cy": camera_info.camera_configuration.calibration_parameters.left_cam.cy,
                },
                "right_cam": {
                    "fx": camera_info.camera_configuration.calibration_parameters.right_cam.fx,
                    "fy": camera_info.camera_configuration.calibration_parameters.right_cam.fy,
                    "cx": camera_info.camera_configuration.calibration_parameters.right_cam.cx,
                    "cy": camera_info.camera_configuration.calibration_parameters.right_cam.cy,
                }
            }
        }
        
        return info
        
    except Exception as e:
        print(f"Error getting ZED camera info: {e}")
        return None


def print_zed_cameras():
    """
    Print information about all available ZED cameras.
    """
    cameras = list_zed_cameras()
    
    if not cameras:
        print("No ZED cameras found")
        return
    
    print(f"Found {len(cameras)} ZED camera(s):")
    print("-" * 50)
    
    for i, camera in enumerate(cameras):
        print(f"Camera {i+1}:")
        print(f"  Serial Number: {camera['serial_number']}")
        print(f"  Model: {camera['camera_model']}")
        print(f"  State: {camera['camera_state']}")
        print(f"  Path: {camera['path']}")
        
        # Get detailed info
        detailed_info = get_zed_camera_info(camera['serial_number'])
        if detailed_info:
            print(f"  Firmware: {detailed_info['firmware_version']}")
            print(f"  Resolution: {detailed_info['camera_resolution']}")
            print(f"  FPS: {detailed_info['camera_fps']}")
        
        print()


if __name__ == "__main__":
    print_zed_cameras() 