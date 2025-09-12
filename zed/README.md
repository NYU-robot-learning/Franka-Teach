# ZED Camera Integration

This module provides integration for ZED cameras in the RPL Vision Utils framework, following the same pattern as RealSense cameras.

## Features

- **Threaded Camera Worker**: Background thread for continuous image capture
- **Color and Depth Support**: Simultaneous color and depth image capture
- **Stereo Camera Support**: Left, right, and side-by-side stereo image modes
- **Camera Selection**: Support for multiple ZED cameras via serial number
- **Intrinsics and Distortion**: Automatic calibration parameter extraction
- **Framework Integration**: Compatible with the existing camera node system

## Requirements

- ZED SDK (pyzed)
- OpenCV
- NumPy
- EasyDict

## Installation

1. Install the ZED SDK following the official documentation: https://www.stereolabs.com/developers/release/
2. Install the Python wrapper: `pip install pyzed`

## Usage

### Basic Usage

```python
from rpl_vision_utils.zed import ZEDInterface
from easydict import EasyDict

# Configure camera streams
color_cfg = EasyDict(
    enabled=True, 
    img_w=1280, 
    img_h=720, 
    img_format="bgr8", 
    fps=30
)

depth_cfg = EasyDict(
    enabled=True, 
    img_w=1280, 
    img_h=720, 
    img_format="z16", 
    fps=30
)

# Initialize camera with stereo support
camera = ZEDInterface(
    device_id=0,
    color_cfg=color_cfg,
    depth_cfg=depth_cfg,
    serial_number="12345678",  # Optional: specify camera by serial number
    stereo_mode='separate'     # 'separate', 'sbs', or 'left_only'
)

# Start camera
camera.start()

# Get images
obs = camera.get_last_obs()
if obs is not None:
    if stereo_mode == 'separate':
        color_left = obs["color_left"]
        color_right = obs["color_right"]
    elif stereo_mode == 'sbs':
        color_sbs = obs["color_sbs"]
    else:  # left_only
        color_img = obs["color"]
    
    if depth_cfg.enabled:
        depth_img = obs["depth"]

# Clean up
camera.close()
```

### Using with Camera Node

```bash
# Run with default settings (stereo mode - publishes both left and right images)
python scripts/run_camera_node.py --camera-type zed --camera-id 0

# Run with specific camera serial number
python scripts/run_camera_node.py --camera-type zed --camera-id 0 --camera-serial 12345678

# Run with only color stream
python scripts/run_camera_node.py --camera-type zed --camera-id 0 --no-depth

# Run with only depth stream
python scripts/run_camera_node.py --camera-type zed --camera-id 0 --no-color
```

### Listing Available Cameras

```python
from rpl_vision_utils.zed import print_zed_cameras, list_zed_cameras

# Print all available cameras
print_zed_cameras()

# Get list of cameras programmatically
cameras = list_zed_cameras()
for camera in cameras:
    print(f"Serial: {camera['serial_number']}, Model: {camera['camera_model']}")
```

### Testing

Run the test scripts to verify your ZED camera setup:

```bash
# Navigate to the ZED test directory
cd rpl_vision_utils/zed/test

# List available cameras
python run_tests.py --list-cameras

# Run integration test (recommended for headless environments)
python run_tests.py --test integration --serial 25146947

# Run basic test (includes visualization)
python run_tests.py --test basic --serial 25146947

# Run performance test
python run_tests.py --test performance --serial 25146947

# Run both tests
python run_tests.py --test both --serial 25146947

# Test with only color stream
python run_tests.py --test integration --serial 25146947 --no-depth

# Test with only depth stream
python run_tests.py --test integration --serial 25146947 --no-color

# Test stereo functionality
python test_stereo.py --serial 25146947 --mode separate
python test_stereo.py --serial 25146947 --mode sbs
python test_stereo.py --serial 25146947 --mode left_only

# Or run individual scripts directly
python test_zed_camera.py --serial 25146947
python test_zed_integration.py --serial 25146947
python test_performance.py --serial 25146947
```

**Note**: The `test_zed_camera.py` script includes visualization which may not work in headless environments. Use `test_zed_integration.py` or `run_tests.py --test integration` for headless testing.

## API Reference

### ZEDInterface

Main interface class for ZED cameras.

#### Constructor

```python
ZEDInterface(
    device_id,
    color_cfg=None,
    depth_cfg=None,
    pc_cfg=None,
    serial_number=None,
    stereo_mode='separate'
)
```

**Parameters:**
- `device_id` (int): Device ID (for compatibility with framework)
- `color_cfg` (dict): Color stream configuration
- `depth_cfg` (dict): Depth stream configuration  
- `pc_cfg` (dict): Point cloud configuration (not implemented)
- `serial_number` (str): Camera serial number (optional)
- `stereo_mode` (str): Stereo mode ('separate', 'sbs', 'left_only')

#### Methods

- `start()`: Start the camera worker thread
- `close()`: Stop the camera worker thread
- `get_last_obs()`: Get the latest observation (color and/or depth images)
- `get_color_intrinsics(mode=None)`: Get color camera intrinsics
- `get_depth_intrinsics(mode=None)`: Get depth camera intrinsics
- `get_color_distortion()`: Get color camera distortion coefficients
- `get_depth_distortion()`: Get depth camera distortion coefficients

### Configuration

#### Color Configuration

```python
color_cfg = EasyDict(
    enabled=True,      # Enable color stream
    img_w=1280,        # Image width
    img_h=720,         # Image height
    img_format="bgr8", # Image format (BGR)
    fps=30            # Frame rate
)
```

#### Depth Configuration

```python
depth_cfg = EasyDict(
    enabled=True,      # Enable depth stream
    img_w=1280,        # Image width
    img_h=720,         # Image height
    img_format="z16",  # Depth format (16-bit)
    fps=30            # Frame rate
)
```

## Supported Resolutions

The ZED camera supports the following resolutions:

- VGA: 640x480
- HD720: 1280x720
- HD1080: 1920x1080
- HD2K: 2208x1242

## Stereo Modes

The ZED camera supports three stereo modes:

- **`separate`** (default): Publishes separate `color_left` and `color_right` images
- **`sbs`**: Publishes a single `color_sbs` side-by-side image (2560x720)
- **`left_only`**: Publishes only the left camera as `color` (backward compatibility)

## Notes

- Depth images are returned in millimeters (uint16 format) for compatibility with RealSense
- Color and depth streams use the same intrinsics (left camera)
- The implementation uses the left camera as the primary camera
- Color images are automatically converted from BGRA to BGR format for proper display
- Point cloud functionality is not yet implemented

## Troubleshooting

1. **"ZED SDK not available"**: Install the ZED SDK and Python wrapper
2. **"Failed to open ZED camera"**: Check camera connection and serial number
3. **No images received**: Ensure camera is not being used by another application
4. **Low frame rate**: Check USB bandwidth and reduce resolution if needed 