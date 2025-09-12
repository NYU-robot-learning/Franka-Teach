#!/usr/bin/env python3
"""
Test script for ZED camera implementation.
"""

import argparse
import time
import cv2
import numpy as np
from easydict import EasyDict

import sys
import os
# Add parent directory to path to allow imports when running from test folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from rpl_vision_utils.zed import ZEDInterface, print_zed_cameras


def test_zed_camera(serial_number=None, enable_depth=True, enable_color=True):
    """
    Test ZED camera functionality.
    
    Args:
        serial_number (str): Camera serial number (optional)
        enable_depth (bool): Enable depth stream
        enable_color (bool): Enable color stream
    """
    print("Testing ZED Camera Implementation")
    print("=" * 40)
    
    # List available cameras
    print("Available ZED cameras:")
    print_zed_cameras()
    print()
    
    # Configure camera
    color_cfg = EasyDict(
        enabled=enable_color, 
        img_w=1280, 
        img_h=720, 
        img_format="bgr8", 
        fps=30
    )
    
    depth_cfg = EasyDict(
        enabled=enable_depth, 
        img_w=1280, 
        img_h=720, 
        img_format="z16", 
        fps=30
    )
    
    pc_cfg = EasyDict(enabled=False)
    
    try:
        # Initialize camera
        print(f"Initializing ZED camera (serial: {serial_number})...")
        camera = ZEDInterface(
            device_id=0,
            color_cfg=color_cfg,
            depth_cfg=depth_cfg,
            pc_cfg=pc_cfg,
            serial_number=serial_number
        )
        
        # Start camera
        print("Starting camera...")
        camera.start()
        
        # Wait for camera to initialize
        print("Waiting for camera to initialize...")
        time.sleep(2)
        
        # Test intrinsics
        print("\nCamera Intrinsics:")
        print("Color intrinsics:", camera.get_color_intrinsics(mode="dict"))
        if enable_depth:
            print("Depth intrinsics:", camera.get_depth_intrinsics(mode="dict"))
        
        # Test image capture
        print("\nTesting image capture...")
        frame_count = 0
        start_time = time.time()
        
        while frame_count < 30:  # Capture 30 frames
            obs = camera.get_last_obs()
            if obs is not None:
                frame_count += 1
                
                if enable_color and "color" in obs:
                    color_img = obs["color"]
                    print(f"Frame {frame_count}: Color image shape: {color_img.shape}")
                    
                    # Display image
                    cv2.imshow("ZED Color", color_img)
                    cv2.waitKey(1)
                
                if enable_depth and "depth" in obs:
                    depth_img = obs["depth"]
                    print(f"Frame {frame_count}: Depth image shape: {depth_img.shape}")
                    
                    # Normalize depth for display
                    depth_display = (depth_img / 1000.0).astype(np.float32)  # Convert mm to meters
                    depth_display = np.clip(depth_display, 0, 10)  # Clip to 0-10 meters
                    depth_display = (depth_display / 10.0 * 255).astype(np.uint8)  # Normalize to 0-255
                    
                    cv2.imshow("ZED Depth", depth_display)
                    cv2.waitKey(1)
            
            time.sleep(0.033)  # ~30 FPS
        
        elapsed_time = time.time() - start_time
        actual_fps = frame_count / elapsed_time
        print(f"\nCaptured {frame_count} frames in {elapsed_time:.2f} seconds")
        print(f"Actual FPS: {actual_fps:.2f}")
        
        # Clean up
        print("\nCleaning up...")
        camera.close()
        cv2.destroyAllWindows()
        
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Test ZED camera implementation")
    parser.add_argument("--serial", type=str, help="Camera serial number")
    parser.add_argument("--no-color", action="store_true", help="Disable color stream")
    parser.add_argument("--no-depth", action="store_true", help="Disable depth stream")
    
    args = parser.parse_args()
    
    test_zed_camera(
        serial_number=args.serial,
        enable_color=not args.no_color,
        enable_depth=not args.no_depth
    )


if __name__ == "__main__":
    main() 