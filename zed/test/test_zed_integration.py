#!/usr/bin/env python3
"""
Simple integration test for ZED camera with the framework.
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

from rpl_vision_utils.zed import ZEDInterface


def test_zed_integration(serial_number=None, enable_depth=True, enable_color=True):
    """
    Test ZED camera integration with the framework.
    """
    print("Testing ZED Camera Integration")
    print("=" * 40)
    
    # Configure camera streams
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
        # Initialize camera (same as in run_camera_node.py)
        print(f"Initializing ZED camera (serial: {serial_number})...")
        camera_interface = ZEDInterface(
            device_id=0,
            color_cfg=color_cfg,
            depth_cfg=depth_cfg,
            pc_cfg=pc_cfg,
            serial_number=serial_number
        )
        
        # Start camera
        print("Starting camera...")
        camera_interface.start()
        
        # Wait for camera to initialize
        print("Waiting for camera to initialize...")
        time.sleep(2)
        
        # Test intrinsics (same as in run_camera_node.py)
        print("\nCamera Intrinsics:")
        if enable_color:
            color_intrinsics = camera_interface.get_color_intrinsics(mode="dict")
            print("Color intrinsics:", color_intrinsics)
        
        if enable_depth:
            depth_intrinsics = camera_interface.get_depth_intrinsics(mode="dict")
            print("Depth intrinsics:", depth_intrinsics)
        
        # Test image capture (same pattern as run_camera_node.py)
        print("\nTesting image capture...")
        frame_count = 0
        start_time = time.time()
        
        while frame_count < 10:  # Capture 10 frames
            capture = camera_interface.get_last_obs()
            
            if capture is not None:
                frame_count += 1
                
                if enable_color and "color" in capture:
                    color_img = capture["color"]
                    print(f"Frame {frame_count}: Color image shape: {color_img.shape}")
                
                if enable_depth and "depth" in capture:
                    depth_img = capture["depth"]
                    print(f"Frame {frame_count}: Depth image shape: {depth_img.shape}")
                    print(f"Frame {frame_count}: Depth range: {depth_img.min()}-{depth_img.max()} mm")
            
            time.sleep(0.1)  # 10 FPS for testing
        
        elapsed_time = time.time() - start_time
        actual_fps = frame_count / elapsed_time
        print(f"\nCaptured {frame_count} frames in {elapsed_time:.2f} seconds")
        print(f"Actual FPS: {actual_fps:.2f}")
        
        # Clean up
        print("\nCleaning up...")
        camera_interface.close()
        
        print("Integration test completed successfully!")
        print("✅ ZED camera integration with framework is working!")
        
    except Exception as e:
        print(f"Error during integration test: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Test ZED camera integration with framework")
    parser.add_argument("--serial", type=str, help="Camera serial number")
    parser.add_argument("--no-color", action="store_true", help="Disable color stream")
    parser.add_argument("--no-depth", action="store_true", help="Disable depth stream")
    
    args = parser.parse_args()
    
    test_zed_integration(
        serial_number=args.serial,
        enable_color=not args.no_color,
        enable_depth=not args.no_depth
    )


if __name__ == "__main__":
    main() 