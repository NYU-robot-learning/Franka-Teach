#!/usr/bin/env python3
"""
Test script to verify ZED stereo camera functionality.
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


def test_stereo(serial_number=None, stereo_mode='separate'):
    """
    Test ZED stereo camera functionality.
    
    Args:
        serial_number (str): Camera serial number (optional)
        stereo_mode (str): Stereo mode ('separate', 'sbs', 'left_only')
    """
    print(f"Testing ZED Stereo Camera - Mode: {stereo_mode}")
    print("=" * 50)
    
    # Configure camera streams
    color_cfg = EasyDict(
        enabled=True, 
        img_w=1280, 
        img_h=720, 
        img_format="bgr8", 
        fps=30
    )
    
    depth_cfg = EasyDict(
        enabled=False, 
        img_w=1280, 
        img_h=720, 
        img_format="z16", 
        fps=30
    )
    
    pc_cfg = EasyDict(enabled=False)
    
    try:
        # Initialize camera
        print(f"Initializing ZED camera (serial: {serial_number})...")
        camera_interface = ZEDInterface(
            device_id=0,
            color_cfg=color_cfg,
            depth_cfg=depth_cfg,
            pc_cfg=pc_cfg,
            serial_number=serial_number,
            stereo_mode=stereo_mode
        )
        
        # Start camera
        print("Starting camera...")
        camera_interface.start()
        
        # Wait for camera to initialize
        print("Waiting for camera to initialize...")
        time.sleep(2)
        
        # Test image capture
        print("\nTesting stereo image capture...")
        frame_count = 0
        
        while frame_count < 5:  # Capture 5 frames
            capture = camera_interface.get_last_obs()
            
            if capture is not None:
                frame_count += 1
                print(f"\nFrame {frame_count}:")
                
                if stereo_mode == 'separate':
                    if "color_left" in capture:
                        color_left = capture["color_left"]
                        print(f"  Left image shape: {color_left.shape}")
                    if "color_right" in capture:
                        color_right = capture["color_right"]
                        print(f"  Right image shape: {color_right.shape}")
                        
                elif stereo_mode == 'sbs':
                    if "color_sbs" in capture:
                        color_sbs = capture["color_sbs"]
                        print(f"  Side-by-side image shape: {color_sbs.shape}")
                        
                elif stereo_mode == 'left_only':
                    if "color" in capture:
                        color = capture["color"]
                        print(f"  Left image shape: {color.shape}")
            
            time.sleep(0.5)  # Wait 0.5 seconds between frames
        
        # Clean up
        print("\nCleaning up...")
        camera_interface.close()
        
        print("Stereo test completed successfully!")
        
    except Exception as e:
        print(f"Error during stereo test: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Test ZED stereo camera functionality")
    parser.add_argument("--serial", type=str, help="Camera serial number")
    parser.add_argument("--mode", choices=["separate", "sbs", "left_only"], 
                       default="separate", help="Stereo mode to test")
    
    args = parser.parse_args()
    
    test_stereo(
        serial_number=args.serial,
        stereo_mode=args.mode
    )


if __name__ == "__main__":
    main() 