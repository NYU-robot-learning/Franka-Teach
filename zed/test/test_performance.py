#!/usr/bin/env python3
"""
Performance test for ZED camera to measure actual frame rate.
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


def test_performance(serial_number=None, enable_depth=True, enable_color=True, duration=10):
    """
    Test ZED camera performance and measure actual frame rate.
    
    Args:
        serial_number (str): Camera serial number (optional)
        enable_depth (bool): Enable depth stream
        enable_color (bool): Enable color stream
        duration (int): Test duration in seconds
    """
    print("ZED Camera Performance Test")
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
        # Initialize camera
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
        
        # Performance test
        print(f"\nRunning performance test for {duration} seconds...")
        print("Capturing frames as fast as possible...")
        
        frame_count = 0
        start_time = time.time()
        last_time = start_time
        
        # Capture frames for the specified duration
        last_frame_hash = None
        unique_frames = 0
        
        while time.time() - start_time < duration:
            capture = camera_interface.get_last_obs()
            
            if capture is not None:
                frame_count += 1
                current_time = time.time()
                
                # Check if this is a unique frame by hashing the color image
                if enable_color and "color" in capture:
                    color_img = capture["color"]
                    frame_hash = hash(color_img.tobytes())
                    if frame_hash != last_frame_hash:
                        unique_frames += 1
                        last_frame_hash = frame_hash
                
                # Print progress every second
                if int(current_time - last_time) >= 1:
                    elapsed = current_time - start_time
                    current_fps = frame_count / elapsed
                    unique_fps = unique_frames / elapsed
                    print(f"Progress: {elapsed:.1f}s/{duration}s, Total calls: {frame_count}, Unique frames: {unique_frames}, Call FPS: {current_fps:.1f}, Unique FPS: {unique_fps:.1f}")
                    last_time = current_time
                
                # Optional: Print frame info for first few frames
                if frame_count <= 3:
                    if enable_color and "color" in capture:
                        color_img = capture["color"]
                        print(f"Frame {frame_count}: Color shape: {color_img.shape}")
                    
                    if enable_depth and "depth" in capture:
                        depth_img = capture["depth"]
                        print(f"Frame {frame_count}: Depth shape: {depth_img.shape}")
        
        # Calculate final statistics
        end_time = time.time()
        total_time = end_time - start_time
        total_fps = frame_count / total_time
        unique_fps = unique_frames / total_time
        
        print(f"\nPerformance Test Results:")
        print(f"Total calls to get_last_obs(): {frame_count}")
        print(f"Unique frames captured: {unique_frames}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Call FPS: {total_fps:.2f}")
        print(f"Unique FPS: {unique_fps:.2f}")
        print(f"Target FPS: 30.0")
        print(f"Performance: {(unique_fps/30.0)*100:.1f}% of target")
        
        if unique_fps >= 25:
            print("✅ Performance is good!")
        elif unique_fps >= 20:
            print("⚠️  Performance is acceptable but below target")
        else:
            print("❌ Performance is below acceptable threshold")
        
        # Clean up
        print("\nCleaning up...")
        camera_interface.close()
        
        print("Performance test completed!")
        
    except Exception as e:
        print(f"Error during performance test: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Test ZED camera performance")
    parser.add_argument("--serial", type=str, help="Camera serial number")
    parser.add_argument("--no-color", action="store_true", help="Disable color stream")
    parser.add_argument("--no-depth", action="store_true", help="Disable depth stream")
    parser.add_argument("--duration", type=int, default=10, help="Test duration in seconds")
    
    args = parser.parse_args()
    
    test_performance(
        serial_number=args.serial,
        enable_color=not args.no_color,
        enable_depth=not args.no_depth,
        duration=args.duration
    )


if __name__ == "__main__":
    main() 