#!/usr/bin/env python3
"""
Convenience script to run ZED camera tests.
"""

import argparse
import subprocess
import sys
import os


def run_test(test_type, serial_number=None, no_color=False, no_depth=False):
    """Run a specific test."""
    if test_type == "basic":
        script = "test_zed_camera.py"
    elif test_type == "integration":
        script = "test_zed_integration.py"
    else:
        print(f"Unknown test type: {test_type}")
        return False
    
    cmd = ["python", script]
    
    if serial_number:
        cmd.extend(["--serial", serial_number])
    
    if no_color:
        cmd.append("--no-color")
    
    if no_depth:
        cmd.append("--no-depth")
    
    print(f"Running: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Test failed with return code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run ZED camera tests")
    parser.add_argument("--test", choices=["basic", "integration", "both"], 
                       default="integration", help="Which test to run")
    parser.add_argument("--serial", type=str, help="Camera serial number")
    parser.add_argument("--no-color", action="store_true", help="Disable color stream")
    parser.add_argument("--no-depth", action="store_true", help="Disable depth stream")
    parser.add_argument("--list-cameras", action="store_true", help="List available cameras")
    
    args = parser.parse_args()
    
    # Add parent directory to path for imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.join(current_dir, '..', '..')
    sys.path.insert(0, parent_dir)
    
    if args.list_cameras:
        try:
            from rpl_vision_utils.zed import print_zed_cameras
            print_zed_cameras()
            return
        except ImportError as e:
            print(f"Error importing ZED utilities: {e}")
            return
    
    if not args.serial:
        print("Warning: No serial number provided. Using first available camera.")
        print("Use --list-cameras to see available cameras.")
        print("Use --serial <number> to specify a camera.")
        print()
    
    success = True
    
    if args.test == "basic":
        success = run_test("basic", args.serial, args.no_color, args.no_depth)
    elif args.test == "integration":
        success = run_test("integration", args.serial, args.no_color, args.no_depth)
    elif args.test == "both":
        print("Running both tests...")
        print("=" * 60)
        
        print("1. Running integration test...")
        success1 = run_test("integration", args.serial, args.no_color, args.no_depth)
        
        print("\n" + "=" * 60)
        print("2. Running basic test...")
        success2 = run_test("basic", args.serial, args.no_color, args.no_depth)
        
        success = success1 and success2
    
    if success:
        print("\n✅ All tests completed successfully!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 