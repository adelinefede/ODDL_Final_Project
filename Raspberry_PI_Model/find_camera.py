#!/usr/bin/env python3
import cv2
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description='Find available cameras')
    parser.add_argument('--max_cameras', type=int, default=5, 
                      help='Maximum number of cameras to check')
    args = parser.parse_args()
    
    print("Searching for available cameras...")
    
    
    found_cameras = []
    
    for i in range(args.max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                found_cameras.append(i)
                h, w = frame.shape[:2]
                print(f"Camera {i}: Available - Resolution: {w}x{h}")
                
            else:
                print(f"Camera {i}: Available but couldn't capture frame")
            cap.release()
        else:
            print(f"Camera {i}: Not available")
    
    print("\nResults:")
    if found_cameras:
        print(f"Found {len(found_cameras)} camera(s): {found_cameras}")
        print("\nTo use a specific camera with the posture model, run:")
        print(f"python3 run_posture_model.py --camera {found_cameras[0]}")
        
    else:
        print("No cameras found. Check your USB connection.")

if __name__ == "__main__":
    main() 