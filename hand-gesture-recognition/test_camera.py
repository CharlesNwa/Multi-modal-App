#!/usr/bin/env python3
"""Test script to diagnose camera and display issues."""

import cv2
import sys

print("🔍 Testing Camera and Display...")
print("=" * 60)

# Test 1: Check OpenCV
print("\n1️⃣  Testing OpenCV...")
print(f"   OpenCV version: {cv2.__version__}")

# Test 2: Try to open camera
print("\n2️⃣  Testing Camera Access...")
for i in range(3):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret:
            print(f"   ✅ Camera {i}: Working! ({frame.shape[1]}x{frame.shape[0]})")
            camera_index = i
            break
        else:
            print(f"   ⚠️  Camera {i}: Opened but no frames")
    else:
        print(f"   ❌ Camera {i}: Not available")
else:
    print("   ❌ No cameras found!")
    sys.exit(1)

# Test 3: Try to display
print("\n3️⃣  Testing Display Window...")
cap = cv2.VideoCapture(camera_index)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print("   ✅ Frame captured successfully")
        print(f"   Frame size: {frame.shape}")
        
        # Try to show frame
        print("   📺 Attempting to display window...")
        print("   (Window should appear - press 'q' to close)")
        
        try:
            cv2.imshow("TEST: Hand Gesture Recognition", frame)
            print("   ✅ Window created!")
            
            # Wait for keypress
            print("   ⏳ Waiting for input... (Press any key or 'q' to exit)")
            key = cv2.waitKey(5000) & 0xFF  # Wait 5 seconds
            if key == ord('q'):
                print("   ✅ User closed window")
            else:
                print("   ⏰ Timeout reached")
            
            cv2.destroyAllWindows()
            print("   ✅ Window closed")
            
        except Exception as e:
            print(f"   ❌ Display error: {e}")
    else:
        print("   ❌ Could not capture frame")
cap.release()

print("\n" + "=" * 60)
print("✅ All tests completed!")
print("\nIf you see a window above, the camera and display are working.")
print("If you don't see a window, there may be a display server issue.")
