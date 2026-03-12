"""
Project setup verification script.
Tests imports and basic functionality without requiring camera.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def verify_imports():
    """Verify all required packages can be imported."""
    print("🔍 Checking imports...")
    
    required_packages = [
        ("cv2", "opencv-python"),
        ("mediapipe", "mediapipe"),
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
    ]
    
    failed = []
    for module_name, package_name in required_packages:
        try:
            __import__(module_name)
            print(f"   ✅ {module_name}")
        except ImportError:
            print(f"   ❌ {module_name} (install: pip install {package_name})")
            failed.append(package_name)
    
    return len(failed) == 0, failed

def verify_project_structure():
    """Verify project directory structure."""
    print("\n🔍 Checking project structure...")
    
    required_dirs = [
        "src",
        "data/raw",
        "data/processed",
        "models/checkpoints",
    ]
    
    required_files = [
        "config.py",
        "requirements.txt",
        "collect_gestures.py",
        "src/__init__.py",
        "src/mediapipe_extractor.py",
        "src/utils.py",
        "README.md",
    ]
    
    all_good = True
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"   ✅ {dir_name}/")
        else:
            print(f"   ❌ {dir_name}/ (missing)")
            all_good = False
    
    for file_name in required_files:
        file_path = project_root / file_name
        if file_path.exists():
            print(f"   ✅ {file_name}")
        else:
            print(f"   ❌ {file_name} (missing)")
            all_good = False
    
    return all_good

def verify_imports_local():
    """Verify local module imports."""
    print("\n🔍 Checking local module imports...")
    
    try:
        import config
        print(f"   ✅ config (gestures: {', '.join(config.GESTURE_NAMES)})")
    except ImportError as e:
        print(f"   ❌ config: {e}")
        return False
    
    try:
        from src.mediapipe_extractor import HandLandmarkExtractor
        print(f"   ✅ src.mediapipe_extractor")
    except ImportError as e:
        print(f"   ❌ src.mediapipe_extractor: {e}")
        return False
    
    try:
        from src.utils import (
            draw_landmarks,
            draw_fps,
            pad_landmarks_sequence,
            create_gesture_directories,
        )
        print(f"   ✅ src.utils")
    except ImportError as e:
        print(f"   ❌ src.utils: {e}")
        return False
    
    return True

def main():
    """Run all verification checks."""
    print("=" * 70)
    print("HAND GESTURE RECOGNITION - PROJECT VERIFICATION")
    print("=" * 70)
    
    print(f"\nProject root: {project_root}\n")
    
    # Check structure
    struct_ok = verify_project_structure()
    
    # Check imports
    imports_ok, failed_packages = verify_imports()
    
    # Check local imports
    local_ok = verify_imports_local()
    
    print("\n" + "=" * 70)
    if struct_ok and imports_ok and local_ok:
        print("✅ ALL CHECKS PASSED! Project is ready to use.")
        print("\nNext steps:")
        print("  1. Install dependencies if not already done:")
        print("     pip install -r requirements.txt")
        print("  2. Start data collection:")
        print("     python collect_gestures.py --mode interactive")
        print("  3. View collection statistics anytime:")
        print("     python collect_gestures.py --stats")
    else:
        print("❌ SOME CHECKS FAILED")
        if not imports_ok:
            print(f"\nMissing packages: {', '.join(failed_packages)}")
            print("Install them with:")
            print(f"  pip install {' '.join(failed_packages)}")
        if not struct_ok:
            print("\n⚠️  Check directory structure and re-run setup")
    
    print("=" * 70)
    
    return struct_ok and imports_ok and local_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
