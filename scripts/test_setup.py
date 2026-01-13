"""
Setup Test Script
Verifies that the system is properly configured
"""

import sys
import json
from pathlib import Path
import importlib.util

def test_python_version():
    """Test Python version"""
    print("Testing Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} - Need 3.8+")
        return False


def test_package(package_name, import_name=None):
    """Test if a package is installed"""
    if import_name is None:
        import_name = package_name

    spec = importlib.util.find_spec(import_name)
    if spec is not None:
        print(f"✓ {package_name} - Installed")
        return True
    else:
        print(f"✗ {package_name} - Not installed")
        return False


def test_dependencies():
    """Test all required dependencies"""
    print("\nTesting dependencies...")

    packages = [
        ("PyTorch", "torch"),
        ("TorchVision", "torchvision"),
        ("OpenCV", "cv2"),
        ("NumPy", "numpy"),
        ("FastAPI", "fastapi"),
        ("Uvicorn", "uvicorn"),
        ("Requests", "requests"),
        ("Pydantic", "pydantic"),
        ("WebSockets", "websockets")
    ]

    results = []
    for name, import_name in packages:
        results.append(test_package(name, import_name))

    return all(results)


def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("✗ CUDA not available - Will use CPU (very slow!)")
            return False
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")
        return False


def test_config_files():
    """Test configuration files"""
    print("\nTesting configuration files...")

    config_dir = Path(__file__).parent.parent / "config"
    files = ["cameras.json", "camera_profiles.json", "rois.json"]

    results = []
    for filename in files:
        filepath = config_dir / filename
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                print(f"✓ {filename} - Valid JSON")
                results.append(True)
            except json.JSONDecodeError as e:
                print(f"✗ {filename} - Invalid JSON: {e}")
                results.append(False)
        else:
            print(f"✗ {filename} - Not found")
            results.append(False)

    return all(results)


def test_model():
    """Test YOLO model"""
    print("\nTesting YOLO model...")

    config_dir = Path(__file__).parent.parent / "config"
    with open(config_dir / "camera_profiles.json", 'r') as f:
        profiles = json.load(f)

    model_path = Path(__file__).parent.parent / profiles["detection_settings"]["model_path"]

    if model_path.exists():
        print(f"✓ Model file found: {model_path}")
        print(f"  Size: {model_path.stat().st_size / 1e6:.1f} MB")
        return True
    else:
        print(f"✗ Model file not found: {model_path}")
        print("  Place your trained model at: weights/yolo-crowd.pt")
        return False


def test_cameras_config():
    """Test camera configuration"""
    print("\nTesting camera configuration...")

    config_dir = Path(__file__).parent.parent / "config"
    with open(config_dir / "cameras.json", 'r') as f:
        cameras = json.load(f)

    num_cameras = len(cameras["cameras"])
    enabled_cameras = len([c for c in cameras["cameras"] if c.get("enabled", True)])

    print(f"  Total cameras: {num_cameras}")
    print(f"  Enabled cameras: {enabled_cameras}")

    if enabled_cameras == 0:
        print("⚠ No cameras enabled")
        return False

    # Check camera configuration
    for camera in cameras["cameras"]:
        if not camera.get("enabled", True):
            continue

        print(f"\n  Camera: {camera['name']}")
        print(f"    Type: {camera['type']}")
        print(f"    IP: {camera['connection']['ip']}")

        # Check required fields
        required = ["id", "name", "type", "connection"]
        missing = [f for f in required if f not in camera]
        if missing:
            print(f"    ✗ Missing fields: {missing}")
            return False

        required_conn = ["ip", "rtsp_url", "username", "password"]
        missing_conn = [f for f in required_conn if f not in camera["connection"]]
        if missing_conn:
            print(f"    ✗ Missing connection fields: {missing_conn}")
            return False

        print(f"    ✓ Configuration valid")

    return True


def test_directory_structure():
    """Test directory structure"""
    print("\nTesting directory structure...")

    base_dir = Path(__file__).parent.parent
    required_dirs = [
        "config",
        "backend",
        "backend/api",
        "backend/detection",
        "backend/camera_control",
        "backend/services",
        "frontend",
        "frontend/js",
        "frontend/css",
        "weights"
    ]

    results = []
    for dir_path in required_dirs:
        full_path = base_dir / dir_path
        if full_path.exists():
            print(f"✓ {dir_path}")
            results.append(True)
        else:
            print(f"✗ {dir_path} - Not found")
            results.append(False)

    return all(results)


def main():
    """Run all tests"""
    print("=" * 60)
    print("SMARTCHAIROUNTER SETUP TEST")
    print("=" * 60)

    tests = [
        ("Python Version", test_python_version),
        ("Dependencies", test_dependencies),
        ("CUDA Support", test_cuda),
        ("Directory Structure", test_directory_structure),
        ("Config Files", test_config_files),
        ("YOLO Model", test_model),
        ("Camera Configuration", test_cameras_config)
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n✗ {name} test failed with error: {e}")
            results[name] = False

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} - {name}")

    passed = sum(results.values())
    total = len(results)

    print("\n" + "=" * 60)
    if all(results.values()):
        print("✓ ALL TESTS PASSED - System ready!")
        print("\nTo start the application, run:")
        print("  python main.py")
    else:
        print(f"✗ {total - passed} TEST(S) FAILED")
        print("\nPlease fix the issues above before starting the application.")
        print("See QUICKSTART.md for setup instructions.")

    print("=" * 60)

    return all(results.values())


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
