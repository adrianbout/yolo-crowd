"""
Test all module imports
Quick verification that all Python modules can be imported
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

def test_import(module_name, import_statement):
    """Test a single import"""
    try:
        exec(import_statement)
        print(f"OK   - {module_name}")
        return True
    except Exception as e:
        print(f"FAIL - {module_name}: {e}")
        return False


def main():
    """Test all imports"""
    print("=" * 60)
    print("TESTING MODULE IMPORTS")
    print("=" * 60)

    tests = [
        ("camera_stream", "from camera_control.camera_stream import CameraStreamManager"),
        ("hikvision_api", "from camera_control.hikvision_api import HikvisionCamera"),
        ("detector", "from detection.detector import YOLODetector, Detection"),
        ("roi_filter", "from detection.roi_filter import ROIFilter, ROIPolygon"),
        ("state_manager", "from services.state_manager import StateManager"),
        ("detection_service", "from services.detection_service import DetectionService"),
        ("api.cameras", "from api import cameras"),
        ("api.roi", "from api import roi"),
        ("api.counting", "from api import counting"),
        ("api.websocket", "from api import websocket"),
    ]

    results = []
    for name, import_stmt in tests:
        results.append(test_import(name, import_stmt))

    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"RESULT: {passed}/{total} imports successful")
    print("=" * 60)

    if all(results):
        print("\nAll imports working! System is ready.")
        return 0
    else:
        print("\nSome imports failed. Check error messages above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
