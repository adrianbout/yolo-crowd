"""
Project Initialization Script
Creates necessary directories and placeholder files
"""

import os
from pathlib import Path
import json


def create_directory_structure():
    """Create all necessary directories"""
    base_dir = Path(__file__).parent.parent

    directories = [
        "config",
        "backend",
        "backend/api",
        "backend/detection",
        "backend/camera_control",
        "backend/services",
        "frontend",
        "frontend/css",
        "frontend/js",
        "scripts",
        "weights",
        "logs",
        "data"
    ]

    print("Creating directory structure...")
    for dir_path in directories:
        full_path = base_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ {dir_path}")

    print("\nDirectory structure created!")


def check_config_files():
    """Check if configuration files exist"""
    base_dir = Path(__file__).parent.parent
    config_dir = base_dir / "config"

    print("\nChecking configuration files...")

    # Check cameras.json
    cameras_file = config_dir / "cameras.json"
    if not cameras_file.exists():
        print("⚠ cameras.json not found")
        print("  Please create it based on the template in the README")
    else:
        print("✓ cameras.json exists")

    # camera_profiles.json should exist (we created it)
    profiles_file = config_dir / "camera_profiles.json"
    if profiles_file.exists():
        print("✓ camera_profiles.json exists")
    else:
        print("⚠ camera_profiles.json not found")

    # rois.json should exist (we created it)
    rois_file = config_dir / "rois.json"
    if rois_file.exists():
        print("✓ rois.json exists")
    else:
        print("⚠ rois.json not found")


def check_model():
    """Check if YOLO model exists"""
    base_dir = Path(__file__).parent.parent
    config_dir = base_dir / "config"

    print("\nChecking YOLO model...")

    # Read model path from config
    try:
        with open(config_dir / "camera_profiles.json", 'r') as f:
            profiles = json.load(f)
        model_path = base_dir / profiles["detection_settings"]["model_path"]

        if model_path.exists():
            size_mb = model_path.stat().st_size / 1e6
            print(f"✓ Model found: {model_path}")
            print(f"  Size: {size_mb:.1f} MB")
        else:
            print(f"⚠ Model not found: {model_path}")
            print(f"  Please place your trained YOLO model at this location")

    except Exception as e:
        print(f"⚠ Error checking model: {e}")


def show_next_steps():
    """Display next steps"""
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("""
1. Edit config/cameras.json with your camera information
   - Update IP addresses
   - Set RTSP URLs
   - Configure credentials

2. Place your YOLO model at: weights/yolo-crowd.pt

3. Run the setup test:
   python scripts/test_setup.py

4. Configure your cameras (optional):
   python scripts/configure_camera.py

5. Start the application:
   python main.py

6. Access the dashboard:
   http://localhost:8000

For detailed instructions, see:
- QUICKSTART.md - Quick start guide
- README.md - Complete documentation
- PROJECT_STRUCTURE.md - Project structure reference
""")


def main():
    """Main function"""
    print("=" * 60)
    print("SMARTCHAIROUNTER PROJECT INITIALIZATION")
    print("=" * 60)

    create_directory_structure()
    check_config_files()
    check_model()
    show_next_steps()

    print("=" * 60)
    print("Initialization complete!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError during initialization: {e}")
        import traceback
        traceback.print_exc()
