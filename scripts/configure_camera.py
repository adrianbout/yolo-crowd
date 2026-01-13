"""
Interactive Camera Configuration Tool
Helps configure Hikvision cameras for optimal detection
"""

import sys
import json
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from camera_control.hikvision_api import HikvisionCamera, configure_camera_from_profile


def load_config():
    """Load camera and profile configurations"""
    config_dir = Path(__file__).parent.parent / "config"

    with open(config_dir / "cameras.json", 'r') as f:
        cameras = json.load(f)

    with open(config_dir / "camera_profiles.json", 'r') as f:
        profiles = json.load(f)

    return cameras, profiles


def list_cameras(cameras_config):
    """Display all configured cameras"""
    print("\n" + "=" * 60)
    print("CONFIGURED CAMERAS")
    print("=" * 60)

    for idx, camera in enumerate(cameras_config['cameras'], 1):
        print(f"\n{idx}. {camera['name']}")
        print(f"   ID: {camera['id']}")
        print(f"   Type: {camera['type']}")
        print(f"   IP: {camera['connection']['ip']}")
        print(f"   Profile: {camera['profile']}")
        print(f"   Enabled: {camera['enabled']}")


def apply_profile_to_camera(camera_config, profile_config):
    """Apply profile settings to a camera"""
    print(f"\nApplying profile to {camera_config['name']}...")

    connection = camera_config['connection']
    camera = HikvisionCamera(
        ip=connection['ip'],
        username=connection['username'],
        password=connection['password'],
        camera_type=camera_config['type']
    )

    profile = profile_config['profiles'][camera_config['profile']]

    if 'hikvision_settings' in profile:
        success = camera.apply_profile(profile['hikvision_settings'])
        if success:
            print(f"✓ Profile applied successfully to {camera_config['name']}")
        else:
            print(f"✗ Some settings failed to apply")
    else:
        print(f"⚠ No Hikvision settings in profile")


def configure_all_cameras(cameras_config, profiles_config):
    """Configure all enabled cameras"""
    print("\n" + "=" * 60)
    print("CONFIGURING ALL CAMERAS")
    print("=" * 60)

    for camera in cameras_config['cameras']:
        if not camera['enabled']:
            print(f"\nSkipping {camera['name']} (disabled)")
            continue

        apply_profile_to_camera(camera, profiles_config)

    print("\n" + "=" * 60)
    print("Configuration complete!")
    print("=" * 60)


def interactive_menu():
    """Main interactive menu"""
    cameras_config, profiles_config = load_config()

    while True:
        print("\n" + "=" * 60)
        print("SMARTCHAIROUNTER CAMERA CONFIGURATION")
        print("=" * 60)
        print("1. List all cameras")
        print("2. Configure a specific camera")
        print("3. Configure all cameras")
        print("4. View camera current settings")
        print("5. Test camera connection")
        print("0. Exit")
        print("=" * 60)

        choice = input("\nEnter your choice: ").strip()

        if choice == "0":
            print("Exiting...")
            break

        elif choice == "1":
            list_cameras(cameras_config)

        elif choice == "2":
            list_cameras(cameras_config)
            cam_num = input("\nEnter camera number: ").strip()

            try:
                cam_idx = int(cam_num) - 1
                camera = cameras_config['cameras'][cam_idx]
                apply_profile_to_camera(camera, profiles_config)
            except (ValueError, IndexError):
                print("Invalid camera number")

        elif choice == "3":
            confirm = input("Configure all enabled cameras? (y/n): ").strip().lower()
            if confirm == 'y':
                configure_all_cameras(cameras_config, profiles_config)

        elif choice == "4":
            list_cameras(cameras_config)
            cam_num = input("\nEnter camera number: ").strip()

            try:
                cam_idx = int(cam_num) - 1
                camera = cameras_config['cameras'][cam_idx]

                connection = camera['connection']
                cam_api = HikvisionCamera(
                    ip=connection['ip'],
                    username=connection['username'],
                    password=connection['password'],
                    camera_type=camera['type']
                )

                print(f"\nFetching settings for {camera['name']}...")
                settings = cam_api.get_all_settings()

                for name, value in settings.items():
                    print(f"\n{name}:")
                    print(value)

            except (ValueError, IndexError):
                print("Invalid camera number")

        elif choice == "5":
            list_cameras(cameras_config)
            cam_num = input("\nEnter camera number: ").strip()

            try:
                cam_idx = int(cam_num) - 1
                camera = cameras_config['cameras'][cam_idx]

                connection = camera['connection']
                print(f"\nTesting connection to {camera['name']}...")
                print(f"IP: {connection['ip']}")
                print(f"RTSP URL: {connection['rtsp_url']}")

                # Test RTSP connection with OpenCV
                import cv2
                cap = cv2.VideoCapture(connection['rtsp_url'])

                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        print("✓ RTSP connection successful!")
                        print(f"  Frame size: {frame.shape}")
                    else:
                        print("✗ Failed to read frame")
                    cap.release()
                else:
                    print("✗ Failed to open RTSP stream")

            except (ValueError, IndexError):
                print("Invalid camera number")
            except Exception as e:
                print(f"Error testing connection: {e}")

        else:
            print("Invalid choice")


if __name__ == "__main__":
    try:
        interactive_menu()
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
