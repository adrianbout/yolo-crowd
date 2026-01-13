"""
State Manager
In-memory state management for counts, ROIs, and camera settings
"""

import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import threading

logger = logging.getLogger(__name__)


class StateManager:
    """
    Manages in-memory state for the application
    - Current counts per camera
    - ROI configurations
    - Camera settings
    - Detection history (optional, limited buffer)
    """

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.lock = threading.Lock()

        # Current counts
        self.counts: Dict[str, int] = {}
        self.total_count: int = 0

        # Timestamps
        self.last_update: Dict[str, datetime] = {}
        self.last_update_total: Optional[datetime] = None

        # ROI configuration
        self.rois: Dict = {}

        # Camera settings
        self.cameras: Dict = {}
        self.profiles: Dict = {}

        # Detection history (limited buffer)
        self.detection_history: Dict[str, List] = {}
        self.history_max_length = 100

    def load_configuration(self):
        """Load all configuration files"""
        logger.info("Loading configuration files...")

        # Load cameras
        cameras_file = self.config_dir / "cameras.json"
        if cameras_file.exists():
            with open(cameras_file, 'r') as f:
                self.cameras = json.load(f)
            logger.info(f"Loaded cameras configuration")

        # Load profiles
        profiles_file = self.config_dir / "camera_profiles.json"
        if profiles_file.exists():
            with open(profiles_file, 'r') as f:
                self.profiles = json.load(f)
            logger.info(f"Loaded camera profiles")

        # Load ROIs
        rois_file = self.config_dir / "rois.json"
        if rois_file.exists():
            with open(rois_file, 'r') as f:
                self.rois = json.load(f)
            logger.info(f"Loaded ROI configuration")

        # Initialize counts for all cameras
        for camera in self.cameras.get("cameras", []):
            camera_id = camera["id"]
            self.counts[camera_id] = 0
            self.detection_history[camera_id] = []

    def update_count(self, camera_id: str, count: int):
        """
        Update count for a camera
        Args:
            camera_id: Camera identifier
            count: New count value
        """
        with self.lock:
            self.counts[camera_id] = count
            self.last_update[camera_id] = datetime.now()

            # Update total
            self.total_count = sum(self.counts.values())
            self.last_update_total = datetime.now()

    def update_counts_batch(self, counts: Dict[str, int]):
        """
        Update counts for multiple cameras
        Args:
            counts: Dictionary mapping camera_id to count
        """
        with self.lock:
            current_time = datetime.now()

            for camera_id, count in counts.items():
                self.counts[camera_id] = count
                self.last_update[camera_id] = current_time

            self.total_count = sum(self.counts.values())
            self.last_update_total = current_time

    def get_count(self, camera_id: str) -> Optional[int]:
        """
        Get count for a camera
        Args:
            camera_id: Camera identifier
        Returns:
            Count or None if camera not found
        """
        with self.lock:
            return self.counts.get(camera_id)

    def get_all_counts(self) -> Dict[str, int]:
        """
        Get counts for all cameras
        Returns:
            Dictionary mapping camera_id to count
        """
        with self.lock:
            return self.counts.copy()

    def get_total_count(self) -> int:
        """
        Get total count across all cameras
        Returns:
            Total count
        """
        with self.lock:
            return self.total_count

    def get_counts_summary(self) -> Dict:
        """
        Get complete counts summary
        Returns:
            Summary dictionary with counts and timestamps
        """
        with self.lock:
            return {
                "total": self.total_count,
                "by_camera": self.counts.copy(),
                "last_update_total": self.last_update_total.isoformat() if self.last_update_total else None,
                "last_update_by_camera": {
                    cam_id: ts.isoformat() for cam_id, ts in self.last_update.items()
                }
            }

    def add_detection_to_history(self, camera_id: str, detections: List[Dict]):
        """
        Add detections to history buffer
        Args:
            camera_id: Camera identifier
            detections: List of detection dictionaries
        """
        with self.lock:
            if camera_id not in self.detection_history:
                self.detection_history[camera_id] = []

            entry = {
                "timestamp": datetime.now().isoformat(),
                "count": len(detections),
                "detections": detections
            }

            self.detection_history[camera_id].append(entry)

            # Limit history length
            if len(self.detection_history[camera_id]) > self.history_max_length:
                self.detection_history[camera_id] = self.detection_history[camera_id][-self.history_max_length:]

    def get_detection_history(self, camera_id: str, limit: int = 10) -> List[Dict]:
        """
        Get recent detection history
        Args:
            camera_id: Camera identifier
            limit: Maximum number of entries to return
        Returns:
            List of detection history entries
        """
        with self.lock:
            history = self.detection_history.get(camera_id, [])
            return history[-limit:]

    def update_roi(self, camera_id: str, roi_config: Dict):
        """
        Update ROI configuration for a camera
        Args:
            camera_id: Camera identifier
            roi_config: ROI configuration
        """
        with self.lock:
            if "rois" not in self.rois:
                self.rois["rois"] = {}

            self.rois["rois"][camera_id] = roi_config

    def get_roi(self, camera_id: str) -> Optional[Dict]:
        """
        Get ROI configuration for a camera
        Args:
            camera_id: Camera identifier
        Returns:
            ROI configuration or None
        """
        with self.lock:
            return self.rois.get("rois", {}).get(camera_id)

    def get_all_rois(self) -> Dict:
        """
        Get all ROI configurations
        Returns:
            Complete ROI configuration
        """
        with self.lock:
            return self.rois.copy()

    def save_rois(self):
        """Save ROI configuration to file"""
        with self.lock:
            rois_file = self.config_dir / "rois.json"
            with open(rois_file, 'w') as f:
                json.dump(self.rois, f, indent=2)
            logger.info("Saved ROI configuration to file")

    def get_camera_config(self, camera_id: str) -> Optional[Dict]:
        """
        Get configuration for a camera
        Args:
            camera_id: Camera identifier
        Returns:
            Camera configuration or None
        """
        for camera in self.cameras.get("cameras", []):
            if camera["id"] == camera_id:
                return camera
        return None

    def get_camera_profile(self, camera_id: str) -> Optional[Dict]:
        """
        Get profile for a camera
        Args:
            camera_id: Camera identifier
        Returns:
            Profile configuration or None
        """
        camera = self.get_camera_config(camera_id)
        if not camera:
            return None

        profile_name = camera.get("profile")
        if not profile_name:
            return None

        return self.profiles.get("profiles", {}).get(profile_name)

    def get_all_cameras(self) -> List[Dict]:
        """
        Get all camera configurations
        Returns:
            List of camera configurations
        """
        return self.cameras.get("cameras", [])

    def get_detection_settings(self) -> Dict:
        """
        Get global detection settings
        Returns:
            Detection settings
        """
        return self.profiles.get("detection_settings", {})

    def manual_override_count(self, camera_id: str, count: int):
        """
        Manual override for count (from API)
        Args:
            camera_id: Camera identifier
            count: New count value
        """
        logger.info(f"Manual override: Camera {camera_id} count set to {count}")
        self.update_count(camera_id, count)

    def reset_counts(self):
        """Reset all counts to zero"""
        with self.lock:
            for camera_id in self.counts.keys():
                self.counts[camera_id] = 0
            self.total_count = 0
            self.last_update_total = datetime.now()
            logger.info("Reset all counts to zero")

    def get_system_status(self) -> Dict:
        """
        Get overall system status
        Returns:
            System status dictionary
        """
        with self.lock:
            return {
                "total_cameras": len(self.cameras.get("cameras", [])),
                "enabled_cameras": len([c for c in self.cameras.get("cameras", []) if c.get("enabled", True)]),
                "total_count": self.total_count,
                "last_update": self.last_update_total.isoformat() if self.last_update_total else None,
                "cameras_with_roi": len([k for k, v in self.rois.get("rois", {}).items() if v.get("enabled", False)])
            }
