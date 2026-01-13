"""
ROI (Region of Interest) Filter
Filters detections based on polygon masks to handle camera overlaps
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from .detector import Detection

logger = logging.getLogger(__name__)


class ROIPolygon:
    """Represents a Region of Interest polygon"""

    def __init__(self, name: str, points: List[List[int]], description: str = ""):
        self.name = name
        self.points = np.array(points, dtype=np.int32)
        self.description = description

    def contains_point(self, point: Tuple[float, float]) -> bool:
        """
        Check if a point is inside the polygon
        Args:
            point: (x, y) coordinates
        Returns:
            True if point is inside polygon
        """
        result = cv2.pointPolygonTest(self.points, point, False)
        return result >= 0  # >= 0 means inside or on the edge

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "name": self.name,
            "points": self.points.tolist(),
            "description": self.description
        }


class ROIFilter:
    """
    Filters detections based on ROI polygons
    """

    def __init__(self):
        self.rois_by_camera: Dict[str, List[ROIPolygon]] = {}
        self.roi_enabled: Dict[str, bool] = {}

    def load_rois(self, rois_config: Dict):
        """
        Load ROIs from configuration
        Args:
            rois_config: ROI configuration from rois.json
        """
        logger.info("Loading ROI configurations")

        self.rois_by_camera.clear()
        self.roi_enabled.clear()

        for camera_id, roi_data in rois_config.get("rois", {}).items():
            self.roi_enabled[camera_id] = roi_data.get("enabled", False)

            if not self.roi_enabled[camera_id]:
                logger.info(f"ROI disabled for camera {camera_id}")
                continue

            polygons = []
            for poly_data in roi_data.get("polygons", []):
                polygon = ROIPolygon(
                    name=poly_data.get("name", "unnamed"),
                    points=poly_data["points"],
                    description=poly_data.get("description", "")
                )
                polygons.append(polygon)

            self.rois_by_camera[camera_id] = polygons
            logger.info(f"Loaded {len(polygons)} ROI polygons for camera {camera_id}")

    def add_roi(self, camera_id: str, polygon: ROIPolygon):
        """
        Add an ROI polygon for a camera
        Args:
            camera_id: Camera identifier
            polygon: ROI polygon
        """
        if camera_id not in self.rois_by_camera:
            self.rois_by_camera[camera_id] = []

        self.rois_by_camera[camera_id].append(polygon)
        self.roi_enabled[camera_id] = True
        logger.info(f"Added ROI '{polygon.name}' for camera {camera_id}")

    def remove_roi(self, camera_id: str, roi_name: str) -> bool:
        """
        Remove an ROI polygon
        Args:
            camera_id: Camera identifier
            roi_name: Name of ROI to remove
        Returns:
            True if removed successfully
        """
        if camera_id not in self.rois_by_camera:
            return False

        original_length = len(self.rois_by_camera[camera_id])
        self.rois_by_camera[camera_id] = [
            roi for roi in self.rois_by_camera[camera_id] if roi.name != roi_name
        ]

        removed = len(self.rois_by_camera[camera_id]) < original_length
        if removed:
            logger.info(f"Removed ROI '{roi_name}' from camera {camera_id}")
        return removed

    def clear_rois(self, camera_id: str):
        """
        Clear all ROIs for a camera
        Args:
            camera_id: Camera identifier
        """
        self.rois_by_camera[camera_id] = []
        self.roi_enabled[camera_id] = False
        logger.info(f"Cleared all ROIs for camera {camera_id}")

    def enable_roi(self, camera_id: str, enabled: bool = True):
        """
        Enable or disable ROI filtering for a camera
        Args:
            camera_id: Camera identifier
            enabled: True to enable, False to disable
        """
        self.roi_enabled[camera_id] = enabled
        logger.info(f"ROI filtering {'enabled' if enabled else 'disabled'} for camera {camera_id}")

    def filter_detections(
        self,
        camera_id: str,
        detections: List[Detection]
    ) -> List[Detection]:
        """
        Filter detections based on ROI polygons
        Args:
            camera_id: Camera identifier
            detections: List of detections to filter
        Returns:
            Filtered list of detections (only those inside ROI)
        """
        # If ROI not enabled or no ROIs defined, return all detections
        if not self.roi_enabled.get(camera_id, False):
            return detections

        if camera_id not in self.rois_by_camera or not self.rois_by_camera[camera_id]:
            return detections

        # Filter detections
        filtered_detections = []
        rois = self.rois_by_camera[camera_id]

        for detection in detections:
            # Check if detection center is inside any ROI polygon
            for roi in rois:
                if roi.contains_point(detection.center):
                    filtered_detections.append(detection)
                    break  # Detection is valid, no need to check other ROIs

        logger.debug(
            f"Camera {camera_id}: Filtered {len(detections)} -> {len(filtered_detections)} detections"
        )

        return filtered_detections

    def filter_detections_batch(
        self,
        detections_by_camera: Dict[str, List[Detection]]
    ) -> Dict[str, List[Detection]]:
        """
        Filter detections for multiple cameras
        Args:
            detections_by_camera: Dictionary mapping camera_id to detections
        Returns:
            Filtered detections by camera
        """
        filtered = {}
        for camera_id, detections in detections_by_camera.items():
            filtered[camera_id] = self.filter_detections(camera_id, detections)
        return filtered

    def get_rois(self, camera_id: str) -> List[ROIPolygon]:
        """
        Get ROI polygons for a camera
        Args:
            camera_id: Camera identifier
        Returns:
            List of ROI polygons
        """
        return self.rois_by_camera.get(camera_id, [])

    def is_enabled(self, camera_id: str) -> bool:
        """
        Check if ROI filtering is enabled for a camera
        Args:
            camera_id: Camera identifier
        Returns:
            True if enabled
        """
        return self.roi_enabled.get(camera_id, False)

    def draw_rois_on_frame(
        self,
        frame: np.ndarray,
        camera_id: str,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        fill_alpha: float = 0.2
    ) -> np.ndarray:
        """
        Draw ROI polygons on a frame for visualization
        Args:
            frame: Input frame
            camera_id: Camera identifier
            color: ROI border color (BGR)
            thickness: Border thickness
            fill_alpha: Fill transparency (0-1)
        Returns:
            Frame with ROIs drawn
        """
        if camera_id not in self.rois_by_camera:
            return frame

        output = frame.copy()
        overlay = frame.copy()

        for roi in self.rois_by_camera[camera_id]:
            # Draw filled polygon on overlay
            cv2.fillPoly(overlay, [roi.points], color)

            # Draw border
            cv2.polylines(output, [roi.points], True, color, thickness)

            # Add label
            if len(roi.points) > 0:
                label_pos = tuple(roi.points[0])
                cv2.putText(
                    output,
                    roi.name,
                    label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        # Blend overlay with output
        cv2.addWeighted(overlay, fill_alpha, output, 1 - fill_alpha, 0, output)

        return output

    def export_config(self) -> Dict:
        """
        Export ROI configuration to dictionary
        Returns:
            ROI configuration dictionary
        """
        config = {"rois": {}}

        for camera_id in set(list(self.rois_by_camera.keys()) + list(self.roi_enabled.keys())):
            config["rois"][camera_id] = {
                "enabled": self.roi_enabled.get(camera_id, False),
                "polygons": [
                    roi.to_dict() for roi in self.rois_by_camera.get(camera_id, [])
                ],
                "notes": f"ROI configuration for {camera_id}"
            }

        return config
