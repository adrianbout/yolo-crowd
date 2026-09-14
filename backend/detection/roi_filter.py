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


class ROIGate:
    """
    A two-point line across a flow zone, with each side named.

    Sides are named rather than numbered so a crossing reads as
    "lobby -> offices" on the dashboard without a legend. The names are
    stored on every crossing row, so renaming a side later does not rewrite
    what history says.
    """

    def __init__(
        self,
        points: List[List[int]],
        side_a: str = "A",
        side_b: str = "B",
        name: str = "gate"
    ):
        if len(points) != 2:
            raise ValueError(f"Gate '{name}' needs exactly 2 points, got {len(points)}")
        self.name = name
        self.points = np.array(points, dtype=np.int32)
        self.side_a = side_a
        self.side_b = side_b

    def side_of(self, point: Tuple[float, float]) -> int:
        """
        Which side of the gate a point falls on.

        Returns +1 for side B, -1 for side A, 0 when exactly on the line.
        A crossing is a change of sign between two observations of one track.
        """
        (x1, y1), (x2, y2) = self.points
        px, py = point
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if cross > 0:
            return 1
        if cross < 0:
            return -1
        return 0

    def direction_label(self, from_side: int, to_side: int) -> Optional[Dict[str, str]]:
        """
        Name a transition, or None when it is not a crossing.

        Sign 0 (exactly on the line) is deliberately not a crossing - treating
        it as one would double-count anyone who pauses on the threshold.
        """
        if from_side == 0 or to_side == 0 or from_side == to_side:
            return None
        if from_side < 0:
            return {"direction": "a_to_b", "side_from": self.side_a, "side_to": self.side_b}
        return {"direction": "b_to_a", "side_from": self.side_b, "side_to": self.side_a}

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "points": self.points.tolist(),
            "side_a": self.side_a,
            "side_b": self.side_b
        }


class ROIPolygon:
    """
    A Region of Interest polygon.

    Carries the fields each role needs: `capacity` for seating zones,
    `gate` for flow zones. Both are optional, so a zone drawn before roles
    existed still loads and behaves as it did.
    """

    def __init__(
        self,
        name: str,
        points: List[List[int]],
        description: str = "",
        capacity: Optional[int] = None,
        gate: Optional[ROIGate] = None
    ):
        self.name = name
        self.points = np.array(points, dtype=np.int32)
        self.description = description
        self.capacity = capacity
        self.gate = gate

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
        d = {
            "name": self.name,
            "points": self.points.tolist(),
            "description": self.description
        }
        # Omitted when unset, so a seating zone carries no empty gate and vice versa
        if self.capacity is not None:
            d["capacity"] = self.capacity
        if self.gate is not None:
            d["gate"] = self.gate.to_dict()
        return d


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
                gate = None
                gate_data = poly_data.get("gate")
                if gate_data:
                    try:
                        gate = ROIGate(
                            points=gate_data["points"],
                            side_a=gate_data.get("side_a", "A"),
                            side_b=gate_data.get("side_b", "B"),
                            name=gate_data.get("name", "gate")
                        )
                    except (KeyError, ValueError) as e:
                        # A malformed gate disables crossing counts for this zone
                        # rather than taking the whole camera down.
                        logger.error(
                            f"Camera {camera_id}, zone "
                            f"'{poly_data.get('name', 'unnamed')}': bad gate ({e})"
                        )

                polygon = ROIPolygon(
                    name=poly_data.get("name", "unnamed"),
                    points=poly_data["points"],
                    description=poly_data.get("description", ""),
                    capacity=poly_data.get("capacity"),
                    gate=gate
                )
                polygons.append(polygon)

            self.rois_by_camera[camera_id] = polygons
            logger.info(f"Loaded {len(polygons)} ROI polygons for camera {camera_id}")

    def get_zones(self, camera_id: str) -> List[ROIPolygon]:
        """Zones defined for a camera, empty when ROI filtering is off."""
        if not self.roi_enabled.get(camera_id, False):
            return []
        return self.rois_by_camera.get(camera_id, [])

    def get_capacity(
        self,
        camera_id: str,
        zone: ROIPolygon,
        camera_config: Optional[Dict] = None
    ) -> int:
        """
        Seats in one zone.

        A zone's own capacity wins. Otherwise the camera's `totalChairs` is
        used, but only when the camera has a single zone - with several zones
        that number describes the camera as a whole, and handing the full
        figure to each zone would report a 5-chair camera as having 10 seats.
        Multi-zone cameras must declare capacity per zone.
        """
        if zone.capacity is not None:
            return int(zone.capacity)

        if camera_config and len(self.rois_by_camera.get(camera_id, [])) == 1:
            return int(camera_config.get("totalChairs", 0) or 0)

        if camera_config and camera_config.get("totalChairs"):
            logger.warning(
                f"Camera {camera_id} zone '{zone.name}' has no capacity and the "
                f"camera has multiple zones; set capacity per zone. Reporting 0."
            )
        return 0

    def get_camera_capacity(self, camera_id: str, camera_config: Optional[Dict] = None) -> int:
        """
        Total seats visible to a camera.

        The sum of whatever its zones declare, falling back to the camera's
        `totalChairs` when no zone declares any - which is how every camera
        configured before zone capacity existed still reports correctly.
        """
        zones = self.get_zones(camera_id)
        declared = [z.capacity for z in zones if z.capacity is not None]
        if declared:
            return int(sum(declared))
        if camera_config:
            return int(camera_config.get("totalChairs", 0) or 0)
        return 0

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
