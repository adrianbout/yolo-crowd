"""
Blob Hotspot Detector
Detects people in thermal images using blob/hotspot analysis
No neural network required - uses thresholding and contour detection
"""

import cv2
import numpy as np
from typing import List, Dict, Optional
import logging

from .detector import Detection

logger = logging.getLogger(__name__)


class BlobHotspotDetector:
    """
    Blob-based hotspot detector for thermal cameras.
    Uses thresholding and contour analysis to detect heat signatures.

    Configurable settings:
    - threshold: Brightness threshold for hot regions (0-255)
    - min_area: Minimum blob area to consider as person
    - max_area: Maximum blob area (filter out large merged blobs)
    - aspect_ratio_min: Minimum height/width ratio for person shape
    - aspect_ratio_max: Maximum height/width ratio for person shape
    - kernel_size: Morphological kernel size for noise reduction
    - dilate_iterations: Number of dilation iterations to merge body parts
    """

    def __init__(
        self,
        threshold: int = 200,
        min_area: int = 2000,
        max_area: int = 50000,
        aspect_ratio_min: float = 0.5,
        aspect_ratio_max: float = 3.0,
        kernel_size: int = 15,
        dilate_iterations: int = 2
    ):
        self.threshold = threshold
        self.min_area = min_area
        self.max_area = max_area
        self.aspect_ratio_min = aspect_ratio_min
        self.aspect_ratio_max = aspect_ratio_max
        self.kernel_size = kernel_size
        self.dilate_iterations = dilate_iterations

        logger.info(f"BlobHotspotDetector initialized with threshold={threshold}, "
                   f"min_area={min_area}, max_area={max_area}, "
                   f"aspect_ratio=[{aspect_ratio_min}, {aspect_ratio_max}]")

    def detect_frame(
        self,
        frame: np.ndarray,
        config: Optional[Dict] = None
    ) -> List[Detection]:
        """
        Detect hotspots in a single frame.

        Args:
            frame: Input frame (BGR format)
            config: Optional per-frame config override

        Returns:
            List of Detection objects
        """
        # Use config overrides if provided
        threshold = config.get("blob_threshold", self.threshold) if config else self.threshold
        min_area = config.get("blob_min_area", self.min_area) if config else self.min_area
        max_area = config.get("blob_max_area", self.max_area) if config else self.max_area
        aspect_min = config.get("blob_aspect_ratio_min", self.aspect_ratio_min) if config else self.aspect_ratio_min
        aspect_max = config.get("blob_aspect_ratio_max", self.aspect_ratio_max) if config else self.aspect_ratio_max
        kernel_size = config.get("blob_kernel_size", self.kernel_size) if config else self.kernel_size
        dilate_iter = config.get("blob_dilate_iterations", self.dilate_iterations) if config else self.dilate_iterations

        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Threshold to detect hot regions
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        # Morphological operations to merge body parts and reduce noise
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.dilate(binary, kernel, iterations=dilate_iter)

        # Find contours (blobs)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by area
            if area < min_area or area > max_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = h / w if w > 0 else 0

            # Filter by aspect ratio (person-like shape)
            if aspect_ratio < aspect_min or aspect_ratio > aspect_max:
                continue

            # Calculate confidence based on area and aspect ratio
            # Larger blobs within range get higher confidence
            area_confidence = min(area / (max_area * 0.5), 1.0)
            aspect_confidence = 1.0 - abs(aspect_ratio - 1.5) / 1.5  # Optimal around 1.5
            confidence = (area_confidence + aspect_confidence) / 2

            detection = Detection(
                bbox=[float(x), float(y), float(x + w), float(y + h)],
                confidence=float(confidence),
                class_id=0  # Person class
            )
            detections.append(detection)

        return detections

    def detect_batch(
        self,
        frames: List[np.ndarray],
        camera_ids: List[str],
        inference_configs: List[Dict],
        preprocessing_configs: Optional[List[Dict]] = None
    ) -> Dict[str, List[Detection]]:
        """
        Perform detection on multiple frames.

        Args:
            frames: List of frames (BGR format)
            camera_ids: List of camera IDs
            inference_configs: List of inference configs for each camera
            preprocessing_configs: Optional preprocessing configs (not used for blob detection)

        Returns:
            Dictionary mapping camera_id to list of detections
        """
        if not frames:
            return {}

        detections_by_camera = {}

        for frame, camera_id, config in zip(frames, camera_ids, inference_configs):
            detections = self.detect_frame(frame, config)
            detections_by_camera[camera_id] = detections

        return detections_by_camera

    def detect_single(
        self,
        frame: np.ndarray,
        inference_config: Dict,
        preprocessing_config: Optional[Dict] = None
    ) -> List[Detection]:
        """
        Detect on a single frame.

        Args:
            frame: Input frame (BGR)
            inference_config: Inference configuration
            preprocessing_config: Optional preprocessing (not used)

        Returns:
            List of detections
        """
        return self.detect_frame(frame, inference_config)
