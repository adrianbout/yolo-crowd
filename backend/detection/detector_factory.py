"""
Detector Factory
Routes cameras to appropriate detectors based on camera type and configuration
"""

import logging
from typing import Dict, List, Optional, Any
import numpy as np

from .detector import YOLODetector, Detection
from .thermal_detector import ThermalYOLODetector
from .blob_hotspot_detector import BlobHotspotDetector

logger = logging.getLogger(__name__)


class DetectorFactory:
    """
    Factory class that manages multiple detectors and routes
    detection requests based on per-camera detection_model setting
    """

    def __init__(
        self,
        rgb_model_path: str = "weights/yolo-crowd.pt",
        thermal_model_path: str = "weights/yolo-thermal-approche2.pt",
        device: str = "cuda",
        half_precision: bool = True,
        default_confidence: float = 0.25,
        default_iou: float = 0.45,
        default_img_size: int = 640
    ):
        self.device = device
        self.half_precision = half_precision
        self.default_confidence = default_confidence
        self.default_iou = default_iou
        self.default_img_size = default_img_size

        # Initialize detectors
        self.rgb_detector: Optional[YOLODetector] = None
        self.thermal_detector: Optional[ThermalYOLODetector] = None
        self.blob_hotspot_detector: Optional[BlobHotspotDetector] = None

        # Track per-camera model selection (set via detection_settings.detection_model)
        self.camera_models: Dict[str, str] = {}

        # Initialize RGB detector (YOLOv5 crowd model)
        logger.info("Initializing RGB detector...")
        self.rgb_detector = YOLODetector(
            model_path=rgb_model_path,
            device=device,
            half_precision=half_precision,
            class_filter=[0],  # Person class
            img_size=default_img_size,
            confidence_threshold=default_confidence,
            iou_threshold=default_iou
        )

        # Initialize Thermal detector (Ultralytics YOLO thermal model)
        logger.info("Initializing Thermal detector...")
        try:
            self.thermal_detector = ThermalYOLODetector(
                model_path=thermal_model_path,
                device=device,
                confidence_threshold=default_confidence,
                iou_threshold=default_iou,
                img_size=default_img_size
            )
        except FileNotFoundError as e:
            logger.warning(f"Thermal model not found: {e}. Thermal option will fall back to RGB detector.")
            self.thermal_detector = None

        # Initialize Blob Hotspot detector (no model file needed)
        logger.info("Initializing Blob Hotspot detector...")
        self.blob_hotspot_detector = BlobHotspotDetector()

        logger.info("DetectorFactory initialized successfully")

    def register_camera_model(self, camera_id: str, detection_model: str):
        """
        Register which detection model a camera should use
        Args:
            camera_id: Camera identifier
            detection_model: Model to use ("rgb", "thermal", or "blob_hotspot")
        """
        self.camera_models[camera_id] = detection_model.lower()
        logger.info(f"Camera {camera_id} configured to use {detection_model} model")

    def get_detector_for_camera(self, camera_id: str):
        """
        Get the appropriate detector for a camera based on its detection_model setting
        Args:
            camera_id: Camera identifier
        Returns:
            Detector instance
        """
        model = self.camera_models.get(camera_id, "rgb")

        if model == "thermal" and self.thermal_detector:
            return self.thermal_detector
        elif model == "blob_hotspot" and self.blob_hotspot_detector:
            return self.blob_hotspot_detector
        return self.rgb_detector

    def detect_batch(
        self,
        frames: List[np.ndarray],
        camera_ids: List[str],
        inference_configs: List[Dict],
        preprocessing_configs: Optional[List[Dict]] = None
    ) -> Dict[str, List[Detection]]:
        """
        Perform batched detection, routing to appropriate detectors by camera type
        Args:
            frames: List of frames (BGR format)
            camera_ids: List of camera IDs corresponding to frames
            inference_configs: List of inference configs for each camera
            preprocessing_configs: Optional list of preprocessing configs
        Returns:
            Dictionary mapping camera_id to list of detections
        """
        if not frames:
            return {}

        # Group frames by detector model setting
        rgb_indices = []
        thermal_indices = []
        blob_indices = []

        for idx, camera_id in enumerate(camera_ids):
            model = self.camera_models.get(camera_id, "rgb")
            if model == "thermal" and self.thermal_detector:
                thermal_indices.append(idx)
            elif model == "blob_hotspot" and self.blob_hotspot_detector:
                blob_indices.append(idx)
            else:
                rgb_indices.append(idx)

        all_detections = {}

        # Process RGB frames
        if rgb_indices:
            rgb_frames = [frames[i] for i in rgb_indices]
            rgb_camera_ids = [camera_ids[i] for i in rgb_indices]
            rgb_inference_configs = [inference_configs[i] for i in rgb_indices]
            rgb_preprocessing_configs = [preprocessing_configs[i] for i in rgb_indices] if preprocessing_configs else None

            rgb_detections = self.rgb_detector.detect_batch(
                frames=rgb_frames,
                camera_ids=rgb_camera_ids,
                inference_configs=rgb_inference_configs,
                preprocessing_configs=rgb_preprocessing_configs
            )
            all_detections.update(rgb_detections)

        # Process Thermal frames
        if thermal_indices and self.thermal_detector:
            thermal_frames = [frames[i] for i in thermal_indices]
            thermal_camera_ids = [camera_ids[i] for i in thermal_indices]
            thermal_inference_configs = [inference_configs[i] for i in thermal_indices]
            thermal_preprocessing_configs = [preprocessing_configs[i] for i in thermal_indices] if preprocessing_configs else None

            thermal_detections = self.thermal_detector.detect_batch(
                frames=thermal_frames,
                camera_ids=thermal_camera_ids,
                inference_configs=thermal_inference_configs,
                preprocessing_configs=thermal_preprocessing_configs
            )
            all_detections.update(thermal_detections)

        # Process Blob Hotspot frames
        if blob_indices and self.blob_hotspot_detector:
            blob_frames = [frames[i] for i in blob_indices]
            blob_camera_ids = [camera_ids[i] for i in blob_indices]
            blob_inference_configs = [inference_configs[i] for i in blob_indices]
            blob_preprocessing_configs = [preprocessing_configs[i] for i in blob_indices] if preprocessing_configs else None

            blob_detections = self.blob_hotspot_detector.detect_batch(
                frames=blob_frames,
                camera_ids=blob_camera_ids,
                inference_configs=blob_inference_configs,
                preprocessing_configs=blob_preprocessing_configs
            )
            all_detections.update(blob_detections)

        return all_detections

    def detect_single(
        self,
        frame: np.ndarray,
        camera_id: str,
        inference_config: Dict,
        preprocessing_config: Optional[Dict] = None
    ) -> List[Detection]:
        """
        Detect on a single frame using the appropriate detector
        Args:
            frame: Input frame (BGR)
            camera_id: Camera identifier for routing
            inference_config: Inference configuration
            preprocessing_config: Optional preprocessing configuration
        Returns:
            List of detections
        """
        detector = self.get_detector_for_camera(camera_id)
        return detector.detect_single(frame, inference_config, preprocessing_config)
