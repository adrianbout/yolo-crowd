"""
Thermal YOLO Detector
Specialized detector for thermal cameras using Ultralytics YOLO
"""

import cv2
import numpy as np
from typing import List, Dict, Optional
import logging
from pathlib import Path

from ultralytics import YOLO

from .detector import Detection

logger = logging.getLogger(__name__)


class ThermalYOLODetector:
    """
    Thermal YOLO detector using Ultralytics YOLO library
    Optimized for thermal camera imagery with specialized preprocessing
    """

    def __init__(
        self,
        model_path: str = "weights/yolo-thermal-approche2.pt",
        device: str = "cuda",
        confidence_threshold: float = 0.6,
        iou_threshold: float = 0.45,
        img_size: int = 640
    ):
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Thermal model file not found: {model_path}")

        logger.info(f"Loading Thermal YOLO model from {model_path}")
        logger.info(f"Settings: img_size={img_size}, conf={confidence_threshold}, iou={iou_threshold}")

        self.model = YOLO(model_path)

        # Set device
        if device == "cuda":
            import torch
            if torch.cuda.is_available():
                self.model.to("cuda")
            else:
                logger.warning("CUDA requested but not available, using CPU")
                self.device = "cpu"

        logger.info(f"Thermal YOLO model loaded successfully on {self.device}")

    def apply_preprocessing(self, frame: np.ndarray, preprocessing_config: Dict) -> np.ndarray:
        """
        Apply thermal-specific preprocessing
        Args:
            frame: Input frame
            preprocessing_config: Preprocessing settings
        Returns:
            Preprocessed frame
        """
        if preprocessing_config.get("equalize_histogram", False):
            if len(frame.shape) == 3:
                yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
                yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
                frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

        if preprocessing_config.get("clahe", False):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            if len(frame.shape) == 3:
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        if preprocessing_config.get("denoise", False):
            frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)

        return frame

    def detect_batch(
        self,
        frames: List[np.ndarray],
        camera_ids: List[str],
        inference_configs: List[Dict],
        preprocessing_configs: Optional[List[Dict]] = None
    ) -> Dict[str, List[Detection]]:
        """
        Perform batched detection on multiple frames
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

        detections_by_camera = {}

        # Process each frame (Ultralytics YOLO handles batching internally)
        for idx, (frame, camera_id, inference_config) in enumerate(zip(frames, camera_ids, inference_configs)):
            # Apply preprocessing if needed
            if preprocessing_configs and preprocessing_configs[idx].get("use_preprocessing", False):
                frame = self.apply_preprocessing(frame, preprocessing_configs[idx])

            # Get per-camera thresholds
            conf_thresh = inference_config.get("confidence_threshold", self.confidence_threshold)
            iou_thresh = inference_config.get("iou_threshold", self.iou_threshold)
            img_size = inference_config.get("img_size", self.img_size)

            # Run inference
            results = self.model(
                frame,
                conf=conf_thresh,
                iou=iou_thresh,
                imgsz=img_size,
                verbose=False,
                classes=[0]  # Person class only
            )

            # Parse results
            detections = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    # Get coordinates in xyxy format
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())

                    detection = Detection(
                        bbox=[float(x1), float(y1), float(x2), float(y2)],
                        confidence=conf,
                        class_id=cls
                    )
                    detections.append(detection)

            detections_by_camera[camera_id] = detections

        return detections_by_camera

    def detect_single(
        self,
        frame: np.ndarray,
        inference_config: Dict,
        preprocessing_config: Optional[Dict] = None
    ) -> List[Detection]:
        """
        Detect on a single frame
        Args:
            frame: Input frame (BGR)
            inference_config: Inference configuration
            preprocessing_config: Optional preprocessing configuration
        Returns:
            List of detections
        """
        result = self.detect_batch(
            frames=[frame],
            camera_ids=["single"],
            inference_configs=[inference_config],
            preprocessing_configs=[preprocessing_config] if preprocessing_config else None
        )
        return result.get("single", [])
