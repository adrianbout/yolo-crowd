"""
YOLO Detection Service
Handles batched inference for multiple camera streams
"""

import sys
import torch
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path

# Add project root to path for custom YOLOv5 models
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import custom YOLOv5 modules
try:
    from models.experimental import attempt_load
    from utils.general import non_max_suppression, scale_coords
    CUSTOM_YOLO_AVAILABLE = True
except ImportError:
    CUSTOM_YOLO_AVAILABLE = False
    logger.warning("Custom YOLOv5 modules not found, will try torch.hub")

logger = logging.getLogger(__name__)


class Detection:
    """Single detection result"""

    def __init__(self, bbox: List[float], confidence: float, class_id: int):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.class_id = class_id
        self.center = self._calculate_center()

    def _calculate_center(self) -> Tuple[float, float]:
        """Calculate center point of bounding box"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "bbox": self.bbox,
            "confidence": float(self.confidence),
            "class_id": int(self.class_id),
            "center": self.center
        }


class YOLODetector:
    """
    YOLO detector with batched inference support
    Optimized for processing multiple camera frames simultaneously
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        half_precision: bool = True,
        class_filter: Optional[List[int]] = None
    ):
        # Auto-detect device if CUDA requested but not available
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, using CPU")
            device = "cpu"

        self.device = device
        self.half_precision = half_precision and device == "cuda"
        self.class_filter = class_filter

        logger.info(f"Loading YOLO model from {model_path}")
        self.model = self._load_model(model_path)
        logger.info(f"Model loaded successfully on {self.device}")

        # Warm up the model
        self._warmup()

    def _load_model(self, model_path: str):
        """Load YOLO model"""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Try custom YOLOv5 first (if available)
        if CUSTOM_YOLO_AVAILABLE:
            try:
                logger.info("Loading model with custom YOLOv5 (models/experimental)")
                model = attempt_load(model_path, map_location=self.device)

                if self.half_precision:
                    model.half()

                model.eval()
                logger.info("Model loaded successfully with custom YOLOv5")
                return model

            except Exception as e:
                logger.error(f"Failed to load with custom YOLOv5: {e}")
                logger.info("Falling back to torch.hub...")

        # Fallback to torch.hub
        try:
            logger.info("Loading model via torch.hub")
            model = torch.hub.load(
                'ultralytics/yolov5',
                'custom',
                path=model_path,
                trust_repo=True,
                force_reload=False,
                _verbose=False
            )
            model.to(self.device)

            if self.half_precision:
                model.half()

            model.eval()
            logger.info("Model loaded successfully via torch.hub")
            return model

        except Exception as e:
            logger.error(f"Failed to load model via torch.hub: {e}")
            logger.info("Attempting direct model loading...")

            try:
                # Try direct loading with weights_only=False
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

                # If checkpoint is a dict with 'model' key
                if isinstance(checkpoint, dict) and 'model' in checkpoint:
                    model = checkpoint['model']
                else:
                    model = checkpoint

                if hasattr(model, 'to'):
                    model = model.to(self.device)

                if self.half_precision and hasattr(model, 'half'):
                    model.half()

                if hasattr(model, 'eval'):
                    model.eval()

                logger.info("Model loaded successfully via direct loading")
                return model

            except Exception as e2:
                logger.error(f"Failed to load model directly: {e2}")
                raise RuntimeError(
                    f"Could not load model from {model_path}. "
                    f"Custom YOLOv5: {e if CUSTOM_YOLO_AVAILABLE else 'Not available'}. "
                    f"Torch hub error: {e}. Direct load error: {e2}"
                )

    def _warmup(self):
        """Warm up the model with dummy inference"""
        logger.info("Warming up model...")
        dummy_img = torch.zeros((1, 3, 640, 640)).to(self.device)
        if self.half_precision:
            dummy_img = dummy_img.half()

        with torch.no_grad():
            _ = self.model(dummy_img)

        logger.info("Model warmup complete")

    def preprocess_frame(self, frame: np.ndarray, target_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
        """
        Preprocess frame for inference
        Args:
            frame: Input frame (BGR)
            target_size: Target size (width, height)
        Returns:
            Preprocessed frame
        """
        # Resize
        frame = cv2.resize(frame, target_size)

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return frame

    def apply_preprocessing(self, frame: np.ndarray, preprocessing_config: Dict) -> np.ndarray:
        """
        Apply additional preprocessing based on camera profile
        Args:
            frame: Input frame
            preprocessing_config: Preprocessing settings from profile
        Returns:
            Preprocessed frame
        """
        if preprocessing_config.get("equalize_histogram", False):
            # Convert to grayscale, equalize, convert back
            if len(frame.shape) == 3:
                yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
                yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
                frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

        if preprocessing_config.get("clahe", False):
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
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

        # Apply preprocessing if needed
        if preprocessing_configs:
            processed_frames = []
            for frame, preproc_cfg in zip(frames, preprocessing_configs):
                if preproc_cfg.get("use_preprocessing", False):
                    frame = self.apply_preprocessing(frame, preproc_cfg)
                processed_frames.append(self.preprocess_frame(frame))
        else:
            processed_frames = [self.preprocess_frame(f) for f in frames]

        # Convert to tensor batch
        # processed_frames is list of numpy arrays [H, W, C]
        # Convert to tensor [B, C, H, W]
        img_tensors = []
        for frame in processed_frames:
            # Convert to tensor and normalize
            img = torch.from_numpy(frame).to(self.device)
            img = img.permute(2, 0, 1)  # HWC to CHW
            img = img.float() / 255.0    # 0-255 to 0.0-1.0

            if self.half_precision:
                img = img.half()

            img_tensors.append(img)

        # Stack into batch
        img_batch = torch.stack(img_tensors, 0)

        # Run inference
        with torch.no_grad():
            # Custom YOLOv5 returns a tuple, torch.hub returns object with .xyxy
            results = self.model(img_batch)

            # Handle different model output formats
            if isinstance(results, tuple):
                # Custom YOLOv5: returns (predictions, ?)
                pred = results[0]
            else:
                # Torch.hub YOLOv5: has .xyxy attribute
                pred = results.xyxy if hasattr(results, 'xyxy') else results

        # Apply NMS if using custom YOLOv5
        if isinstance(results, tuple):
            # Import NMS from custom YOLOv5
            if CUSTOM_YOLO_AVAILABLE:
                pred = non_max_suppression(
                    pred,
                    conf_thres=0.001,  # Will filter later per camera
                    iou_thres=0.45
                )

        # Parse results
        detections_by_camera = {}

        for idx, (camera_id, inference_config) in enumerate(zip(camera_ids, inference_configs)):
            conf_thresh = inference_config.get("confidence_threshold", 0.5)
            iou_thresh = inference_config.get("iou_threshold", 0.45)

            # Get detections for this frame
            detections = []

            # Handle different result formats
            if isinstance(pred, list):
                # Custom YOLOv5 with NMS: list of tensors per image
                frame_pred = pred[idx].cpu().numpy()  # [x1, y1, x2, y2, conf, class]
            else:
                # Torch.hub format
                frame_pred = pred[idx].cpu().numpy()

            for det in frame_pred:
                x1, y1, x2, y2, conf, cls = det

                # Filter by confidence
                if conf < conf_thresh:
                    continue

                # Filter by class if specified
                if self.class_filter and int(cls) not in self.class_filter:
                    continue

                detection = Detection(
                    bbox=[float(x1), float(y1), float(x2), float(y2)],
                    confidence=float(conf),
                    class_id=int(cls)
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


class DetectionAggregator:
    """
    Handles de-duplication and aggregation of detections across cameras
    """

    def __init__(self):
        pass

    def deduplicate_detections(
        self,
        detections_by_camera: Dict[str, List[Detection]],
        iou_threshold: float = 0.5
    ) -> Dict[str, List[Detection]]:
        """
        Remove duplicate detections across cameras
        Note: This is a simple approach. For more accurate deduplication,
        camera calibration and spatial mapping would be needed.

        Args:
            detections_by_camera: Detections for each camera
            iou_threshold: IOU threshold for considering detections as duplicates
        Returns:
            Deduplicated detections
        """
        # For now, this is a placeholder
        # In practice, you'd need camera calibration to map detections to world coordinates
        # For ROI-based approach, this isn't needed as ROIs prevent overlap
        return detections_by_camera

    def calculate_total_count(self, detections_by_camera: Dict[str, List[Detection]]) -> int:
        """
        Calculate total count across all cameras
        Args:
            detections_by_camera: Detections for each camera
        Returns:
            Total count
        """
        total = sum(len(dets) for dets in detections_by_camera.values())
        return total


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union between two boxes
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    Returns:
        IOU value
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Calculate intersection area
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
        return 0.0

    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)

    # Calculate union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0
