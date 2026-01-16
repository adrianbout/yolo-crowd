"""
Detection Service
Coordinates camera streams, YOLO detection, and ROI filtering
"""

import logging
import time
import threading
from typing import Dict, List, Optional
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from detection.detector import YOLODetector, DetectionAggregator
from detection.roi_filter import ROIFilter
from camera_control.camera_stream import CameraStreamManager
from services.state_manager import StateManager

logger = logging.getLogger(__name__)


class DetectionService:
    """
    Main detection service that coordinates:
    - Camera stream management
    - Batched YOLO inference
    - ROI filtering
    - State updates
    """

    def __init__(
        self,
        state_manager: StateManager,
        batch_size: int = 20,
        inference_interval: float = 0.5
    ):
        self.state_manager = state_manager
        self.batch_size = batch_size
        self.inference_interval = inference_interval

        # Components
        self.camera_manager: Optional[CameraStreamManager] = None
        self.detector: Optional[YOLODetector] = None
        self.roi_filter: ROIFilter = ROIFilter()
        self.aggregator: DetectionAggregator = DetectionAggregator()

        # Service state
        self.running = False
        self.detection_thread: Optional[threading.Thread] = None

        # Statistics
        self.total_inferences = 0
        self.start_time = None

    def initialize(self):
        """Initialize all components"""
        logger.info("Initializing detection service...")

        # Load configuration
        self.state_manager.load_configuration()

        # Initialize camera manager
        self.camera_manager = CameraStreamManager(buffer_size=1)
        self.camera_manager.load_from_config(self.state_manager.cameras)

        # Initialize YOLO detector
        detection_settings = self.state_manager.get_detection_settings()
        model_path = detection_settings.get("model_path", "weights/yolo-crowd.pt")

        self.detector = YOLODetector(
            model_path=model_path,
            device=detection_settings.get("device", "cuda"),
            half_precision=detection_settings.get("half_precision", True),
            class_filter=detection_settings.get("class_filter", None),
            img_size=detection_settings.get("img_size", 608),
            confidence_threshold=detection_settings.get("confidence_threshold", 0.25),
            iou_threshold=detection_settings.get("iou_threshold", 0.45)
        )

        # Initialize ROI filter
        self.roi_filter.load_rois(self.state_manager.get_all_rois())

        logger.info("Detection service initialized successfully")

    def start(self):
        """Start the detection service"""
        if self.running:
            logger.warning("Detection service is already running")
            return

        logger.info("Starting detection service...")

        # Start camera streams
        self.camera_manager.start_all()

        # Wait for cameras to connect
        logger.info("Waiting for cameras to connect...")
        time.sleep(3)

        # Start detection loop
        self.running = True
        self.start_time = time.time()
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()

        logger.info("Detection service started")

    def stop(self):
        """Stop the detection service"""
        logger.info("Stopping detection service...")

        self.running = False

        if self.detection_thread:
            self.detection_thread.join(timeout=5)

        if self.camera_manager:
            self.camera_manager.stop_all()

        logger.info("Detection service stopped")

    def _detection_loop(self):
        """Main detection loop"""
        logger.info("Detection loop started")

        while self.running:
            try:
                loop_start = time.time()

                # Get frames from all cameras
                frames_dict = self.camera_manager.get_all_frames()

                if not frames_dict:
                    logger.debug("No frames available")
                    time.sleep(self.inference_interval)
                    continue

                # Prepare batch
                camera_ids = list(frames_dict.keys())
                frames = [frames_dict[cam_id] for cam_id in camera_ids]

                # Get inference configs for each camera
                inference_configs = []
                preprocessing_configs = []

                for camera_id in camera_ids:
                    # Get camera config to check for per-camera overrides
                    camera_config = self.state_manager.get_camera_config(camera_id)
                    camera_detection_settings = camera_config.get("detection_settings") if camera_config else None

                    # Get profile as base settings
                    profile = self.state_manager.get_camera_profile(camera_id)

                    if camera_detection_settings:
                        # Use per-camera detection settings override
                        inference_config = {
                            "confidence_threshold": camera_detection_settings.get("confidence_threshold", 0.25),
                            "iou_threshold": camera_detection_settings.get("iou_threshold", 0.45),
                            "img_size": camera_detection_settings.get("img_size", 640)
                        }
                        # Build preprocessing config from per-camera settings
                        preprocessing_mode = camera_detection_settings.get("preprocessing", "none")
                        preprocessing_config = {
                            "use_preprocessing": preprocessing_mode != "none",
                            "clahe": preprocessing_mode == "clahe",
                            "equalize_histogram": preprocessing_mode == "equalize",
                            "denoise": preprocessing_mode == "denoise"
                        }
                        inference_configs.append(inference_config)
                        preprocessing_configs.append(preprocessing_config)
                    elif profile:
                        inference_configs.append(profile.get("inference_settings", {}))
                        preprocessing_configs.append(profile.get("preprocessing", {}))
                    else:
                        # Default settings
                        inference_configs.append({"confidence_threshold": 0.5, "iou_threshold": 0.45})
                        preprocessing_configs.append({})

                # Run batched detection
                detections_by_camera = self.detector.detect_batch(
                    frames=frames,
                    camera_ids=camera_ids,
                    inference_configs=inference_configs,
                    preprocessing_configs=preprocessing_configs
                )

                # Apply ROI filtering
                filtered_detections = self.roi_filter.filter_detections_batch(detections_by_camera)

                # Update counts
                counts = {cam_id: len(dets) for cam_id, dets in filtered_detections.items()}
                self.state_manager.update_counts_batch(counts)

                # Add to history (optional, limited)
                for cam_id, dets in filtered_detections.items():
                    detection_dicts = [d.to_dict() for d in dets]
                    self.state_manager.add_detection_to_history(cam_id, detection_dicts)

                # Statistics
                self.total_inferences += 1

                # Log with timing info
                total_count = sum(counts.values())
                elapsed = time.time() - loop_start
                logger.info(
                    f"Inference {self.total_inferences}: {len(frames)} cameras, "
                    f"{total_count} detections, {elapsed*1000:.1f}ms"
                )

                # Control inference rate
                elapsed = time.time() - loop_start
                sleep_time = max(0, self.inference_interval - elapsed)
                time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Error in detection loop: {e}", exc_info=True)
                time.sleep(1)

        logger.info("Detection loop ended")

    def get_statistics(self) -> Dict:
        """
        Get service statistics
        Returns:
            Statistics dictionary
        """
        uptime = time.time() - self.start_time if self.start_time else 0

        return {
            "running": self.running,
            "uptime_seconds": uptime,
            "total_inferences": self.total_inferences,
            "inferences_per_second": self.total_inferences / uptime if uptime > 0 else 0,
            "camera_status": self.camera_manager.get_all_status() if self.camera_manager else {}
        }

    def get_current_detections(self, camera_id: str) -> List[Dict]:
        """
        Get current detections for a camera (from history)
        Args:
            camera_id: Camera identifier
        Returns:
            List of recent detections
        """
        history = self.state_manager.get_detection_history(camera_id, limit=1)
        if history:
            return history[-1].get("detections", [])
        return []

    def update_roi(self, camera_id: str, roi_config: Dict):
        """
        Update ROI configuration
        Args:
            camera_id: Camera identifier
            roi_config: ROI configuration
        """
        # Update in state manager
        self.state_manager.update_roi(camera_id, roi_config)

        # Reload ROI filter
        self.roi_filter.load_rois(self.state_manager.get_all_rois())

        # Save to file
        self.state_manager.save_rois()

        logger.info(f"Updated ROI for camera {camera_id}")

    def get_frame_with_detections(self, camera_id: str, draw_rois: bool = True) -> Optional[bytes]:
        """
        Get frame with detections and ROIs drawn
        Args:
            camera_id: Camera identifier
            draw_rois: Whether to draw ROIs
        Returns:
            JPEG encoded frame or None
        """
        import cv2

        frame = self.camera_manager.get_frame(camera_id)
        if frame is None:
            return None

        # Draw ROIs
        if draw_rois:
            frame = self.roi_filter.draw_rois_on_frame(frame, camera_id)

        # Draw detections
        detections = self.get_current_detections(camera_id)
        for det in detections:
            bbox = det["bbox"]
            x1, y1, x2, y2 = map(int, bbox)
            conf = det["confidence"]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw confidence
            label = f"{conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes()
