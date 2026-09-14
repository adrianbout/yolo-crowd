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
from detection.detector_factory import DetectorFactory
from detection.roi_filter import ROIFilter
from detection.heatmap import HeatmapManager
from services.floorplan import FloorPlan
from services.node_identity import NodeIdentity
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
        self.frame_skip = 0  # Number of frames to skip between processing

        # Identity, config versioning and per-camera tracker sessions
        self.node = NodeIdentity(config_dir=str(state_manager.config_dir))

        # Components
        self.camera_manager: Optional[CameraStreamManager] = None
        self.detector: Optional[YOLODetector] = None
        self.detector_factory: Optional[DetectorFactory] = None
        self.roi_filter: ROIFilter = ROIFilter()
        self.aggregator: DetectionAggregator = DetectionAggregator()
        # Flow cameras only; seating zones have a capacity, not a traffic pattern
        self.heatmaps: HeatmapManager = HeatmapManager()
        # Ties each camera's view to a shared plan, so flow can be read across
        # the building rather than one rectangle at a time.
        self.floorplan: FloorPlan = FloorPlan(config_dir=str(state_manager.config_dir))

        # Service state
        self.running = False
        self.detection_thread: Optional[threading.Thread] = None
        self._frame_skip_counter = 0  # Internal counter for frame skipping

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

        # Initialize detector factory with both RGB and thermal models
        detection_settings = self.state_manager.get_detection_settings()
        model_path = detection_settings.get("model_path", "weights/yolo-crowd.pt")
        thermal_model_path = detection_settings.get("thermal_model_path", "weights/yolo-thermal-approche2.pt")
        pose_model_path = detection_settings.get("pose_model_path", "weights/yolo11m-pose.pt")

        self.detector_factory = DetectorFactory(
            rgb_model_path=model_path,
            thermal_model_path=thermal_model_path,
            pose_model_path=pose_model_path,
            device=detection_settings.get("device", "cuda"),
            half_precision=detection_settings.get("half_precision", True),
            default_confidence=detection_settings.get("confidence_threshold", 0.25),
            default_iou=detection_settings.get("iou_threshold", 0.45),
            default_img_size=detection_settings.get("img_size", 608)
        )

        # Register per-camera detection model with the factory
        for camera in self.state_manager.cameras.get("cameras", []):
            camera_detection_settings = camera.get("detection_settings") or {}
            detection_model = camera_detection_settings.get("detection_model", "rgb")
            self.detector_factory.register_camera_model(camera["id"], detection_model)

        # Keep reference to RGB detector for backward compatibility
        self.detector = self.detector_factory.rgb_detector

        # Initialize ROI filter
        self.roi_filter.load_rois(self.state_manager.get_all_rois())

        # Load frame_skip setting
        self.frame_skip = detection_settings.get("frame_skip", 0)

        logger.info(f"Detection service initialized successfully (frame_skip={self.frame_skip})")

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

                # Frame skip logic - skip processing if counter hasn't reached threshold
                if self.frame_skip > 0:
                    self._frame_skip_counter += 1
                    if self._frame_skip_counter <= self.frame_skip:
                        time.sleep(0.01)  # Small sleep to avoid busy-waiting
                        continue
                    self._frame_skip_counter = 0  # Reset counter after processing

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

                # Run batched detection via detector factory (routes to appropriate detector)
                detections_by_camera = self.detector_factory.detect_batch(
                    frames=frames,
                    camera_ids=camera_ids,
                    inference_configs=inference_configs,
                    preprocessing_configs=preprocessing_configs
                )

                # Apply ROI filtering
                filtered_detections = self.roi_filter.filter_detections_batch(detections_by_camera)

                # Accumulate flow-zone heatmaps from the same filtered detections
                # the counts come from, so the two can never disagree.
                for cam_id, dets in filtered_detections.items():
                    if self.get_camera_role(cam_id) != "flow":
                        continue
                    frame = frames_dict.get(cam_id)
                    if frame is not None:
                        self.heatmaps.update(cam_id, dets, frame.shape)

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
                    f"{total_count} empty chairs, {elapsed*1000:.1f}ms"
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

    def switch_camera_model(self, camera_id: str, detection_model: str):
        """
        Point a camera at a different detector while running.

        Every detector is already resident in the factory, so this is a
        routing change rather than a model load - no restart, and detection
        continues on the next batch.

        Switching discards the camera's tracker state, so the session is
        restarted: track IDs either side of a switch belong to different runs
        and must never be joined. The config version moves too, marking the
        boundary in stored rows.
        """
        if not self.detector_factory:
            logger.warning(f"Cannot switch model for {camera_id}: factory not initialized")
            return

        self.detector_factory.register_camera_model(camera_id, detection_model)
        self.node.new_session_id(camera_id, reason=f"model switched to {detection_model}")
        self.node.bump_config_version(reason=f"{camera_id} model -> {detection_model}")

    def get_camera_role(self, camera_id: str) -> str:
        """Role for a camera - its own if set, otherwise the node default."""
        return self.node.resolve_role(self.state_manager.get_camera_config(camera_id))

    def get_heatmap(self, camera_id: str) -> Optional[Dict]:
        """
        A flow camera's accumulated heatmap, or None.

        Returns None for seating cameras rather than an empty grid, so the
        dashboard can tell "no traffic yet" apart from "not a flow zone".
        """
        if self.get_camera_role(camera_id) != "flow":
            return None
        return self.heatmaps.get(camera_id)

    def get_floorplan_heatmap(self) -> Optional[Dict]:
        """
        Every calibrated camera's traffic, projected onto the one plan.

        Returns None until a plan exists and at least one camera is
        calibrated against it - an empty plan and an uncalibrated one are
        worth telling apart in the UI.
        """
        if not self.floorplan.has_image:
            return None
        combined = self.floorplan.combined_grid(self.heatmaps.raw_grids())
        if combined is None:
            return None
        # Downsampled on the way out: a full plan-resolution float grid is
        # megabytes of JSON, and the client draws it scaled anyway.
        h, w = combined.shape
        step = max(1, int(max(w, h) / 240))
        small = combined[::step, ::step]
        return {
            "width": int(small.shape[1]),
            "height": int(small.shape[0]),
            "plan_width": self.floorplan.width,
            "plan_height": self.floorplan.height,
            "cameras": sorted(
                cid for cid in self.heatmaps.raw_grids()
                if self.floorplan.is_calibrated(cid)
            ),
            "grid": [[round(float(v), 4) for v in row] for row in small],
        }

    def get_floorplan_detections(self) -> Dict[str, List[Dict]]:
        """Current detections from every calibrated camera, in plan coordinates."""
        out = {}
        for camera_id in list(self.floorplan.calibrations.keys()):
            if not self.floorplan.is_calibrated(camera_id):
                continue
            dets = self.get_current_detections(camera_id)
            rows = self.floorplan.project_detections(camera_id, dets)
            if rows:
                out[camera_id] = rows
        return out

    def reset_heatmap(self, camera_id: Optional[str] = None):
        """Clear accumulated traffic - after moving a camera, or re-drawing a gate."""
        self.heatmaps.reset(camera_id)

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

        # Geometry and capacity changes make rows before and after
        # incomparable, so the version moves with them.
        self.node.bump_config_version(reason=f"{camera_id} zones edited")

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

        # Heatmap sits under the ROI outlines and boxes so those stay readable
        camera_config = self.state_manager.get_camera_config(camera_id)
        show_heatmap = camera_config.get("show_heatmap", True) if camera_config else True
        if show_heatmap and self.get_camera_role(camera_id) == "flow":
            frame = self.heatmaps.render_overlay(frame, camera_id)

        # Draw ROIs
        if draw_rois:
            frame = self.roi_filter.draw_rois_on_frame(frame, camera_id)

        # Check if show_boxes is enabled for this camera (default True)
        show_boxes = camera_config.get("show_boxes", True) if camera_config else True

        # Draw detections only if show_boxes is enabled
        if show_boxes:
            detections = self.get_current_detections(camera_id)
            for det in detections:
                bbox = det["bbox"]
                x1, y1, x2, y2 = map(int, bbox)
                conf = det["confidence"]

                # Posture tag is only present for the pose detector; non-pose
                # cameras render exactly as before (green box, conf label).
                posture = det.get("posture")
                if posture == "sitting":
                    color = (0, 255, 0)      # green
                    label = f"sitting {conf:.2f}"
                elif posture == "standing":
                    color = (0, 165, 255)    # orange (BGR)
                    label = f"standing {conf:.2f}"
                elif posture is not None:
                    color = (0, 255, 255)    # yellow for unknown posture
                    label = f"? {conf:.2f}"
                else:
                    color = (0, 255, 0)
                    label = f"{conf:.2f}"

                # Pose detector only: show the stable per-person track ID
                track_id = det.get("track_id")
                if track_id is not None:
                    label = f"#{track_id} {label}"

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw label
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes()
