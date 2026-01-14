"""
Camera Stream Manager
Handles RTSP streams from multiple cameras with frame buffering
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List
import threading
import logging
import time
from queue import Queue, Empty
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CameraInfo:
    """Camera information"""
    camera_id: str
    name: str
    rtsp_url: str
    enabled: bool
    fps_target: int = 10


class CameraStream:
    """
    Single camera stream handler
    Runs in separate thread to continuously read frames
    """

    def __init__(
        self,
        camera_id: str,
        rtsp_url: str,
        buffer_size: int = 1,
        reconnect_attempts: int = 3,
        reconnect_delay: int = 5
    ):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.buffer_size = buffer_size
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay

        self.frame_queue: Queue = Queue(maxsize=buffer_size)
        self.latest_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()

        self.capture: Optional[cv2.VideoCapture] = None
        self.running = False
        self.connected = False
        self.thread: Optional[threading.Thread] = None

        self.frame_count = 0
        self.fps = 0.0
        self.last_fps_time = time.time()

        # Detect if this is a video file (not an RTSP stream)
        self.is_video_file = not self.rtsp_url.lower().startswith(('rtsp://', 'http://', 'https://'))

    def start(self):
        """Start the camera stream"""
        if self.running:
            logger.warning(f"Camera {self.camera_id} is already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logger.info(f"Started camera stream: {self.camera_id}")

    def stop(self):
        """Stop the camera stream"""
        self.running = False

        if self.thread:
            self.thread.join(timeout=5)

        if self.capture:
            self.capture.release()

        logger.info(f"Stopped camera stream: {self.camera_id}")

    def _connect(self) -> bool:
        """
        Connect to RTSP stream
        Returns:
            True if connected successfully
        """
        try:
            logger.info(f"Connecting to camera {self.camera_id}: {self.rtsp_url}")

            self.capture = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

            # Set buffer size (reduce latency)
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

            # Try to read a frame to verify connection
            ret, frame = self.capture.read()
            if ret and frame is not None:
                self.connected = True
                logger.info(f"Successfully connected to camera {self.camera_id}")
                return True
            else:
                self.connected = False
                logger.error(f"Failed to read frame from camera {self.camera_id}")
                return False

        except Exception as e:
            self.connected = False
            logger.error(f"Error connecting to camera {self.camera_id}: {e}")
            return False

    def _capture_loop(self):
        """Main capture loop running in thread"""
        attempt = 0

        while self.running:
            # Try to connect
            if not self.connected:
                if attempt >= self.reconnect_attempts:
                    logger.error(
                        f"Camera {self.camera_id}: Max reconnect attempts reached"
                    )
                    break

                if self._connect():
                    attempt = 0
                else:
                    attempt += 1
                    logger.warning(
                        f"Camera {self.camera_id}: Reconnect attempt {attempt}/{self.reconnect_attempts}"
                    )
                    time.sleep(self.reconnect_delay)
                    continue

            # Read frame
            try:
                ret, frame = self.capture.read()

                if not ret or frame is None:
                    # For video files, loop back to the beginning
                    if self.is_video_file:
                        # Silently loop - don't spam logs
                        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = self.capture.read()

                        # If still can't read, reconnect
                        if not ret or frame is None:
                            logger.warning(f"Camera {self.camera_id}: Failed to loop video, reconnecting...")
                            self.connected = False
                            continue
                    else:
                        # For streams, reconnect
                        logger.warning(f"Camera {self.camera_id}: Failed to read frame")
                        self.connected = False
                        continue

                # Update latest frame (no copy needed here, copy on read)
                with self.frame_lock:
                    self.latest_frame = frame

                # Update FPS
                self.frame_count += 1
                current_time = time.time()
                elapsed = current_time - self.last_fps_time

                if elapsed >= 1.0:
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.last_fps_time = current_time

            except Exception as e:
                logger.error(f"Camera {self.camera_id}: Error reading frame: {e}")
                self.connected = False
                time.sleep(1)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest frame
        Returns:
            Latest frame or None if not available
        """
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
        return None

    def is_connected(self) -> bool:
        """Check if camera is connected"""
        return self.connected

    def get_fps(self) -> float:
        """Get current FPS"""
        return self.fps


class CameraStreamManager:
    """
    Manages multiple camera streams
    """

    def __init__(self, buffer_size: int = 1):
        self.buffer_size = buffer_size
        self.streams: Dict[str, CameraStream] = {}
        self.camera_info: Dict[str, CameraInfo] = {}

    def add_camera(
        self,
        camera_id: str,
        name: str,
        rtsp_url: str,
        enabled: bool = True,
        fps_target: int = 10,
        reconnect_attempts: int = 3,
        reconnect_delay: int = 5
    ):
        """
        Add a camera to the manager
        Args:
            camera_id: Camera identifier
            name: Camera name
            rtsp_url: RTSP URL
            enabled: Whether camera is enabled
            fps_target: Target FPS
            reconnect_attempts: Number of reconnection attempts
            reconnect_delay: Delay between reconnections (seconds)
        """
        self.camera_info[camera_id] = CameraInfo(
            camera_id=camera_id,
            name=name,
            rtsp_url=rtsp_url,
            enabled=enabled,
            fps_target=fps_target
        )

        if enabled:
            stream = CameraStream(
                camera_id=camera_id,
                rtsp_url=rtsp_url,
                buffer_size=self.buffer_size,
                reconnect_attempts=reconnect_attempts,
                reconnect_delay=reconnect_delay
            )
            self.streams[camera_id] = stream
            logger.info(f"Added camera: {camera_id} - {name}")

    def start_all(self):
        """Start all camera streams"""
        logger.info(f"Starting {len(self.streams)} camera streams...")

        for camera_id, stream in self.streams.items():
            stream.start()

        logger.info("All camera streams started")

    def stop_all(self):
        """Stop all camera streams"""
        logger.info("Stopping all camera streams...")

        for camera_id, stream in self.streams.items():
            stream.stop()

        logger.info("All camera streams stopped")

    def get_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """
        Get latest frame from a camera
        Args:
            camera_id: Camera identifier
        Returns:
            Frame or None if not available
        """
        if camera_id not in self.streams:
            return None

        return self.streams[camera_id].get_latest_frame()

    def get_all_frames(self) -> Dict[str, np.ndarray]:
        """
        Get latest frames from all cameras
        Returns:
            Dictionary mapping camera_id to frame
        """
        frames = {}

        for camera_id, stream in self.streams.items():
            frame = stream.get_latest_frame()
            if frame is not None:
                frames[camera_id] = frame

        return frames

    def get_batch_frames(self, camera_ids: list) -> Tuple[List[np.ndarray], List[str]]:
        """
        Get frames for a batch of cameras
        Args:
            camera_ids: List of camera IDs
        Returns:
            Tuple of (frames, valid_camera_ids)
        """
        frames = []
        valid_ids = []

        for camera_id in camera_ids:
            frame = self.get_frame(camera_id)
            if frame is not None:
                frames.append(frame)
                valid_ids.append(camera_id)

        return frames, valid_ids

    def get_camera_status(self, camera_id: str) -> Dict:
        """
        Get status information for a camera
        Args:
            camera_id: Camera identifier
        Returns:
            Status dictionary
        """
        if camera_id not in self.camera_info:
            return {"error": "Camera not found"}

        info = self.camera_info[camera_id]
        status = {
            "camera_id": camera_id,
            "name": info.name,
            "enabled": info.enabled,
            "connected": False,
            "fps": 0.0
        }

        if camera_id in self.streams:
            stream = self.streams[camera_id]
            status["connected"] = stream.is_connected()
            status["fps"] = stream.get_fps()

        return status

    def get_all_status(self) -> Dict[str, Dict]:
        """
        Get status for all cameras
        Returns:
            Dictionary mapping camera_id to status
        """
        status = {}
        for camera_id in self.camera_info.keys():
            status[camera_id] = self.get_camera_status(camera_id)
        return status

    def load_from_config(self, cameras_config: Dict):
        """
        Load cameras from configuration
        Args:
            cameras_config: Configuration from cameras.json
        """
        logger.info("Loading cameras from configuration")

        global_settings = cameras_config.get("global_settings", {})
        reconnect_attempts = global_settings.get("reconnect_attempts", 3)
        reconnect_delay = global_settings.get("reconnect_delay_seconds", 5)

        for camera_data in cameras_config.get("cameras", []):
            self.add_camera(
                camera_id=camera_data["id"],
                name=camera_data["name"],
                rtsp_url=camera_data["connection"]["rtsp_url"],
                enabled=camera_data.get("enabled", True),
                reconnect_attempts=reconnect_attempts,
                reconnect_delay=reconnect_delay
            )

        logger.info(f"Loaded {len(self.camera_info)} cameras from configuration")
