"""
Pose YOLO Detector
People detection with posture classification (sitting / standing) using
Ultralytics YOLOv11-pose. Counts ALL people like the RGB detector, but tags
each detection with a posture so the frontend / API can show or filter it.

Pipeline per frame:
  1. Run pose inference (boxes + 17 COCO keypoints per person).
  2. Drop duplicate detections of the same person via keypoint-box IoU.
  3. Feed detections into a per-camera BoT-SORT tracker to get a stable
     per-person track_id across frames (matched back to detections by IoU,
     since BoT-SORT's own internal detection index isn't a reliable handle
     once detections have been filtered/reordered upstream).
  4. Classify posture per detection from knee angle (primary), falling back
     to a torso/thigh ratio when the angle is ambiguous or unavailable, and
     to "sitting" when the legs are fully occluded.
  5. For tracked people, apply a motion override (moving -> standing, unless
     the knee angle is bent enough to trust sitting regardless) and smooth
     the displayed label over recent frames so it doesn't flicker.

This detector is fully opt-in per camera via detection_settings.detection_model = "pose".
If the pose weights file is missing, DetectorFactory falls back to the RGB detector,
so the existing flow is never affected.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field

from ultralytics import YOLO
from ultralytics.trackers import BOTSORT
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml

from .detector import Detection

logger = logging.getLogger(__name__)

# COCO keypoint indices used for posture estimation
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
KP_LEFT_KNEE = 13
KP_RIGHT_KNEE = 14
KP_LEFT_ANKLE = 15
KP_RIGHT_ANKLE = 16

# Minimum keypoint confidence to trust a point for posture math
KP_CONF_MIN = 0.25

# --- Posture classification (knee-angle based) -----------------------------
# Primary cue: the interior angle at the knee, between the thigh (hip->knee)
# and shin (ankle->knee). A straight leg reads close to 180 deg; a bent knee
# reads well under that.
KNEE_ANGLE_SIT_MAX = 130.0     # angle <= this -> sitting
KNEE_ANGLE_STAND_MIN = 170.0   # angle >= this -> standing, if not contradicted (see below)
# Between the two: ambiguous -> torso/thigh ratio breaks the tie. Also used
# when the knee angle itself can't be computed (e.g. ankle occluded) but
# hip/knee/shoulder are visible, and as a corroboration check on a
# knee-angle "standing" verdict: on a steep/top-down camera, a person seated
# but leaning far forward can project hip/knee/ankle into a near-straight
# line (faking a straight standing leg) -- if the ratio clearly disagrees,
# it wins.
TORSO_THIGH_STAND_RATIO = 0.6  # thigh_len / torso_len >= this -> standing

# --- Duplicate detections (same person detected twice) ---------------------
DUPLICATE_KEYPOINT_IOU = 0.7

# --- Motion override --------------------------------------------------------
MOTION_HISTORY_FRAMES = 10        # centroid samples kept per track for motion check
MOTION_DISPLACEMENT_RATIO = 0.12  # net displacement / bbox height to count as "moving"
KNEE_ANGLE_SIT_TRUST = 100.0      # below this, always sitting - motion can't override

# --- Temporal label smoothing (avoid frame-to-frame flicker) ---------------
SMOOTH_WINDOW = 15
SMOOTH_AGREEMENT_RATIO = 0.65

# --- Track state bookkeeping -------------------------------------------------
TRACK_STALE_FRAMES = 30   # matches BoT-SORT's own default track_buffer
TRACK_MATCH_MIN_IOU = 0.3  # min bbox IoU to accept a tracker row <-> detection match


def _bbox_iou(a: List[float], b: List[float]) -> float:
    """Standard IoU between two [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class _TrackState:
    """Per-track (per-person) state, kept across frames for one camera."""
    centroid_history: deque = field(default_factory=lambda: deque(maxlen=MOTION_HISTORY_FRAMES))
    label_history: deque = field(default_factory=lambda: deque(maxlen=SMOOTH_WINDOW))
    displayed_label: Optional[str] = None
    last_seen_frame: int = 0


class PoseYOLODetector:
    """
    YOLOv11-pose detector with per-camera BoT-SORT tracking and temporally
    smoothed sitting/standing classification. Same interface as YOLODetector /
    ThermalYOLODetector (detect_batch / detect_single returning Detection
    objects), so it slots into DetectorFactory with no changes to the
    detection loop.
    """

    def __init__(
        self,
        model_path: str = "weights/yolo11m-pose.pt",
        device: str = "cuda",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        img_size: int = 640
    ):
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Pose model file not found: {model_path}")

        logger.info(f"Loading Pose YOLO model from {model_path}")
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

        logger.info(f"Pose YOLO model loaded successfully on {self.device}")

        # Per-camera tracking state (lazily created on first frame seen)
        self._trackers: Dict[str, BOTSORT] = {}
        self._track_states: Dict[str, Dict[int, _TrackState]] = {}
        self._frame_counters: Dict[str, int] = {}

    def _get_tracker(self, camera_id: str) -> BOTSORT:
        """Get (or lazily create) this camera's own BoT-SORT tracker instance.

        Each camera needs its own tracker: frames from different cameras are
        interleaved through this same detector, and a shared tracker would mix
        their detections into one bogus "stream".
        """
        tracker = self._trackers.get(camera_id)
        if tracker is None:
            cfg = IterableSimpleNamespace(**YAML.load(check_yaml("botsort.yaml")))
            tracker = BOTSORT(args=cfg, frame_rate=30)
            self._trackers[camera_id] = tracker
            self._track_states[camera_id] = {}
            self._frame_counters[camera_id] = 0
            logger.info(f"Created BoT-SORT tracker for camera {camera_id}")
        return tracker

    # --- Duplicate removal --------------------------------------------------

    @staticmethod
    def _keypoint_bbox(kp_xy: np.ndarray, kp_conf: np.ndarray) -> Optional[List[float]]:
        """Tight bbox around a person's confidently-visible keypoints only."""
        mask = kp_conf >= KP_CONF_MIN
        if mask.sum() < 2:
            return None
        pts = kp_xy[mask]
        return [float(pts[:, 0].min()), float(pts[:, 1].min()),
                float(pts[:, 0].max()), float(pts[:, 1].max())]

    def _dedupe_by_keypoints(
        self, kp_xy: np.ndarray, kp_conf: np.ndarray, confs: List[float]
    ) -> List[int]:
        """
        Suppress duplicate detections of the same person: when two detections'
        keypoint boxes overlap heavily, keep only the higher-confidence one.
        Standard greedy NMS, but on keypoint geometry rather than the box head's
        (often near-identical, occlusion-prone) bounding boxes.
        """
        n = len(confs)
        if n <= 1:
            return list(range(n))

        kp_boxes = [self._keypoint_bbox(kp_xy[i], kp_conf[i]) for i in range(n)]
        order = sorted(range(n), key=lambda i: confs[i], reverse=True)

        keep: List[int] = []
        for i in order:
            box_i = kp_boxes[i]
            is_duplicate = False
            if box_i is not None:
                for j in keep:
                    box_j = kp_boxes[j]
                    if box_j is not None and _bbox_iou(box_i, box_j) >= DUPLICATE_KEYPOINT_IOU:
                        is_duplicate = True
                        break
            if not is_duplicate:
                keep.append(i)

        return sorted(keep)

    # --- Posture geometry ----------------------------------------------------

    @staticmethod
    def _knee_angle_deg(hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray) -> float:
        """Interior angle at the knee between thigh (knee->hip) and shin (knee->ankle)."""
        v1, v2 = hip - knee, ankle - knee
        n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
        if n1 < 1e-6 or n2 < 1e-6:
            return 180.0
        cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_theta)))

    def _compute_knee_angle(self, kp_xy: np.ndarray, kp_conf: np.ndarray) -> Optional[float]:
        """Average the knee angle over whichever leg(s) have hip+knee+ankle visible."""
        def pt(idx):
            return kp_xy[idx] if kp_conf[idx] >= KP_CONF_MIN else None

        angles = []
        for hip_i, knee_i, ankle_i in (
            (KP_LEFT_HIP, KP_LEFT_KNEE, KP_LEFT_ANKLE),
            (KP_RIGHT_HIP, KP_RIGHT_KNEE, KP_RIGHT_ANKLE),
        ):
            hip, knee, ankle = pt(hip_i), pt(knee_i), pt(ankle_i)
            if hip is not None and knee is not None and ankle is not None:
                angles.append(self._knee_angle_deg(hip, knee, ankle))

        return sum(angles) / len(angles) if angles else None

    def _torso_thigh_ratio(self, kp_xy: np.ndarray, kp_conf: np.ndarray) -> Optional[float]:
        """thigh_len / torso_len -- extended thigh relative to torso leans standing."""
        def pt(idx):
            return kp_xy[idx] if kp_conf[idx] >= KP_CONF_MIN else None

        def avg(a_idx, b_idx):
            a, b = pt(a_idx), pt(b_idx)
            if a is not None and b is not None:
                return (a + b) / 2.0
            return a if a is not None else b

        shoulder = avg(KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER)
        hip = avg(KP_LEFT_HIP, KP_RIGHT_HIP)
        knee = avg(KP_LEFT_KNEE, KP_RIGHT_KNEE)
        if shoulder is None or hip is None or knee is None:
            return None

        torso_len = float(np.linalg.norm(hip - shoulder))
        if torso_len < 1e-3:
            return None
        return float(np.linalg.norm(knee - hip)) / torso_len

    def _classify_posture(
        self, knee_angle: Optional[float], kp_xy: np.ndarray, kp_conf: np.ndarray
    ) -> str:
        """
        "sitting" | "standing", from knee angle primarily, torso/thigh ratio as
        a tie-breaker, "sitting" as the default when the legs are unreadable.

        A knee-angle "standing" verdict also needs the torso/thigh ratio not to
        contradict it: a forward-leaning seated pose can align hip/knee/ankle
        into a near-straight line under a steep camera angle, faking a straight
        leg, and the ratio (usually) doesn't fall for the same illusion.
        """
        ratio = self._torso_thigh_ratio(kp_xy, kp_conf)

        if knee_angle is not None:
            if knee_angle <= KNEE_ANGLE_SIT_MAX:
                return "sitting"
            if knee_angle >= KNEE_ANGLE_STAND_MIN:
                if ratio is not None and ratio < TORSO_THIGH_STAND_RATIO:
                    return "sitting"  # ratio contradicts the angle -> don't trust it
                return "standing"

        if ratio is not None:
            return "standing" if ratio >= TORSO_THIGH_STAND_RATIO else "sitting"

        # Legs (and/or torso) fully occluded: no usable cue -> default sitting
        return "sitting"

    # --- Motion + temporal smoothing -----------------------------------------

    @staticmethod
    def _is_moving(state: _TrackState) -> bool:
        history = state.centroid_history
        if len(history) < 3:
            return False
        x0, y0, h0 = history[0]
        x1, y1, h1 = history[-1]
        disp = float(np.hypot(x1 - x0, y1 - y0))
        scale = max((h0 + h1) / 2.0, 1e-3)
        return (disp / scale) >= MOTION_DISPLACEMENT_RATIO

    @staticmethod
    def _smooth_label(state: _TrackState, raw_label: str) -> str:
        """Majority-vote over the last SMOOTH_WINDOW frames; only flip the
        displayed label once the new label has a clear (>=65%) majority, so a
        single noisy frame doesn't flicker the shown posture."""
        state.label_history.append(raw_label)
        if state.displayed_label is None:
            state.displayed_label = raw_label
            return state.displayed_label

        counts: Dict[str, int] = {}
        for lbl in state.label_history:
            counts[lbl] = counts.get(lbl, 0) + 1
        majority_label, majority_count = max(counts.items(), key=lambda kv: kv[1])
        agreement = majority_count / len(state.label_history)

        if majority_label != state.displayed_label and agreement >= SMOOTH_AGREEMENT_RATIO:
            state.displayed_label = majority_label

        return state.displayed_label

    # --- Main detection entry point ------------------------------------------

    def detect_batch(
        self,
        frames: List[np.ndarray],
        camera_ids: List[str],
        inference_configs: List[Dict],
        preprocessing_configs: Optional[List[Dict]] = None
    ) -> Dict[str, List[Detection]]:
        """
        Perform detection + tracking + posture tagging on multiple frames.
        Counts every person (posture is a tag, not a filter), matching RGB
        behavior -- duplicate keypoint detections of the same person are the
        only thing this drops from the count.
        """
        if not frames:
            return {}

        detections_by_camera: Dict[str, List[Detection]] = {}

        for frame, camera_id, inference_config in zip(frames, camera_ids, inference_configs):
            conf_thresh = inference_config.get("confidence_threshold", self.confidence_threshold)
            iou_thresh = inference_config.get("iou_threshold", self.iou_threshold)
            img_size = inference_config.get("img_size", self.img_size)

            results = self.model(
                frame,
                conf=conf_thresh,
                iou=iou_thresh,
                imgsz=img_size,
                verbose=False,
                classes=[0]  # Person class only
            )

            raw_bboxes: List[List[float]] = []
            raw_confs: List[float] = []
            raw_classes: List[int] = []
            kp_xy = None
            kp_conf = None
            boxes = None

            if len(results) > 0 and results[0].boxes is not None:
                result = results[0]
                boxes = result.boxes
                keypoints = result.keypoints

                if keypoints is not None and keypoints.xy is not None:
                    kp_xy = keypoints.xy.cpu().numpy()
                    if keypoints.conf is not None:
                        kp_conf = keypoints.conf.cpu().numpy()

                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    raw_bboxes.append([float(x1), float(y1), float(x2), float(y2)])
                    raw_confs.append(float(box.conf[0].cpu().numpy()))
                    raw_classes.append(int(box.cls[0].cpu().numpy()))

            if not raw_bboxes:
                detections_by_camera[camera_id] = []
                continue

            # --- Duplicate removal ------------------------------------------
            if kp_xy is not None:
                keep_idx = self._dedupe_by_keypoints(kp_xy, kp_conf, raw_confs)
            else:
                keep_idx = list(range(len(raw_bboxes)))

            bboxes = [raw_bboxes[i] for i in keep_idx]
            confs = [raw_confs[i] for i in keep_idx]
            classes = [raw_classes[i] for i in keep_idx]
            det_kp_xy = kp_xy[keep_idx] if kp_xy is not None else None
            det_kp_conf = kp_conf[keep_idx] if kp_conf is not None else None

            # --- Tracking -----------------------------------------------------
            tracker = self._get_tracker(camera_id)
            self._frame_counters[camera_id] += 1
            frame_idx = self._frame_counters[camera_id]

            track_boxes = boxes.cpu().numpy()[np.array(keep_idx, dtype=int)]
            try:
                tracks = tracker.update(track_boxes, frame) if len(track_boxes) else np.empty((0, 8))
            except Exception as e:
                logger.warning(f"BoT-SORT update failed for camera {camera_id}: {e}")
                tracks = np.empty((0, 8))

            track_id_for_idx: List[Optional[int]] = [None] * len(bboxes)
            for trow in tracks:
                tbox = [float(trow[0]), float(trow[1]), float(trow[2]), float(trow[3])]
                tid = int(trow[4])
                best_i, best_iou = -1, 0.0
                for i, bbox in enumerate(bboxes):
                    if track_id_for_idx[i] is not None:
                        continue
                    iou = _bbox_iou(tbox, bbox)
                    if iou > best_iou:
                        best_i, best_iou = i, iou
                if best_i >= 0 and best_iou >= TRACK_MATCH_MIN_IOU:
                    track_id_for_idx[best_i] = tid

            # --- Classify + motion override + smoothing ------------------------
            states = self._track_states.setdefault(camera_id, {})
            detections: List[Detection] = []

            for i, bbox in enumerate(bboxes):
                kxy = det_kp_xy[i] if det_kp_xy is not None else None
                kconf = det_kp_conf[i] if det_kp_conf is not None else None
                tid = track_id_for_idx[i]

                if kxy is None:
                    label = "unknown"
                    knee_angle = None
                else:
                    knee_angle = self._compute_knee_angle(kxy, kconf)
                    label = self._classify_posture(knee_angle, kxy, kconf)

                if tid is not None:
                    state = states.setdefault(tid, _TrackState())
                    state.last_seen_frame = frame_idx

                    if label in ("sitting", "standing"):
                        cx = (bbox[0] + bbox[2]) / 2.0
                        cy = (bbox[1] + bbox[3]) / 2.0
                        h = max(bbox[3] - bbox[1], 1e-3)
                        state.centroid_history.append((cx, cy, h))

                        if knee_angle is not None and knee_angle < KNEE_ANGLE_SIT_TRUST:
                            label = "sitting"  # bent knee: trusted regardless of motion
                        elif self._is_moving(state):
                            label = "standing"

                        label = self._smooth_label(state, label)

                detections.append(Detection(
                    bbox=bbox,
                    confidence=confs[i],
                    class_id=classes[i],
                    posture=label,
                    track_id=tid
                ))

            # Drop state for tracks not seen in a while, so memory doesn't grow
            # unbounded over a long-running stream.
            stale = [tid for tid, st in states.items() if frame_idx - st.last_seen_frame > TRACK_STALE_FRAMES]
            for tid in stale:
                del states[tid]

            detections_by_camera[camera_id] = detections

        return detections_by_camera

    def detect_single(
        self,
        frame: np.ndarray,
        inference_config: Dict,
        preprocessing_config: Optional[Dict] = None
    ) -> List[Detection]:
        """Detect on a single frame."""
        result = self.detect_batch(
            frames=[frame],
            camera_ids=["single"],
            inference_configs=[inference_config],
            preprocessing_configs=[preprocessing_config] if preprocessing_config else None
        )
        return result.get("single", [])
