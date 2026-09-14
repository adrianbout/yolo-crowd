"""
Floor plan calibration and projection.

Ties each camera's view to a shared floor plan, so traffic from eight
rectangles can be read as one building instead of eight separate pictures.

The link is a homography: a 3x3 transform between the camera's image and the
plan. Four matching point pairs determine it, and every extra pair improves
the fit. This works because both are flat - the floor is a plane, the plan is
a picture of that plane - and that same assumption is the one limit worth
remembering: only points *on the floor* project correctly. A person's feet
land where they are standing; the top of their head projects to somewhere
past them, further from the camera. So detections are projected by the bottom
edge of the box, never the centre.
"""

import json
import logging
import os
import threading
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Fewer than four pairs cannot determine a homography.
MIN_POINTS = 4

# Calibration points must span at least this many pixels on each axis of the
# camera image. Points crowded into one spot are perfectly solvable and still
# useless: the transform is fitted to a tiny patch and extrapolated across the
# whole frame, so a one-pixel click error becomes metres of drift at the far
# wall. Low enough never to reject a real survey.
MIN_SPREAD_PX = 20.0


class CameraCalibration:
    """
    One camera's mapping onto the plan.

    Holds the point pairs rather than only the resulting matrix, so a
    calibration can be reopened and corrected a point at a time instead of
    being redone from scratch.
    """

    def __init__(
        self,
        camera_id: str,
        camera_points: Optional[List[List[float]]] = None,
        plan_points: Optional[List[List[float]]] = None
    ):
        self.camera_id = camera_id
        self.camera_points: List[List[float]] = [list(p) for p in (camera_points or [])]
        self.plan_points: List[List[float]] = [list(p) for p in (plan_points or [])]
        self.homography: Optional[np.ndarray] = None
        self.error_px: Optional[float] = None
        self.solve()

    # -- solving -----------------------------------------------------------

    def solve(self) -> bool:
        """
        Recompute the transform from the current point pairs.

        RANSAC is used once there are more than the minimum four pairs: with
        points clicked by hand, one mis-click should be outvoted rather than
        allowed to bend the whole mapping.
        """
        self.homography = None
        self.error_px = None

        n = min(len(self.camera_points), len(self.plan_points))
        if n < MIN_POINTS:
            return False

        src = np.array(self.camera_points[:n], dtype=np.float32)
        dst = np.array(self.plan_points[:n], dtype=np.float32)

        spread = src.max(axis=0) - src.min(axis=0)
        if spread[0] < MIN_SPREAD_PX or spread[1] < MIN_SPREAD_PX:
            logger.warning(
                f"Camera {self.camera_id}: calibration points span only "
                f"{spread[0]:.0f}x{spread[1]:.0f}px - spread them across the "
                f"floor the camera sees, not one corner of it"
            )
            return False

        method = cv2.RANSAC if n > MIN_POINTS else 0
        H, _ = cv2.findHomography(src, dst, method, 5.0)
        if H is None:
            logger.warning(
                f"Camera {self.camera_id}: {n} point pairs did not yield a "
                f"homography - they are probably near-duplicate"
            )
            return False

        if not self._is_well_conditioned(H):
            # findHomography returns a matrix for degenerate input rather than
            # failing - four points along a corridor wall produce one with a
            # zero determinant. Accepting it would give a calibration that
            # looks finished and projects everything onto a line.
            logger.warning(
                f"Camera {self.camera_id}: {n} point pairs are degenerate - "
                f"they lie too close to a straight line to fix a perspective"
            )
            return False

        self.homography = H
        self.error_px = self._reprojection_error(src, dst, H)
        return True

    @staticmethod
    def _is_well_conditioned(H: np.ndarray) -> bool:
        """
        Whether a solved matrix describes a real perspective or a collapse.

        The condition number is used rather than the determinant because a
        homography is only defined up to scale - the determinant moves with
        that scale, while the conditioning does not. Points on a line send it
        to infinity; a legitimate shallow camera angle stays far below the
        threshold.
        """
        if H is None or not np.isfinite(H).all():
            return False
        try:
            return float(np.linalg.cond(H)) < 1e12
        except np.linalg.LinAlgError:
            return False

    @staticmethod
    def _reprojection_error(src: np.ndarray, dst: np.ndarray, H: np.ndarray) -> float:
        """
        Mean distance, in plan pixels, between where each clicked point lands
        and where it was said to be.

        Surfaced to the operator because a calibration that is subtly wrong
        looks exactly like one that is right until detections start landing
        through walls.
        """
        projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
        return float(np.mean(np.linalg.norm(projected - dst, axis=1)))

    @property
    def is_valid(self) -> bool:
        return self.homography is not None

    # -- projection --------------------------------------------------------

    def project(self, points) -> Optional[np.ndarray]:
        """
        Camera pixels to plan pixels. Returns None when uncalibrated.

        Points must lie on the floor plane to land correctly - pass the
        bottom edge of a detection box, not its centre.
        """
        if not self.is_valid or points is None or len(points) == 0:
            return None
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.homography).reshape(-1, 2)

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> Dict:
        d = {
            "camera_id": self.camera_id,
            "camera_points": self.camera_points,
            "plan_points": self.plan_points,
            "calibrated": self.is_valid,
            "point_count": min(len(self.camera_points), len(self.plan_points)),
        }
        # The matrix is derived, not authored - published so a client can
        # project without a round trip, but always recomputed on load.
        if self.is_valid:
            d["homography"] = self.homography.tolist()
            d["error_px"] = round(self.error_px, 2)
        return d

    @classmethod
    def from_dict(cls, camera_id: str, data: Dict) -> "CameraCalibration":
        return cls(camera_id, data.get("camera_points"), data.get("plan_points"))


class FloorPlan:
    """
    The plan image plus every camera's calibration against it.

    Saved next to the other configuration as plain JSON, so a site survey can
    be version-controlled and copied between nodes like any other config.
    """

    def __init__(self, config_dir: str = "config", filename: str = "floorplan.json"):
        self.config_dir = config_dir
        self.path = os.path.join(config_dir, filename)
        self.image: Optional[str] = None      # filename, relative to config_dir
        self.width: int = 0
        self.height: int = 0
        self.calibrations: Dict[str, CameraCalibration] = {}
        self._lock = threading.Lock()
        self.load()

    # -- persistence -------------------------------------------------------

    def load(self):
        if not os.path.exists(self.path):
            logger.info("No floor plan configured yet")
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError) as e:
            logger.error(f"Could not read floor plan config: {e}")
            return

        self.image = data.get("image")
        self.width = int(data.get("width", 0))
        self.height = int(data.get("height", 0))
        self.calibrations = {
            cam_id: CameraCalibration.from_dict(cam_id, cal)
            for cam_id, cal in (data.get("cameras") or {}).items()
        }
        ready = sum(1 for c in self.calibrations.values() if c.is_valid)
        logger.info(
            f"Floor plan loaded: {self.image} {self.width}x{self.height}, "
            f"{ready}/{len(self.calibrations)} cameras calibrated"
        )

    def save(self):
        with self._lock:
            data = {
                "image": self.image,
                "width": self.width,
                "height": self.height,
                "cameras": {cid: c.to_dict() for cid, c in self.calibrations.items()},
            }
        os.makedirs(self.config_dir, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Floor plan saved to {self.path}")

    # -- the plan image ----------------------------------------------------

    def set_image(self, filename: str, width: int, height: int):
        """
        Point the plan at a new image.

        Existing calibrations are kept but flagged, not silently dropped: if
        the new image is the same plan re-exported at the same size they are
        still good, and re-surveying every camera because someone re-uploaded
        a PNG would be its own kind of failure.
        """
        resized = (self.width, self.height) != (width, height) and self.width > 0
        self.image = filename
        self.width = int(width)
        self.height = int(height)
        if resized and self.calibrations:
            logger.warning(
                f"Floor plan dimensions changed to {width}x{height}; "
                f"{len(self.calibrations)} existing calibrations may no longer line up"
            )
        self.save()

    @property
    def has_image(self) -> bool:
        return bool(self.image) and self.width > 0 and self.height > 0

    # -- calibration -------------------------------------------------------

    def set_calibration(
        self,
        camera_id: str,
        camera_points: List[List[float]],
        plan_points: List[List[float]]
    ) -> CameraCalibration:
        """Store point pairs for a camera and solve. Raises if they cannot be solved."""
        if len(camera_points) != len(plan_points):
            raise ValueError(
                f"Need matching pairs: got {len(camera_points)} camera points "
                f"and {len(plan_points)} plan points"
            )
        if len(camera_points) < MIN_POINTS:
            raise ValueError(
                f"Need at least {MIN_POINTS} point pairs, got {len(camera_points)}"
            )

        cal = CameraCalibration(camera_id, camera_points, plan_points)
        if not cal.is_valid:
            raise ValueError(
                "Those points do not define a mapping. Spread them out across "
                "the floor - points along one line, or bunched in one corner, "
                "cannot fix a perspective."
            )
        with self._lock:
            self.calibrations[camera_id] = cal
        self.save()
        logger.info(
            f"Camera {camera_id} calibrated from {len(camera_points)} points, "
            f"error {cal.error_px:.1f}px"
        )
        return cal

    def clear_calibration(self, camera_id: str) -> bool:
        with self._lock:
            existed = self.calibrations.pop(camera_id, None) is not None
        if existed:
            self.save()
        return existed

    def get_calibration(self, camera_id: str) -> Optional[CameraCalibration]:
        return self.calibrations.get(camera_id)

    def is_calibrated(self, camera_id: str) -> bool:
        cal = self.calibrations.get(camera_id)
        return cal is not None and cal.is_valid

    # -- projection --------------------------------------------------------

    def project_detections(self, camera_id: str, detections: List) -> List[Dict]:
        """
        Put a camera's detections on the plan.

        Each detection contributes the bottom-centre of its box - where the
        person meets the floor - because that is the only point of a standing
        body that actually lies on the plane the homography describes.
        """
        cal = self.calibrations.get(camera_id)
        if cal is None or not cal.is_valid or not detections:
            return []

        # Accepts Detection objects from the detection loop and the plain
        # dicts the history buffer stores, so both callers use one path.
        def field(d, name):
            return d.get(name) if isinstance(d, dict) else getattr(d, name, None)

        feet, kept = [], []
        for d in detections:
            bbox = field(d, "bbox")
            if not bbox or len(bbox) < 4:
                continue
            x1, x2, y2 = bbox[0], bbox[2], bbox[3]
            feet.append([(x1 + x2) / 2.0, y2])
            kept.append(d)
        if not feet:
            return []

        projected = cal.project(feet)
        out = []
        for det, (px, py) in zip(kept, projected):
            row = {"x": round(float(px), 1), "y": round(float(py), 1)}
            tid = field(det, "track_id")
            if tid is not None:
                row["track_id"] = int(tid)
            out.append(row)
        return out

    def warp_grid(
        self,
        camera_id: str,
        grid: np.ndarray,
        frame_w: int,
        frame_h: int,
        out_w: Optional[int] = None,
        out_h: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        Warp one camera's heatmap grid into plan space.

        The grid is stored normalised, so it is first scaled up to the frame
        it was accumulated from and then carried through the homography in a
        single combined transform - one warp rather than two resamplings.
        """
        cal = self.calibrations.get(camera_id)
        if cal is None or not cal.is_valid or grid is None or not grid.size:
            return None

        out_w = out_w or self.width
        out_h = out_h or self.height
        if out_w <= 0 or out_h <= 0:
            return None

        gh, gw = grid.shape[:2]
        if gw < 2 or gh < 2 or frame_w <= 0 or frame_h <= 0:
            return None

        # Grid cell -> frame pixel. The grid spans the frame inclusively, so
        # the last cell centre sits on the last pixel, hence the -1 terms.
        scale = np.array([
            [frame_w / (gw - 1.0), 0.0, 0.0],
            [0.0, frame_h / (gh - 1.0), 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        combined = cal.homography.astype(np.float64) @ scale
        return cv2.warpPerspective(
            grid.astype(np.float32), combined, (out_w, out_h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )

    def combined_grid(self, grids_by_camera: Dict[str, Tuple[np.ndarray, int, int]]) -> Optional[np.ndarray]:
        """
        Every calibrated camera's heat, added together on one plan.

        Summed rather than averaged: where two cameras overlap, that floor
        genuinely was watched twice, and averaging would dim exactly the
        junctions and doorways that overlap gets used to cover.
        """
        if not self.has_image:
            return None
        total = np.zeros((self.height, self.width), dtype=np.float32)
        used = 0
        for camera_id, (grid, fw, fh) in grids_by_camera.items():
            warped = self.warp_grid(camera_id, grid, fw, fh)
            if warped is not None:
                total += warped
                used += 1
        if used == 0:
            return None
        peak = float(total.max())
        return total / peak if peak > 1e-6 else total

    # -- reporting ---------------------------------------------------------

    def status(self) -> Dict:
        """What is set up and what still needs a survey."""
        cams = {cid: c.to_dict() for cid, c in self.calibrations.items()}
        return {
            "has_image": self.has_image,
            "image": self.image,
            "width": self.width,
            "height": self.height,
            "cameras": cams,
            "calibrated_count": sum(1 for c in self.calibrations.values() if c.is_valid),
        }
