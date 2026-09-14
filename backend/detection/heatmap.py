"""
Occupancy heatmap for flow zones.

Accumulates where people were seen, rather than how many. A flow camera's
count is a rate - crossings per minute - which says nothing about *where* in
the zone people walk. The heatmap answers that, and is what the dashboard
draws over the zone.

Deliberately built from the detections the batched pipeline already produces
rather than from `ultralytics.solutions.Heatmap`. Solutions processes one
frame per instance and would force flow cameras off the batched path, giving
up the single largest throughput win available. Splatting centres into a
small grid costs microseconds and keeps every camera in one batch.
"""

import logging
import threading
import time
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraHeatmap:
    """
    One camera's accumulator: a small grid in normalised coordinates.

    The grid is normalised rather than pixel-sized so it survives a camera
    changing resolution or img_size - the stored history stays meaningful
    instead of being invalidated by a config change.

    Weight decays exponentially, so the grid shows recent activity rather
    than everything since start-up. Without decay a camera left running for
    a week saturates and every cell reads the same.
    """

    def __init__(
        self,
        grid_w: int = 64,
        grid_h: int = 36,
        half_life: float = 300.0,
        blur: int = 5
    ):
        if grid_w < 2 or grid_h < 2:
            raise ValueError(f"Grid must be at least 2x2, got {grid_w}x{grid_h}")
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.half_life = half_life
        self.blur = blur if blur % 2 == 1 else blur + 1  # cv2 needs odd kernels
        self.grid = np.zeros((grid_h, grid_w), dtype=np.float32)
        self.observations = 0
        self.last_update = time.time()
        # The frame the grid was accumulated against. Kept because the grid
        # itself is normalised: projecting it onto a floor plan needs the
        # pixel dimensions back.
        self.frame_w = 0
        self.frame_h = 0
        # Bumped whenever the *shape* of the grid changes. Decay scales every
        # cell equally and normalised() divides by the peak, so ageing alone
        # leaves the rendered picture identical - which is what lets the
        # overlay be cached between inference ticks.
        self.version = 0

    def _decay_to(self, now: float):
        """
        Age the grid to `now`.

        Applied from elapsed time rather than per-tick so the half-life means
        the same thing whatever the inference interval is - and so a camera
        that stalls for a minute comes back correctly faded instead of
        keeping a stale peak.
        """
        dt = now - self.last_update
        if dt <= 0:
            return
        if self.half_life > 0:
            self.grid *= float(0.5 ** (dt / self.half_life))
        self.last_update = now

    def add(
        self,
        centers: Iterable[Tuple[float, float]],
        frame_w: int,
        frame_h: int,
        now: Optional[float] = None
    ):
        """
        Splat detection centres into the grid.

        Weight is spread bilinearly over the four neighbouring cells so a
        person drifting slowly across a boundary produces a smooth track
        rather than a staircase of hard cell hits.
        """
        if frame_w <= 0 or frame_h <= 0:
            return
        now = time.time() if now is None else now
        self._decay_to(now)
        self.frame_w, self.frame_h = int(frame_w), int(frame_h)

        for cx, cy in centers:
            # Normalise, then clamp: a box centre can sit just outside the
            # frame when a detection is clipped at the edge.
            nx = min(max(cx / frame_w, 0.0), 1.0) * (self.grid_w - 1)
            ny = min(max(cy / frame_h, 0.0), 1.0) * (self.grid_h - 1)

            x0, y0 = int(nx), int(ny)
            x1, y1 = min(x0 + 1, self.grid_w - 1), min(y0 + 1, self.grid_h - 1)
            fx, fy = nx - x0, ny - y0

            self.grid[y0, x0] += (1 - fx) * (1 - fy)
            self.grid[y0, x1] += fx * (1 - fy)
            self.grid[y1, x0] += (1 - fx) * fy
            self.grid[y1, x1] += fx * fy
            self.observations += 1
        self.version += 1

    def normalised(self, now: Optional[float] = None) -> np.ndarray:
        """
        The grid as 0..1, smoothed, or all zeros when nothing is accumulated.

        The floor on the divisor stops a nearly empty grid being stretched
        to full scale, which would render faint noise as a hot zone.
        """
        self._decay_to(time.time() if now is None else now)
        peak = float(self.grid.max())
        if peak < 1e-3:
            return np.zeros_like(self.grid)
        smoothed = cv2.GaussianBlur(self.grid, (self.blur, self.blur), 0)
        peak = float(smoothed.max())
        if peak < 1e-6:
            return np.zeros_like(self.grid)
        return smoothed / peak

    def reset(self):
        self.grid.fill(0.0)
        self.observations = 0
        self.last_update = time.time()
        self.version += 1

    def to_dict(self) -> Dict:
        """Serialisable form: the normalised grid plus what it was built from."""
        grid = self.normalised()
        return {
            "width": self.grid_w,
            "height": self.grid_h,
            "half_life": self.half_life,
            "observations": self.observations,
            "peak": round(float(self.grid.max()), 3),
            "grid": [[round(float(v), 4) for v in row] for row in grid],
        }


class HeatmapManager:
    """
    Per-camera heatmaps, created on demand.

    The detection loop writes and the stream and API read, on different
    threads, so every public method holds the lock. Grids are tiny - 64x36
    floats - so the contention this introduces is not measurable against the
    inference it sits next to.
    """

    def __init__(
        self,
        grid_w: int = 64,
        grid_h: int = 36,
        half_life: float = 300.0
    ):
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.half_life = half_life
        self._maps: Dict[str, CameraHeatmap] = {}
        self._overlay_cache: Dict[str, Tuple] = {}
        self._lock = threading.Lock()

    def _get_or_create(self, camera_id: str) -> CameraHeatmap:
        hm = self._maps.get(camera_id)
        if hm is None:
            hm = CameraHeatmap(self.grid_w, self.grid_h, self.half_life)
            self._maps[camera_id] = hm
            logger.info(f"Heatmap started for camera {camera_id}")
        return hm

    def update(self, camera_id: str, detections: List, frame_shape: Tuple):
        """
        Feed one camera's detections for this tick.

        Takes Detection objects straight from the batched path - the same
        list the counts are built from - so the heatmap can never disagree
        with what the dashboard reports.
        """
        if not frame_shape or len(frame_shape) < 2:
            return
        h, w = frame_shape[0], frame_shape[1]
        centers = [d.center for d in detections if getattr(d, "center", None)]
        with self._lock:
            self._get_or_create(camera_id).add(centers, w, h)

    def get(self, camera_id: str) -> Optional[Dict]:
        """The serialisable grid, or None if this camera has never been fed."""
        with self._lock:
            hm = self._maps.get(camera_id)
            return hm.to_dict() if hm else None

    def raw_grids(self) -> Dict[str, Tuple[np.ndarray, int, int]]:
        """
        Every camera's normalised grid with the frame size it was built from.

        For projection onto a floor plan, which needs to undo the
        normalisation before applying the camera's homography.
        """
        with self._lock:
            return {
                cid: (hm.normalised(), hm.frame_w, hm.frame_h)
                for cid, hm in self._maps.items()
                if hm.frame_w > 0 and hm.frame_h > 0
            }

    def reset(self, camera_id: Optional[str] = None):
        """Clear one camera, or all of them when no id is given."""
        with self._lock:
            if camera_id is None:
                for hm in self._maps.values():
                    hm.reset()
            elif camera_id in self._maps:
                self._maps[camera_id].reset()

    def discard(self, camera_id: str):
        """Drop a camera's map entirely - on delete, or a role change away from flow."""
        with self._lock:
            self._overlay_cache.pop(camera_id, None)
            if self._maps.pop(camera_id, None) is not None:
                logger.info(f"Heatmap discarded for camera {camera_id}")

    def render_overlay(
        self,
        frame: np.ndarray,
        camera_id: str,
        alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Blend the heatmap over a frame for the live view.

        Cold cells are left fully transparent rather than tinted deep blue:
        a JET colormap applied flat would wash the whole frame and make the
        underlying video unreadable where nothing has happened.

        Three things keep this off the stream's critical path. The coloured
        layer is cached per camera and frame size, and rebuilt only when the
        grid changes - twice a second against a stream asking ten times a
        second. Colouring happens at grid resolution and is upscaled, not the
        reverse. And only the rectangle that actually contains heat is
        blended, which on a typical corridor is about a quarter of the frame.
        """
        h, w = frame.shape[:2]

        with self._lock:
            hm = self._maps.get(camera_id)
            if hm is None:
                return frame
            key = (hm.version, w, h, alpha, colormap)
            cached = self._overlay_cache.get(camera_id)
            if cached is not None and cached[0] == key:
                layer = cached[1]
            else:
                layer = self._build_layer(hm, w, h, alpha, colormap)
                self._overlay_cache[camera_id] = (key, layer)

        if layer is None:
            return frame

        # Blended outside the lock: the cached arrays are never mutated, and
        # holding the lock through the blend would stall the detection loop
        # behind every streamed frame.
        (y0, y1, x0, x1), coloured, weight, inv_weight = layer
        out = frame.copy()
        out[y0:y1, x0:x1] = cv2.blendLinear(
            frame[y0:y1, x0:x1].astype(np.float32), coloured, inv_weight, weight
        ).astype(np.uint8)
        return out

    @staticmethod
    def _build_layer(hm: CameraHeatmap, w: int, h: int, alpha: float, colormap: int):
        """
        The cached pieces of the blend, or None when there is nothing to draw.

        Everything is cropped to the region carrying visible heat, so the
        per-frame work scales with how much of the view sees traffic rather
        than with the frame size. Argument order matters at the call site:
        cv2.blendLinear computes (src1*w1 + src2*w2) / (w1 + w2), so the
        frame takes the inverse weight and the colour takes the heat.
        """
        grid = hm.normalised()
        if not grid.any():
            return None

        weight = cv2.resize(grid, (w, h), interpolation=cv2.INTER_LINEAR) * alpha

        # Below this the tint is under one grey level - not worth blending.
        hot = weight > 0.01
        rows, cols = np.where(hot.any(1))[0], np.where(hot.any(0))[0]
        if rows.size == 0 or cols.size == 0:
            return None
        y0, y1 = int(rows[0]), int(rows[-1]) + 1
        x0, x1 = int(cols[0]), int(cols[-1]) + 1

        small = (np.clip(grid, 0.0, 1.0) * 255).astype(np.uint8)
        coloured = cv2.resize(cv2.applyColorMap(small, colormap), (w, h),
                              interpolation=cv2.INTER_LINEAR)

        crop_w = weight[y0:y1, x0:x1]
        return ((y0, y1, x0, x1),
                coloured[y0:y1, x0:x1].astype(np.float32),
                crop_w,
                1.0 - crop_w)
