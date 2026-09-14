"""
Floor Plan API

Upload a plan, match each camera's view to it, and read traffic back in plan
coordinates instead of camera pixels.
"""

import logging
import os
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .dependencies import get_detection_service, get_state_manager

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_IMAGE_BYTES = 25 * 1024 * 1024

# The plan arrives as a raw request body rather than a multipart form, so the
# format is read from the bytes themselves. This deliberately avoids the
# python-multipart dependency, which is not installed in the project
# environment; a browser sends a File straight through fetch() as the body.
MAGIC = (
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"\xff\xd8\xff", ".jpg"),
)


def _sniff_suffix(raw: bytes) -> Optional[str]:
    """File extension from the leading bytes, or None if it is not an image we take."""
    for magic, suffix in MAGIC:
        if raw.startswith(magic):
            return suffix
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return ".webp"
    return None


class CalibrationRequest(BaseModel):
    """
    Matching points between one camera's view and the plan.

    Both lists are pixel coordinates in their own image, and the pairs are
    positional: camera_points[i] is the same physical spot as plan_points[i].
    """
    camera_points: List[List[float]]
    plan_points: List[List[float]]


@router.get("/floorplan")
async def get_floorplan(detection_service=Depends(get_detection_service)) -> Dict:
    """The plan, and which cameras have been matched to it."""
    return detection_service.floorplan.status()


@router.get("/floorplan/image")
async def get_floorplan_image(detection_service=Depends(get_detection_service)):
    """The plan image itself."""
    fp = detection_service.floorplan
    if not fp.has_image:
        raise HTTPException(status_code=404, detail="No floor plan uploaded yet")

    path = os.path.join(fp.config_dir, fp.image)
    if not os.path.exists(path):
        # The config points at a file that is no longer on disk - worth saying
        # plainly rather than returning an empty image.
        raise HTTPException(
            status_code=404,
            detail=f"Floor plan config references '{fp.image}', which is missing from disk"
        )
    return FileResponse(path)


@router.post("/floorplan/image")
async def upload_floorplan_image(
    request: Request,
    detection_service=Depends(get_detection_service)
) -> Dict:
    """
    Replace the plan image. Send the file as the raw request body.

    Existing calibrations are kept. If the new image is the same plan at the
    same pixel size they are still correct; if it is not, the response says
    so rather than deleting a survey someone spent an afternoon on.
    """
    raw = await request.body()
    if not raw:
        raise HTTPException(status_code=400, detail="That file is empty")

    suffix = _sniff_suffix(raw)
    if suffix is None:
        raise HTTPException(
            status_code=400,
            detail="Send a PNG, JPEG or WebP image as the request body"
        )

    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"Plan is {len(raw) // (1024*1024)}MB; the limit is {MAX_IMAGE_BYTES // (1024*1024)}MB"
        )

    # Decoded before saving: a file that OpenCV cannot read would leave the
    # plan pointing at something the projection code can never measure.
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="That file is not a readable image")

    fp = detection_service.floorplan
    height, width = image.shape[:2]
    previous = (fp.width, fp.height) if fp.has_image else None

    filename = "floorplan" + suffix
    os.makedirs(fp.config_dir, exist_ok=True)
    with open(os.path.join(fp.config_dir, filename), "wb") as f:
        f.write(raw)

    fp.set_image(filename, width, height)
    logger.info(f"Floor plan uploaded: {filename} {width}x{height}")

    resized = previous is not None and previous != (width, height)
    return {
        "image": filename,
        "width": width,
        "height": height,
        "calibrated_count": fp.status()["calibrated_count"],
        "recalibration_needed": resized,
        "message": (
            f"Plan size changed from {previous[0]}x{previous[1]} to {width}x{height}; "
            f"existing calibrations will not line up and should be redone"
            if resized else "Plan updated"
        ),
    }


@router.get("/floorplan/cameras/{camera_id}/calibration")
async def get_calibration(
    camera_id: str,
    state_manager=Depends(get_state_manager),
    detection_service=Depends(get_detection_service)
) -> Dict:
    """One camera's point pairs, so a survey can be reopened and corrected."""
    if not state_manager.get_camera_config(camera_id):
        raise HTTPException(status_code=404, detail="Camera not found")

    cal = detection_service.floorplan.get_calibration(camera_id)
    if cal is None:
        return {"camera_id": camera_id, "calibrated": False,
                "camera_points": [], "plan_points": [], "point_count": 0}
    return cal.to_dict()


@router.put("/floorplan/cameras/{camera_id}/calibration")
async def set_calibration(
    camera_id: str,
    body: CalibrationRequest,
    state_manager=Depends(get_state_manager),
    detection_service=Depends(get_detection_service)
) -> Dict:
    """
    Match a camera to the plan with at least four point pairs.

    Pick spots that are unambiguous in both pictures and on the floor -
    door thresholds, pillar bases, corners of floor markings. Points on
    walls or furniture tops are not on the floor plane and will bend the
    whole mapping.
    """
    if not state_manager.get_camera_config(camera_id):
        raise HTTPException(status_code=404, detail="Camera not found")

    fp = detection_service.floorplan
    if not fp.has_image:
        raise HTTPException(
            status_code=409,
            detail="Upload a floor plan before matching cameras to it"
        )

    try:
        cal = fp.set_calibration(camera_id, body.camera_points, body.plan_points)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        **cal.to_dict(),
        # Surfaced because a calibration that is subtly wrong looks finished.
        "quality": _describe_error(cal.error_px),
    }


@router.delete("/floorplan/cameras/{camera_id}/calibration")
async def clear_calibration(
    camera_id: str,
    detection_service=Depends(get_detection_service)
) -> Dict:
    """Forget a camera's match to the plan."""
    removed = detection_service.floorplan.clear_calibration(camera_id)
    if not removed:
        raise HTTPException(status_code=404, detail="That camera has no calibration")
    return {"camera_id": camera_id, "status": "cleared"}


@router.get("/floorplan/heatmap")
async def get_floorplan_heatmap(detection_service=Depends(get_detection_service)) -> Dict:
    """
    Traffic from every calibrated camera, combined on the plan.

    Overlapping cameras add rather than average, so a junction watched twice
    reads as busy rather than being dimmed by its own redundancy.
    """
    fp = detection_service.floorplan
    if not fp.has_image:
        raise HTTPException(status_code=404, detail="No floor plan uploaded yet")

    heatmap = detection_service.get_floorplan_heatmap()
    return {
        "plan_width": fp.width,
        "plan_height": fp.height,
        "calibrated_count": fp.status()["calibrated_count"],
        # None when no calibrated camera has seen anyone yet - which the UI
        # should show as "no traffic", not as an empty plan.
        "heatmap": heatmap,
    }


@router.get("/floorplan/detections")
async def get_floorplan_detections(detection_service=Depends(get_detection_service)) -> Dict:
    """Where each camera's current detections stand, in plan coordinates."""
    fp = detection_service.floorplan
    if not fp.has_image:
        raise HTTPException(status_code=404, detail="No floor plan uploaded yet")

    by_camera = detection_service.get_floorplan_detections()
    return {
        "plan_width": fp.width,
        "plan_height": fp.height,
        "cameras": by_camera,
        "total": sum(len(v) for v in by_camera.values()),
    }


def _describe_error(error_px: Optional[float]) -> str:
    """
    Plain words for the reprojection error, in plan pixels.

    A number alone does not tell an operator whether to redo the survey.
    """
    if error_px is None:
        return "unknown"
    if error_px < 3:
        return "excellent"
    if error_px < 10:
        return "good"
    if error_px < 25:
        return "usable, but check the points sit on the floor"
    return "poor - the points probably do not match, or some are not on the floor"
