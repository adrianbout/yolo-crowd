"""
ROI API Endpoints
Manage ROI polygons for cameras
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Optional
from pydantic import BaseModel
from .dependencies import get_state_manager, get_detection_service

router = APIRouter()


class ROIGate(BaseModel):
    """
    Two-point line across a flow zone, with named sides.

    Names are what the dashboard shows for a crossing ("lobby -> offices"),
    so they should read as places rather than as A and B.
    """
    points: List[List[int]]  # exactly 2 points
    side_a: str = "A"
    side_b: str = "B"
    name: str = "gate"


class ROIPolygon(BaseModel):
    """ROI polygon model"""
    name: str
    points: List[List[int]]
    description: str = ""
    # Seating zones: seats in this zone. Falls back to the camera's
    # totalChairs when unset, so pre-existing cameras keep their capacity.
    capacity: Optional[int] = None
    # Flow zones: the gate whose crossings are counted.
    gate: Optional[ROIGate] = None


class ROIUpdate(BaseModel):
    """ROI update model"""
    enabled: bool
    polygons: List[ROIPolygon]
    notes: str = ""


@router.get("/roi")
async def get_all_rois(state_manager = Depends(get_state_manager)) -> Dict:
    """Get all ROI configurations"""
    return state_manager.get_all_rois()


@router.get("/roi/{camera_id}")
async def get_camera_roi(camera_id: str, state_manager = Depends(get_state_manager)) -> Dict:
    """Get ROI configuration for a camera"""
    roi = state_manager.get_roi(camera_id)
    if not roi:
        return {
            "enabled": False,
            "polygons": [],
            "notes": "No ROI defined"
        }
    return roi


@router.put("/roi/{camera_id}")
async def update_camera_roi(camera_id: str, roi_update: ROIUpdate, detection_service = Depends(get_detection_service)) -> Dict:
    """Update ROI configuration for a camera"""
    roi_config = {
        "enabled": roi_update.enabled,
        # exclude_none keeps a seating zone free of an empty gate and a flow
        # zone free of a null capacity, so the saved config stays readable.
        "polygons": [p.model_dump(exclude_none=True) for p in roi_update.polygons],
        "notes": roi_update.notes
    }

    detection_service.update_roi(camera_id, roi_config)

    return {"message": "ROI updated successfully", "camera_id": camera_id}


@router.delete("/roi/{camera_id}")
async def clear_camera_roi(camera_id: str, detection_service = Depends(get_detection_service)) -> Dict:
    """Clear ROI configuration for a camera"""
    roi_config = {
        "enabled": False,
        "polygons": [],
        "notes": "ROI cleared"
    }

    detection_service.update_roi(camera_id, roi_config)

    return {"message": "ROI cleared successfully", "camera_id": camera_id}
