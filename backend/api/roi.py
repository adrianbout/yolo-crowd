"""
ROI API Endpoints
Manage ROI polygons for cameras
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict
from pydantic import BaseModel

router = APIRouter()


class ROIPolygon(BaseModel):
    """ROI polygon model"""
    name: str
    points: List[List[int]]
    description: str = ""


class ROIUpdate(BaseModel):
    """ROI update model"""
    enabled: bool
    polygons: List[ROIPolygon]
    notes: str = ""


@router.get("/roi")
async def get_all_rois(state_manager) -> Dict:
    """Get all ROI configurations"""
    return state_manager.get_all_rois()


@router.get("/roi/{camera_id}")
async def get_camera_roi(camera_id: str, state_manager) -> Dict:
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
async def update_camera_roi(camera_id: str, roi_update: ROIUpdate, detection_service) -> Dict:
    """Update ROI configuration for a camera"""
    roi_config = {
        "enabled": roi_update.enabled,
        "polygons": [p.dict() for p in roi_update.polygons],
        "notes": roi_update.notes
    }

    detection_service.update_roi(camera_id, roi_config)

    return {"message": "ROI updated successfully", "camera_id": camera_id}


@router.delete("/roi/{camera_id}")
async def clear_camera_roi(camera_id: str, detection_service) -> Dict:
    """Clear ROI configuration for a camera"""
    roi_config = {
        "enabled": False,
        "polygons": [],
        "notes": "ROI cleared"
    }

    detection_service.update_roi(camera_id, roi_config)

    return {"message": "ROI cleared successfully", "camera_id": camera_id}
