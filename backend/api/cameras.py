"""
Camera API Endpoints
CRUD operations for cameras and settings
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict
from pydantic import BaseModel
from .dependencies import get_state_manager, get_detection_service

router = APIRouter()


class CameraSettings(BaseModel):
    """Camera settings update model"""
    profile: str
    enabled: bool


@router.get("/cameras")
async def get_cameras(state_manager = Depends(get_state_manager)) -> List[Dict]:
    """Get all cameras"""
    return state_manager.get_all_cameras()


@router.get("/cameras/{camera_id}")
async def get_camera(camera_id: str, state_manager = Depends(get_state_manager)) -> Dict:
    """Get a specific camera"""
    camera = state_manager.get_camera_config(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    return camera


@router.get("/cameras/{camera_id}/status")
async def get_camera_status(camera_id: str, detection_service = Depends(get_detection_service)) -> Dict:
    """Get camera status"""
    if not detection_service.camera_manager:
        raise HTTPException(status_code=503, detail="Camera manager not initialized")

    status = detection_service.camera_manager.get_camera_status(camera_id)
    if "error" in status:
        raise HTTPException(status_code=404, detail=status["error"])

    return status


@router.get("/cameras/{camera_id}/frame")
async def get_camera_frame(camera_id: str, detection_service = Depends(get_detection_service), draw_rois: bool = True):
    """Get current camera frame with detections and ROIs"""
    from fastapi.responses import Response

    frame_bytes = detection_service.get_frame_with_detections(camera_id, draw_rois)
    if frame_bytes is None:
        raise HTTPException(status_code=404, detail="Frame not available")

    return Response(content=frame_bytes, media_type="image/jpeg")


@router.get("/cameras/{camera_id}/profile")
async def get_camera_profile(camera_id: str, state_manager = Depends(get_state_manager)) -> Dict:
    """Get camera profile"""
    profile = state_manager.get_camera_profile(camera_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile
