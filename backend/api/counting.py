"""
Counting API Endpoints
Get counts and manage manual overrides
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict
from pydantic import BaseModel
from .dependencies import get_state_manager, get_detection_service

router = APIRouter()


class CountOverride(BaseModel):
    """Manual count override model"""
    count: int


@router.get("/counting/total")
async def get_total_count(state_manager = Depends(get_state_manager)) -> Dict:
    """Get total count across all cameras"""
    return {
        "total": state_manager.get_total_count(),
        "timestamp": state_manager.last_update_total.isoformat() if state_manager.last_update_total else None
    }


@router.get("/counting/summary")
async def get_counts_summary(state_manager = Depends(get_state_manager)) -> Dict:
    """Get complete counts summary"""
    return state_manager.get_counts_summary()


@router.get("/counting/{camera_id}")
async def get_camera_count(camera_id: str, state_manager = Depends(get_state_manager)) -> Dict:
    """Get count for a specific camera"""
    count = state_manager.get_count(camera_id)
    if count is None:
        raise HTTPException(status_code=404, detail="Camera not found")

    return {
        "camera_id": camera_id,
        "count": count,
        "timestamp": state_manager.last_update.get(camera_id).isoformat() if camera_id in state_manager.last_update else None
    }


@router.put("/counting/{camera_id}/override")
async def override_camera_count(camera_id: str, override: CountOverride, state_manager = Depends(get_state_manager)) -> Dict:
    """Manual override for camera median count"""
    camera = state_manager.get_camera_config(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    state_manager.manual_override_count(camera_id, override.count)

    return {
        "message": "Count overridden successfully",
        "camera_id": camera_id,
        "new_count": override.count
    }


@router.delete("/counting/{camera_id}/override")
async def clear_camera_override(camera_id: str, state_manager = Depends(get_state_manager)) -> Dict:
    """Clear manual override for camera median count (return to auto-calculated median)"""
    camera = state_manager.get_camera_config(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    state_manager.clear_manual_override(camera_id)

    return {
        "message": "Override cleared successfully",
        "camera_id": camera_id,
        "median_count": state_manager.get_median_count(camera_id)
    }


@router.post("/counting/reset")
async def reset_all_counts(state_manager = Depends(get_state_manager)) -> Dict:
    """Reset all counts to zero"""
    state_manager.reset_counts()
    return {"message": "All counts reset to zero"}


@router.get("/counting/{camera_id}/detections")
async def get_camera_detections(camera_id: str, detection_service = Depends(get_detection_service), limit: int = 1) -> Dict:
    """Get recent detections for a camera"""
    history = detection_service.state_manager.get_detection_history(camera_id, limit)
    return {
        "camera_id": camera_id,
        "history": history
    }
