"""
Camera API Endpoints
CRUD operations for cameras and settings
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Optional
from pydantic import BaseModel
import json
import logging
from pathlib import Path
from .dependencies import get_state_manager, get_detection_service

logger = logging.getLogger(__name__)

router = APIRouter()


class CameraSettings(BaseModel):
    """Camera settings update model"""
    profile: str
    enabled: bool


class CameraConnection(BaseModel):
    """Camera connection details"""
    ip: str = "localhost"
    rtsp_port: int = 554
    http_port: int = 80
    username: str = "admin"
    password: str = ""
    rtsp_url: str
    isapi_base: str = ""


class CameraPosition(BaseModel):
    """Camera position details"""
    description: str = ""
    floor: int = 1


class CameraDetectionSettings(BaseModel):
    """Per-camera detection settings override"""
    detection_model: str = "rgb"  # rgb, thermal, blob_hotspot
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    img_size: int = 640
    preprocessing: str = "none"  # none, clahe, equalize, denoise
    # Blob hotspot settings (only used when detection_model="blob_hotspot")
    blob_threshold: int = 200  # Brightness threshold (0-255)
    blob_min_area: int = 2000  # Minimum blob area
    blob_max_area: int = 50000  # Maximum blob area
    blob_aspect_ratio_min: float = 0.5  # Min height/width ratio
    blob_aspect_ratio_max: float = 3.0  # Max height/width ratio


class CameraCreate(BaseModel):
    """Model for creating a new camera"""
    name: str
    type: str = "rgb"  # rgb, thermal, infrared
    enabled: bool = True
    connection: CameraConnection
    profile: Optional[str] = None  # Will be auto-set based on type if not provided
    position: CameraPosition = CameraPosition()
    detection_settings: Optional[CameraDetectionSettings] = None  # Per-camera overrides


class CameraUpdate(BaseModel):
    """Model for updating a camera"""
    name: Optional[str] = None
    type: Optional[str] = None
    enabled: Optional[bool] = None
    connection: Optional[CameraConnection] = None
    profile: Optional[str] = None
    position: Optional[CameraPosition] = None
    detection_settings: Optional[CameraDetectionSettings] = None  # Per-camera overrides


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


@router.get("/profiles")
async def get_available_profiles(state_manager = Depends(get_state_manager)) -> Dict:
    """Get all available camera profiles"""
    return state_manager.profiles.get("profiles", {})


@router.post("/cameras")
async def create_camera(
    camera: CameraCreate,
    state_manager = Depends(get_state_manager)
) -> Dict:
    """Create a new camera"""
    # Generate unique ID
    existing_ids = [c["id"] for c in state_manager.cameras.get("cameras", [])]
    new_id = f"cam_{len(existing_ids) + 1:03d}"

    # Ensure unique ID
    counter = len(existing_ids) + 1
    while new_id in existing_ids:
        counter += 1
        new_id = f"cam_{counter:03d}"

    # Auto-set profile based on type if not provided
    profile = camera.profile
    if not profile:
        profile_map = {
            "rgb": "rgb_default",
            "thermal": "thermal_default",
            "infrared": "infrared_default"
        }
        profile = profile_map.get(camera.type, "rgb_default")

    # Create camera config
    new_camera = {
        "id": new_id,
        "name": camera.name,
        "type": camera.type,
        "enabled": camera.enabled,
        "connection": camera.connection.dict(),
        "profile": profile,
        "position": camera.position.dict(),
        "detection_settings": camera.detection_settings.dict() if camera.detection_settings else None
    }

    # Add to cameras list
    if "cameras" not in state_manager.cameras:
        state_manager.cameras["cameras"] = []
    state_manager.cameras["cameras"].append(new_camera)

    # Save to file
    _save_cameras_config(state_manager)

    logger.info(f"Created new camera: {new_id} - {camera.name}")

    return {"status": "ok", "camera": new_camera, "message": "Camera created. Restart server to connect."}


@router.put("/cameras/{camera_id}")
async def update_camera(
    camera_id: str,
    camera: CameraUpdate,
    state_manager = Depends(get_state_manager)
) -> Dict:
    """Update an existing camera"""
    # Find camera
    cameras_list = state_manager.cameras.get("cameras", [])
    camera_index = None

    for i, c in enumerate(cameras_list):
        if c["id"] == camera_id:
            camera_index = i
            break

    if camera_index is None:
        raise HTTPException(status_code=404, detail="Camera not found")

    # Update fields
    existing = cameras_list[camera_index]

    if camera.name is not None:
        existing["name"] = camera.name
    if camera.type is not None:
        existing["type"] = camera.type
    if camera.enabled is not None:
        existing["enabled"] = camera.enabled
    if camera.connection is not None:
        existing["connection"] = camera.connection.dict()
    if camera.profile is not None:
        existing["profile"] = camera.profile
    if camera.position is not None:
        existing["position"] = camera.position.dict()
    # Handle detection_settings - can be set to None to remove custom settings
    if "detection_settings" in camera.__fields_set__:
        existing["detection_settings"] = camera.detection_settings.dict() if camera.detection_settings else None

    # Save to file
    _save_cameras_config(state_manager)

    logger.info(f"Updated camera: {camera_id}")

    return {"status": "ok", "camera": existing, "message": "Camera updated. Restart server to apply changes."}


@router.delete("/cameras/{camera_id}")
async def delete_camera(
    camera_id: str,
    state_manager = Depends(get_state_manager)
) -> Dict:
    """Delete a camera"""
    cameras_list = state_manager.cameras.get("cameras", [])

    # Find and remove camera
    for i, c in enumerate(cameras_list):
        if c["id"] == camera_id:
            deleted = cameras_list.pop(i)
            _save_cameras_config(state_manager)
            logger.info(f"Deleted camera: {camera_id}")
            return {"status": "ok", "message": f"Camera {deleted['name']} deleted. Restart server to apply."}

    raise HTTPException(status_code=404, detail="Camera not found")


def _save_cameras_config(state_manager):
    """Save cameras configuration to file"""
    config_path = state_manager.config_dir / "cameras.json"
    with open(config_path, 'w') as f:
        json.dump(state_manager.cameras, f, indent=2)
    logger.info("Saved cameras configuration")
