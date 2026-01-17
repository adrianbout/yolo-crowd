"""
Settings API
Endpoints for managing detection settings
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import json
from pathlib import Path
import logging

from .dependencies import get_state_manager, get_detection_service
from services.state_manager import StateManager

logger = logging.getLogger(__name__)

router = APIRouter()


class DetectionSettings(BaseModel):
    """Detection settings model"""
    img_size: int = Field(608, ge=320, le=1280, description="Input image size for YOLO")
    confidence_threshold: float = Field(0.25, ge=0.1, le=0.9, description="Confidence threshold")
    iou_threshold: float = Field(0.45, ge=0.1, le=0.9, description="IOU threshold for NMS")
    half_precision: bool = Field(True, description="Use FP16 for faster inference")
    denoise: bool = Field(False, description="Apply denoising (slow)")
    clahe: bool = Field(False, description="Apply CLAHE contrast enhancement")
    equalize_histogram: bool = Field(False, description="Apply histogram equalization")
    inference_interval: float = Field(0.1, ge=0.0, le=1.0, description="Seconds between inferences")
    frame_skip: int = Field(0, ge=0, le=30, description="Number of frames to skip between processing")


class SettingsResponse(BaseModel):
    """Response model for settings"""
    detection_settings: dict
    preprocessing_defaults: dict
    status: str


@router.get("/settings", response_model=SettingsResponse)
async def get_settings(state_manager: StateManager = Depends(get_state_manager)):
    """Get current detection settings"""
    detection_settings = state_manager.get_detection_settings()

    # Get preprocessing defaults from profiles
    preprocessing_defaults = {
        "denoise": False,
        "clahe": False,
        "equalize_histogram": False
    }

    # Check if any profile has preprocessing enabled
    for profile_name, profile in state_manager.profiles.get("profiles", {}).items():
        preproc = profile.get("preprocessing", {})
        if preproc.get("denoise"):
            preprocessing_defaults["denoise"] = True
        if preproc.get("clahe"):
            preprocessing_defaults["clahe"] = True
        if preproc.get("equalize_histogram"):
            preprocessing_defaults["equalize_histogram"] = True

    return SettingsResponse(
        detection_settings=detection_settings,
        preprocessing_defaults=preprocessing_defaults,
        status="ok"
    )


@router.put("/settings")
async def update_settings(
    settings: DetectionSettings,
    state_manager: StateManager = Depends(get_state_manager)
):
    """
    Update detection settings
    Note: Some settings require restart to take effect
    """
    config_path = state_manager.config_dir / "camera_profiles.json"

    try:
        # Load current config
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Update detection settings
        config["detection_settings"]["img_size"] = settings.img_size
        config["detection_settings"]["confidence_threshold"] = settings.confidence_threshold
        config["detection_settings"]["iou_threshold"] = settings.iou_threshold
        config["detection_settings"]["half_precision"] = settings.half_precision

        # Update preprocessing in all profiles
        for profile_name in config.get("profiles", {}):
            config["profiles"][profile_name]["preprocessing"]["denoise"] = settings.denoise
            config["profiles"][profile_name]["preprocessing"]["clahe"] = settings.clahe
            config["profiles"][profile_name]["preprocessing"]["equalize_histogram"] = settings.equalize_histogram

        # Save config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        # Also update frame_skip in cameras.json
        cameras_path = state_manager.config_dir / "cameras.json"
        with open(cameras_path, 'r') as f:
            cameras_config = json.load(f)
        cameras_config["global_settings"]["frame_skip"] = settings.frame_skip
        with open(cameras_path, 'w') as f:
            json.dump(cameras_config, f, indent=2)

        # Reload configuration in state manager
        state_manager.load_configuration()

        logger.info(f"Settings updated: img_size={settings.img_size}, conf={settings.confidence_threshold}, frame_skip={settings.frame_skip}")

        return {
            "status": "ok",
            "message": "Settings saved. Some changes may require restart to take effect.",
            "settings": settings.dict(),
            "requires_restart": ["img_size", "half_precision"]
        }

    except Exception as e:
        logger.error(f"Failed to update settings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update settings: {str(e)}")


@router.post("/settings/apply")
async def apply_settings_live(
    settings: DetectionSettings,
    state_manager: StateManager = Depends(get_state_manager),
    detection_service = Depends(get_detection_service)
):
    """
    Apply settings that can be changed without restart
    """
    try:
        # Update detector thresholds (these can be changed live)
        if detection_service.detector:
            detection_service.detector.confidence_threshold = settings.confidence_threshold
            detection_service.detector.iou_threshold = settings.iou_threshold

        # Update inference interval and frame skip
        detection_service.inference_interval = settings.inference_interval
        detection_service.frame_skip = settings.frame_skip

        # Save to file as well
        await update_settings(settings, state_manager)

        return {
            "status": "ok",
            "message": "Settings applied live",
            "applied_live": ["confidence_threshold", "iou_threshold", "inference_interval", "frame_skip"],
            "requires_restart": ["img_size", "half_precision", "denoise", "clahe", "equalize_histogram"]
        }

    except Exception as e:
        logger.error(f"Failed to apply settings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to apply settings: {str(e)}")
