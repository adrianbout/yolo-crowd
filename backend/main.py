"""
FastAPI Main Application
Entry point for the backend server
"""

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import asyncio
from contextlib import asynccontextmanager

from services.state_manager import StateManager
from services.detection_service import DetectionService
from api import cameras, roi, counting, websocket, dependencies

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
state_manager = StateManager(config_dir="config")
detection_service = DetectionService(state_manager, batch_size=1, inference_interval=0.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    # Startup
    logger.info("Starting SmartChairCounter application...")

    try:
        # Initialize detection service
        detection_service.initialize()

        # Start detection service
        detection_service.start()

        # Start broadcast task
        asyncio.create_task(broadcast_task())

        logger.info("Application started successfully")

    except Exception as e:
        logger.error(f"Failed to start application: {e}", exc_info=True)
        raise

    yield

    # Shutdown
    logger.info("Shutting down application...")
    detection_service.stop()
    logger.info("Application shut down")


# Create FastAPI app
app = FastAPI(
    title="SmartChairCounter API",
    description="Real-time chair counting system with multi-camera support",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency injection
def get_state_manager():
    return state_manager


def get_detection_service():
    return detection_service


# Include routers
app.include_router(
    cameras.router,
    prefix="/api",
    tags=["cameras"]
)

app.include_router(
    roi.router,
    prefix="/api",
    tags=["roi"]
)

app.include_router(
    counting.router,
    prefix="/api",
    tags=["counting"]
)

# Override dependency functions globally
app.dependency_overrides[dependencies.get_state_manager] = get_state_manager
app.dependency_overrides[dependencies.get_detection_service] = get_detection_service

app.include_router(
    websocket.router,
    prefix="/api",
    tags=["websocket"]
)


# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "SmartChairCounter API",
        "version": "1.0.0",
        "status": "running"
    }


# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "detection_service": detection_service.running,
        "system": state_manager.get_system_status()
    }


# System status
@app.get("/api/status")
async def get_system_status():
    """Get complete system status"""
    camera_status = detection_service.camera_manager.get_all_status() if detection_service.camera_manager else {}

    # Add median counts and override status to camera data
    for camera_id in camera_status:
        # yolo_median: always the calculated median (for display only)
        yolo_median = state_manager._calculate_median(camera_id)
        # adjusted_count: manual override if set, otherwise yolo_median (used for total)
        adjusted_count = state_manager.get_median_count(camera_id)
        has_override = state_manager.manual_overrides.get(camera_id) is not None

        camera_status[camera_id]["yolo_median"] = yolo_median
        camera_status[camera_id]["adjusted_count"] = adjusted_count
        camera_status[camera_id]["has_override"] = has_override

    return {
        "system": state_manager.get_system_status(),
        "service": detection_service.get_statistics(),
        "cameras": camera_status
    }


# Background task for broadcasting updates
async def broadcast_task():
    """Periodically broadcast count updates to WebSocket clients"""
    logger.info("Starting broadcast task...")

    while detection_service.running:
        try:
            await asyncio.sleep(1)  # Broadcast every second

            # Broadcast count update
            if len(websocket.manager.active_connections) > 0:
                await websocket.broadcast_count_update(state_manager)

        except Exception as e:
            logger.error(f"Error in broadcast task: {e}")
            await asyncio.sleep(1)

    logger.info("Broadcast task ended")


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Not found", "detail": str(exc)}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        log_level="info"
    )
