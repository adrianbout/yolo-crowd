"""
WebSocket API for real-time updates
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
import asyncio
import json
import logging
from .dependencies import get_state_manager, get_detection_service

logger = logging.getLogger(__name__)

router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections"""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        disconnected = []

        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending message to client: {e}")
                disconnected.append(connection)

        # Remove disconnected clients
        for connection in disconnected:
            if connection in self.active_connections:
                self.active_connections.remove(connection)


manager = ConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    state_manager = Depends(get_state_manager),
    detection_service = Depends(get_detection_service)
):
    """WebSocket endpoint for real-time count updates"""
    await manager.connect(websocket)

    try:
        # Send initial state
        initial_state = {
            "type": "initial",
            "data": state_manager.get_counts_summary()
        }
        await websocket.send_json(initial_state)

        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Receive message (ping/pong or requests)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)

                message = json.loads(data)

                # Handle different message types
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})

                elif message.get("type") == "get_counts":
                    counts = state_manager.get_counts_summary()
                    await websocket.send_json({
                        "type": "counts_update",
                        "data": counts
                    })

                elif message.get("type") == "get_status":
                    status = detection_service.get_statistics()
                    await websocket.send_json({
                        "type": "status_update",
                        "data": status
                    })

            except asyncio.TimeoutError:
                # No message received, continue
                pass

            except json.JSONDecodeError:
                logger.error("Invalid JSON received")

            # Small delay
            await asyncio.sleep(0.1)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


async def broadcast_count_update(state_manager):
    """Broadcast count update to all connected clients"""
    counts = state_manager.get_counts_summary()
    message = {
        "type": "counts_update",
        "data": counts
    }
    await manager.broadcast(message)


# Export manager for use in main app
__all__ = ["router", "manager", "broadcast_count_update"]
