/**
 * WebSocket Client
 * Manages real-time communication with backend
 */

class WebSocketClient {
    constructor(url) {
        this.url = url;
        this.ws = null;
        this.reconnectInterval = 3000;
        this.reconnectTimer = null;
        this.isConnected = false;
    }

    connect() {
        console.log('Connecting to WebSocket...');

        try {
            this.ws = new WebSocket(this.url);

            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.isConnected = true;
                this.updateConnectionStatus(true);

                // Clear reconnect timer
                if (this.reconnectTimer) {
                    clearTimeout(this.reconnectTimer);
                    this.reconnectTimer = null;
                }

                // Send ping every 30 seconds to keep connection alive
                this.startPingInterval();
            };

            this.ws.onmessage = (event) => {
                this.handleMessage(event.data);
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };

            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
                this.isConnected = false;
                this.updateConnectionStatus(false);
                this.stopPingInterval();

                // Attempt to reconnect
                this.reconnectTimer = setTimeout(() => {
                    this.connect();
                }, this.reconnectInterval);
            };

        } catch (error) {
            console.error('Failed to create WebSocket:', error);
            this.reconnectTimer = setTimeout(() => {
                this.connect();
            }, this.reconnectInterval);
        }
    }

    handleMessage(data) {
        try {
            const message = JSON.parse(data);

            switch (message.type) {
                case 'initial':
                    console.log('Received initial state:', message.data);
                    this.handleCountsUpdate(message.data);
                    break;

                case 'counts_update':
                    this.handleCountsUpdate(message.data);
                    break;

                case 'status_update':
                    this.handleStatusUpdate(message.data);
                    break;

                case 'pong':
                    // Received pong response
                    break;

                default:
                    console.log('Unknown message type:', message.type);
            }

        } catch (error) {
            console.error('Error parsing WebSocket message:', error);
        }
    }

    handleCountsUpdate(data) {
        // Update total count
        const totalCountElement = document.getElementById('totalCount');
        if (totalCountElement) {
            totalCountElement.textContent = data.total || 0;
        }

        // Update last update time
        const lastUpdateElement = document.getElementById('lastUpdate');
        if (lastUpdateElement && data.last_update_total) {
            const date = new Date(data.last_update_total);
            lastUpdateElement.textContent = `Last updated: ${date.toLocaleTimeString()}`;
        }

        // Update individual camera counts
        if (data.by_camera && cameraGrid) {
            cameraGrid.updateCounts(data.by_camera);
        }
    }

    handleStatusUpdate(data) {
        // Update system status
        const systemStatusElement = document.getElementById('systemStatus');
        if (systemStatusElement) {
            systemStatusElement.textContent = data.running ? 'Running' : 'Stopped';
        }

        // Update uptime
        const uptimeElement = document.getElementById('uptime');
        if (uptimeElement && data.uptime_seconds) {
            const hours = Math.floor(data.uptime_seconds / 3600);
            const minutes = Math.floor((data.uptime_seconds % 3600) / 60);
            uptimeElement.textContent = `Uptime: ${hours}h ${minutes}m`;
        }

        // Update active cameras
        const activeCamerasElement = document.getElementById('activeCameras');
        if (activeCamerasElement && data.camera_status) {
            const connectedCount = Object.values(data.camera_status)
                .filter(status => status.connected).length;
            activeCamerasElement.textContent = connectedCount;
        }
    }

    updateConnectionStatus(connected) {
        const statusDot = document.getElementById('connectionStatus');
        const statusText = document.getElementById('connectionText');

        if (statusDot) {
            statusDot.className = `status-dot ${connected ? 'connected' : 'disconnected'}`;
        }

        if (statusText) {
            statusText.textContent = connected ? 'Connected' : 'Disconnected';
        }
    }

    send(message) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(message));
        }
    }

    startPingInterval() {
        this.pingInterval = setInterval(() => {
            this.send({ type: 'ping' });
        }, 30000);
    }

    stopPingInterval() {
        if (this.pingInterval) {
            clearInterval(this.pingInterval);
            this.pingInterval = null;
        }
    }

    disconnect() {
        this.stopPingInterval();

        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }

        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }

    requestCounts() {
        this.send({ type: 'get_counts' });
    }

    requestStatus() {
        this.send({ type: 'get_status' });
    }
}

// Global WebSocket client instance
let wsClient = null;

// Initialize WebSocket client
function initWebSocket() {
    const wsUrl = 'ws://localhost:8000/api/ws';
    wsClient = new WebSocketClient(wsUrl);
    wsClient.connect();
}
