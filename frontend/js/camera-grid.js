/**
 * Camera Grid Management
 * Handles display and updates for camera feeds
 */

const API_BASE = 'http://localhost:8000/api';

class CameraGrid {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.cameras = [];
        this.counts = {};
    }

    async loadCameras() {
        try {
            const response = await fetch(`${API_BASE}/cameras`);
            if (!response.ok) throw new Error('Failed to fetch cameras');

            this.cameras = await response.json();
            this.renderGrid();
        } catch (error) {
            console.error('Error loading cameras:', error);
        }
    }

    renderGrid() {
        this.container.innerHTML = '';

        this.cameras.forEach(camera => {
            const card = this.createCameraCard(camera);
            this.container.appendChild(card);
        });

        // Start updating camera feeds
        this.startFeedUpdates();
    }

    createCameraCard(camera) {
        const card = document.createElement('div');
        card.className = 'camera-card';
        card.id = `camera-${camera.id}`;

        card.innerHTML = `
            <div class="camera-header">
                <div class="camera-name">${camera.name}</div>
                <div class="camera-status">
                    <span class="camera-status-dot" id="status-${camera.id}"></span>
                    <span id="status-text-${camera.id}">Connecting...</span>
                </div>
            </div>
            <div class="camera-feed" id="feed-${camera.id}">
                <span>Loading camera feed...</span>
            </div>
            <div class="camera-info">
                <div class="camera-count">
                    <span class="count-label">Current Count:</span>
                    <span class="count-value" id="count-${camera.id}">0</span>
                </div>
                <div class="camera-actions">
                    <button class="btn btn-primary btn-small" onclick="openROIEditor('${camera.id}', '${camera.name}')">
                        Edit ROI
                    </button>
                    <button class="btn btn-secondary btn-small" onclick="viewCameraDetails('${camera.id}')">
                        Details
                    </button>
                </div>
            </div>
        `;

        return card;
    }

    startFeedUpdates() {
        // Update camera feeds every 2 seconds
        this.feedUpdateInterval = setInterval(() => {
            this.cameras.forEach(camera => {
                this.updateCameraFeed(camera.id);
            });
        }, 2000);
    }

    async updateCameraFeed(cameraId) {
        try {
            const feedElement = document.getElementById(`feed-${cameraId}`);
            if (!feedElement) return;

            // Create or update img element
            let img = feedElement.querySelector('img');
            if (!img) {
                feedElement.innerHTML = '';
                img = document.createElement('img');
                feedElement.appendChild(img);
            }

            // Update image source with timestamp to avoid caching
            img.src = `${API_BASE}/cameras/${cameraId}/frame?draw_rois=true&t=${Date.now()}`;

            // Update status
            this.updateCameraStatus(cameraId, true);

        } catch (error) {
            console.error(`Error updating feed for camera ${cameraId}:`, error);
            this.updateCameraStatus(cameraId, false);
        }
    }

    updateCameraStatus(cameraId, connected) {
        const statusDot = document.getElementById(`status-${cameraId}`);
        const statusText = document.getElementById(`status-text-${cameraId}`);

        if (statusDot) {
            statusDot.className = `camera-status-dot ${connected ? 'connected' : ''}`;
        }

        if (statusText) {
            statusText.textContent = connected ? 'Connected' : 'Disconnected';
        }
    }

    updateCount(cameraId, count) {
        const countElement = document.getElementById(`count-${cameraId}`);
        if (countElement) {
            countElement.textContent = count;
        }
        this.counts[cameraId] = count;
    }

    updateCounts(countsByCamera) {
        Object.entries(countsByCamera).forEach(([cameraId, count]) => {
            this.updateCount(cameraId, count);
        });
    }

    stopFeedUpdates() {
        if (this.feedUpdateInterval) {
            clearInterval(this.feedUpdateInterval);
        }
    }
}

// Global camera grid instance
let cameraGrid = null;

// Initialize camera grid
function initCameraGrid() {
    cameraGrid = new CameraGrid('cameraGrid');
    cameraGrid.loadCameras();
}

// View camera details
async function viewCameraDetails(cameraId) {
    try {
        const response = await fetch(`${API_BASE}/cameras/${cameraId}`);
        const camera = await response.json();

        const statusResponse = await fetch(`${API_BASE}/cameras/${cameraId}/status`);
        const status = await statusResponse.json();

        alert(`Camera Details:\n\nName: ${camera.name}\nType: ${camera.type}\nStatus: ${status.connected ? 'Connected' : 'Disconnected'}\nFPS: ${status.fps.toFixed(2)}`);
    } catch (error) {
        console.error('Error fetching camera details:', error);
        alert('Failed to fetch camera details');
    }
}
