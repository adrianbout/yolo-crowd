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
        this.medianCounts = {};
        this.hasOverride = {};
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
                    <span class="count-label">YOLO Count:</span>
                    <span class="count-value" id="count-${camera.id}">0</span>
                </div>
                <div class="camera-yolo-median">
                    <span class="count-label">YOLO Median:</span>
                    <span class="yolo-median-value" id="yolo-median-${camera.id}">0</span>
                </div>
                <div class="camera-median-count">
                    <span class="count-label">Adjusted Count:</span>
                    <div class="median-controls">
                        <button class="btn-adjust btn-minus" onclick="adjustMedianCount('${camera.id}', -1)">-</button>
                        <span class="median-value" id="median-${camera.id}">0</span>
                        <button class="btn-adjust btn-plus" onclick="adjustMedianCount('${camera.id}', 1)">+</button>
                        <button class="btn-reset-override" id="reset-override-${camera.id}" onclick="clearOverride('${camera.id}')" style="display: none;" title="Reset to YOLO median">&#x21ba;</button>
                    </div>
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
        // Update camera feeds every 100ms (10 FPS) like quick_view
        this.feedUpdateInterval = setInterval(() => {
            this.cameras.forEach(camera => {
                this.updateCameraFeed(camera.id);
            });
        }, 100);
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

    updateMedianCount(cameraId, medianCount, hasOverride) {
        const medianElement = document.getElementById(`median-${cameraId}`);
        if (medianElement) {
            medianElement.textContent = medianCount;
            // Highlight if manual override is active
            if (hasOverride) {
                medianElement.classList.add('has-override');
            } else {
                medianElement.classList.remove('has-override');
            }
        }
        this.medianCounts[cameraId] = medianCount;
        this.hasOverride[cameraId] = hasOverride;

        // Show/hide reset button
        const resetBtn = document.getElementById(`reset-override-${cameraId}`);
        if (resetBtn) {
            resetBtn.style.display = hasOverride ? 'inline-block' : 'none';
        }
    }

    updateMedianCounts(cameraData) {
        Object.entries(cameraData).forEach(([cameraId, data]) => {
            // Update YOLO median (always the calculated value)
            if (data.yolo_median !== undefined) {
                this.updateYoloMedian(cameraId, data.yolo_median);
            }
            // Update adjusted count (manual override or yolo median)
            if (data.adjusted_count !== undefined) {
                this.updateMedianCount(cameraId, data.adjusted_count, data.has_override || false);
            }
        });
    }

    updateYoloMedian(cameraId, yoloMedian) {
        const yoloMedianElement = document.getElementById(`yolo-median-${cameraId}`);
        if (yoloMedianElement) {
            yoloMedianElement.textContent = yoloMedian;
        }
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

// Adjust median count manually
async function adjustMedianCount(cameraId, delta) {
    try {
        // Get current median count
        const currentCount = cameraGrid.medianCounts[cameraId] || 0;
        const newCount = Math.max(0, currentCount + delta);

        // Send override to API
        const response = await fetch(`${API_BASE}/counting/${cameraId}/override`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ count: newCount })
        });

        if (!response.ok) throw new Error('Failed to update count');

        // Update UI immediately
        cameraGrid.updateMedianCount(cameraId, newCount, true);

    } catch (error) {
        console.error('Error adjusting median count:', error);
    }
}

// Clear manual override
async function clearOverride(cameraId) {
    try {
        const response = await fetch(`${API_BASE}/counting/${cameraId}/override`, {
            method: 'DELETE'
        });

        if (!response.ok) throw new Error('Failed to clear override');

        const result = await response.json();
        cameraGrid.updateMedianCount(cameraId, result.median_count, false);

    } catch (error) {
        console.error('Error clearing override:', error);
    }
}
