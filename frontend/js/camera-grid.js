/**
 * Camera Grid Management
 * Handles display and updates for camera feeds
 */

const API_BASE = `${window.location.origin}/api`;

class CameraGrid {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.cameras = [];
        this.counts = {};
        this.medianCounts = {};
        this.adjustedEmptyChairs = {};
        this.hasOverride = {};
    }

    async loadCameras() {
        try {
            const response = await fetch(`${API_BASE}/cameras`);
            if (!response.ok) throw new Error('Failed to fetch cameras');

            this.cameras = await response.json();
            console.log('Camera grid loaded cameras:', this.cameras.map(c => ({ id: c.id, name: c.name, totalChairs: c.totalChairs })));
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
                    <span class="camera-resolution" id="resolution-${camera.id}"></span>
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
                <div class="camera-empty-chairs">
                    <span class="count-label">Empty Chairs:</span>
                    <span class="empty-chairs-value" id="empty-chairs-${camera.id}">--</span>
                </div>
                <div class="camera-median-count">
                    <span class="count-label">Adjusted Empty Chairs:</span>
                    <div class="median-controls">
                        <button class="btn-adjust btn-minus" onclick="adjustEmptyChairs('${camera.id}', -1)">-</button>
                        <span class="median-value" id="adjusted-empty-${camera.id}">0</span>
                        <button class="btn-adjust btn-plus" onclick="adjustEmptyChairs('${camera.id}', 1)">+</button>
                        <button class="btn-reset-override" id="reset-override-${camera.id}" onclick="clearOverride('${camera.id}')" style="display: none;" title="Reset to calculated value">&#x21ba;</button>
                    </div>
                </div>
                <div class="camera-actions">
                    <button class="btn btn-primary btn-small" onclick="openROIEditor('${camera.id}', '${camera.name}')">
                        Edit ROI
                    </button>
                    <button class="btn btn-secondary btn-small" onclick="editCameraFromGrid('${camera.id}')">
                        Edit
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

    updateCameraStatus(cameraId, connected, resolution = null) {
        const statusDot = document.getElementById(`status-${cameraId}`);
        const statusText = document.getElementById(`status-text-${cameraId}`);
        const resolutionText = document.getElementById(`resolution-${cameraId}`);

        if (statusDot) {
            statusDot.className = `camera-status-dot ${connected ? 'connected' : ''}`;
        }

        if (statusText) {
            statusText.textContent = connected ? 'Connected' : 'Disconnected';
        }

        if (resolutionText && resolution && resolution.width > 0) {
            resolutionText.textContent = `${resolution.width}x${resolution.height}`;
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

    updateAdjustedEmptyChairs(cameraId, adjustedEmptyChairs, hasOverride) {
        const adjustedElement = document.getElementById(`adjusted-empty-${cameraId}`);
        if (adjustedElement) {
            adjustedElement.textContent = adjustedEmptyChairs;
            // Highlight if manual override is active
            if (hasOverride) {
                adjustedElement.classList.add('has-override');
            } else {
                adjustedElement.classList.remove('has-override');
            }
        }
        this.adjustedEmptyChairs[cameraId] = adjustedEmptyChairs;
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
            // Update empty chairs (calculated from YOLO)
            if (data.empty_chairs !== undefined && data.empty_chairs !== null) {
                this.updateEmptyChairsValue(cameraId, data.empty_chairs, data.total_chairs || 0, data.yolo_median || 0);
            }
            // Update adjusted empty chairs (manual override or calculated)
            if (data.adjusted_empty_chairs !== undefined) {
                this.updateAdjustedEmptyChairs(cameraId, data.adjusted_empty_chairs, data.has_override || false);
            }
        });
    }

    updateEmptyChairsValue(cameraId, emptyChairs, totalChairs, detectedPeople) {
        const emptyChairsElement = document.getElementById(`empty-chairs-${cameraId}`);
        if (!emptyChairsElement) return;

        if (totalChairs === 0) {
            emptyChairsElement.textContent = '--';
            emptyChairsElement.title = 'Total chairs not configured. Click Edit to set.';
            emptyChairsElement.style.color = '#999';
        } else {
            emptyChairsElement.textContent = emptyChairs;
            emptyChairsElement.title = `${totalChairs} total chairs - ${detectedPeople} detected people = ${emptyChairs} empty chairs`;
            emptyChairsElement.style.color = emptyChairs === 0 ? '#ff6b6b' : '#4CAF50';
        }
    }

    updateYoloMedian(cameraId, yoloMedian) {
        const yoloMedianElement = document.getElementById(`yolo-median-${cameraId}`);
        if (yoloMedianElement) {
            yoloMedianElement.textContent = yoloMedian;
        }
    }

    updateEmptyChairs(cameraId, detectedPeople) {
        const emptyChairsElement = document.getElementById(`empty-chairs-${cameraId}`);
        if (!emptyChairsElement) return;

        // Find the camera configuration to get totalChairs
        const camera = this.cameras.find(c => c.id === cameraId);
        if (!camera) {
            emptyChairsElement.textContent = '--';
            return;
        }

        const totalChairs = camera.totalChairs || 0;
        const detected = detectedPeople !== undefined ? detectedPeople : 0;
        
        if (totalChairs === 0) {
            emptyChairsElement.textContent = '--';
            emptyChairsElement.title = 'Total chairs not configured. Click Edit to set.';
            emptyChairsElement.style.color = '#999';
        } else {
            const emptyChairs = Math.max(0, totalChairs - detected);
            emptyChairsElement.textContent = emptyChairs;
            emptyChairsElement.title = `${totalChairs} total chairs - ${detected} detected people = ${emptyChairs} empty chairs`;
            emptyChairsElement.style.color = emptyChairs === 0 ? '#ff6b6b' : '#4CAF50';
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

        const resolution = status.resolution ? `${status.resolution.width}x${status.resolution.height}` : 'N/A';
        alert(`Camera Details:\n\nName: ${camera.name}\nType: ${camera.type}\nStatus: ${status.connected ? 'Connected' : 'Disconnected'}\nResolution: ${resolution}\nFPS: ${status.fps.toFixed(2)}`);
    } catch (error) {
        console.error('Error fetching camera details:', error);
        alert('Failed to fetch camera details');
    }
}

// Adjust empty chairs manually
async function adjustEmptyChairs(cameraId, delta) {
    try {
        // Get current adjusted empty chairs count
        const currentCount = cameraGrid.adjustedEmptyChairs[cameraId] || 0;
        const newCount = Math.max(0, currentCount + delta);

        // Send override to API
        const response = await fetch(`${API_BASE}/counting/${cameraId}/override`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ count: newCount })
        });

        if (!response.ok) throw new Error('Failed to update empty chairs count');

        // Update UI immediately
        cameraGrid.updateAdjustedEmptyChairs(cameraId, newCount, true);

    } catch (error) {
        console.error('Error adjusting empty chairs:', error);
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
        // Update with the calculated empty chairs value
        cameraGrid.updateAdjustedEmptyChairs(cameraId, result.empty_chairs || 0, false);

    } catch (error) {
        console.error('Error clearing override:', error);
    }
}

// Edit camera directly from grid (opens camera manager in edit mode)
async function editCameraFromGrid(cameraId) {
    // Open camera manager modal
    const modal = document.getElementById('cameraManagerModal');
    modal.classList.add('show');

    // Load cameras and profiles first
    await loadCameras();
    await loadProfiles();

    // Hide list, directly go to edit form
    document.querySelector('.camera-list-section').style.display = 'none';

    // Call editCamera to populate the form
    editCamera(cameraId);
}
