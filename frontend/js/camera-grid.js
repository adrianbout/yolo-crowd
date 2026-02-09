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
        // Manual seat marking
        this.manualMarkers = {};  // { cameraId: [{x, y, id}, ...] }
        this.markingModeEnabled = {};  // { cameraId: boolean }
        this.markerIdCounter = 0;
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

        // Default show_boxes to true if not set
        const showBoxes = camera.show_boxes !== false;

        card.innerHTML = `
            <div class="camera-header">
                <div class="camera-name">${camera.name}</div>
                <div class="camera-status">
                    <label class="toggle-switch show-boxes-toggle" title="Show detection boxes">
                        <input type="checkbox" id="show-boxes-${camera.id}" ${showBoxes ? 'checked' : ''} onchange="toggleShowBoxes('${camera.id}')">
                        <span class="toggle-slider"></span>
                    </label>
                    <span class="camera-status-dot" id="status-${camera.id}"></span>
                    <span id="status-text-${camera.id}">Connecting...</span>
                    <span class="camera-resolution" id="resolution-${camera.id}"></span>
                </div>
            </div>
            <div class="camera-feed" id="feed-${camera.id}">
                <span>Loading camera feed...</span>
                <canvas class="manual-marker-canvas" id="marker-canvas-${camera.id}"></canvas>
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
                <div class="manual-marking-section">
                    <div class="manual-marking-header">
                        <span class="count-label">Manual Seat Count:</span>
                        <span class="manual-count-value" id="manual-count-${camera.id}">0</span>
                    </div>
                    <div class="manual-marking-controls">
                        <button class="btn btn-small marking-toggle" id="marking-toggle-${camera.id}" onclick="toggleMarkingMode('${camera.id}')">
                            Mark Seats
                        </button>
                        <button class="btn btn-small btn-secondary" id="clear-markers-${camera.id}" onclick="clearManualMarkers('${camera.id}')" style="display: none;">
                            Clear All
                        </button>
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
                // Don't clear innerHTML - it will remove the canvas!
                // Just remove the loading text if it exists
                const loadingText = feedElement.querySelector('span');
                if (loadingText) {
                    loadingText.remove();
                }
                
                img = document.createElement('img');
                feedElement.insertBefore(img, feedElement.firstChild); // Insert before canvas
            }

            // Update image source with timestamp to avoid caching
            img.src = `${API_BASE}/cameras/${cameraId}/frame?draw_rois=true&t=${Date.now()}`;

            // Setup marker canvas overlay (resizes and redraws as needed)
            this.setupMarkerCanvas(cameraId);

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

    // Setup canvas overlay for a camera
    setupMarkerCanvas(cameraId) {
        const feedElement = document.getElementById(`feed-${cameraId}`);
        const canvas = document.getElementById(`marker-canvas-${cameraId}`);
        if (!feedElement || !canvas) {
            console.warn(`Canvas setup failed for camera ${cameraId}: feedElement=${!!feedElement}, canvas=${!!canvas}`);
            return;
        }

        const img = feedElement.querySelector('img');
        if (!img) {
            console.log(`No image yet for camera ${cameraId}, will retry`);
            return;
        }

        // Match canvas size to feed container
        const rect = feedElement.getBoundingClientRect();
        
        // Ensure canvas has actual dimensions
        if (rect.width > 0 && rect.height > 0) {
            if (canvas.width !== rect.width || canvas.height !== rect.height) {
                canvas.width = rect.width;
                canvas.height = rect.height;
                // Also set CSS dimensions to ensure visibility
                canvas.style.width = rect.width + 'px';
                canvas.style.height = rect.height + 'px';
                console.log(`Canvas ${cameraId} resized to ${rect.width}x${rect.height}`);
            }
        } else {
            console.warn(`Canvas ${cameraId} has zero dimensions: ${rect.width}x${rect.height}`);
        }

        // Initialize markers array if not exists
        if (!this.manualMarkers[cameraId]) {
            this.manualMarkers[cameraId] = [];
        }
        if (this.markingModeEnabled[cameraId] === undefined) {
            this.markingModeEnabled[cameraId] = false;
        }

        // Add click handler if not already added
        if (!canvas.dataset.clickHandlerAdded) {
            canvas.addEventListener('click', (e) => this.handleCanvasClick(cameraId, e));
            canvas.dataset.clickHandlerAdded = 'true';
            console.log(`Click handler added for camera ${cameraId}`);
        }

        // Redraw markers
        this.drawMarkers(cameraId);
    }

    // Handle click on canvas
    handleCanvasClick(cameraId, event) {
        if (!this.markingModeEnabled[cameraId]) {
            console.log(`Click ignored for camera ${cameraId} - marking mode not enabled`);
            return;
        }

        const canvas = document.getElementById(`marker-canvas-${cameraId}`);
        if (!canvas) return;

        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        console.log(`Canvas click at (${x.toFixed(0)}, ${y.toFixed(0)}) for camera ${cameraId}`);

        // Check if clicking on existing marker (within 15px radius)
        const markers = this.manualMarkers[cameraId] || [];
        const clickRadius = 15;
        const existingMarkerIndex = markers.findIndex(marker => {
            const distance = Math.sqrt(Math.pow(marker.x - x, 2) + Math.pow(marker.y - y, 2));
            return distance <= clickRadius;
        });

        if (existingMarkerIndex !== -1) {
            // Remove existing marker
            markers.splice(existingMarkerIndex, 1);
            console.log(`Removed marker #${existingMarkerIndex + 1} from camera ${cameraId}. Total: ${markers.length}`);
        } else {
            // Add new marker
            markers.push({
                x: x,
                y: y,
                id: ++this.markerIdCounter,
                // Store normalized coordinates for persistence across resize
                normalizedX: x / canvas.width,
                normalizedY: y / canvas.height
            });
            console.log(`Added marker #${markers.length} to camera ${cameraId} at (${x.toFixed(0)}, ${y.toFixed(0)})`);
        }

        this.manualMarkers[cameraId] = markers;
        this.drawMarkers(cameraId);
        this.updateManualCount(cameraId);
    }

    // Draw markers on canvas
    drawMarkers(cameraId) {
        const canvas = document.getElementById(`marker-canvas-${cameraId}`);
        if (!canvas) {
            console.warn(`Cannot draw markers - canvas not found for ${cameraId}`);
            return;
        }

        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw test indicator when marking mode is active
        if (this.markingModeEnabled[cameraId]) {
            // Draw corner indicators (smaller)
            ctx.fillStyle = 'rgba(255, 193, 7, 0.6)';
            ctx.fillRect(5, 5, 10, 10); // Top-left
            ctx.fillRect(canvas.width - 15, 5, 10, 10); // Top-right
            ctx.fillRect(5, canvas.height - 15, 10, 10); // Bottom-left
            ctx.fillRect(canvas.width - 15, canvas.height - 15, 10, 10); // Bottom-right
        }

        const markers = this.manualMarkers[cameraId] || [];
        console.log(`Drawing ${markers.length} markers for camera ${cameraId}`);

        markers.forEach((marker, index) => {
            // Recalculate position from normalized coordinates
            const x = marker.normalizedX ? marker.normalizedX * canvas.width : marker.x;
            const y = marker.normalizedY ? marker.normalizedY * canvas.height : marker.y;

            // Draw small outer glow
            ctx.beginPath();
            ctx.arc(x, y, 7, 0, 2 * Math.PI);
            ctx.fillStyle = 'rgba(255, 193, 7, 0.3)';
            ctx.fill();

            // Draw main circle (very small)
            ctx.beginPath();
            ctx.arc(x, y, 5, 0, 2 * Math.PI);
            ctx.fillStyle = 'rgba(255, 193, 7, 0.9)';
            ctx.fill();
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 1.5;
            ctx.stroke();

            // Draw number inside
            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 7px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText((index + 1).toString(), x, y);
            
            console.log(`Drew marker #${index + 1} at (${x.toFixed(0)}, ${y.toFixed(0)})`);
        });
    }

    // Update manual count display
    updateManualCount(cameraId) {
        const countElement = document.getElementById(`manual-count-${cameraId}`);
        const clearButton = document.getElementById(`clear-markers-${cameraId}`);
        const markers = this.manualMarkers[cameraId] || [];

        if (countElement) {
            countElement.textContent = markers.length;
        }
        if (clearButton) {
            clearButton.style.display = markers.length > 0 ? 'inline-block' : 'none';
        }

        // Update adjusted empty chairs based on manual markers
        this.updateAdjustedEmptyChairsFromMarkers(cameraId, markers.length);
    }

    // Update adjusted empty chairs based on manual marker count
    async updateAdjustedEmptyChairsFromMarkers(cameraId, markerCount) {
        try {
            // Send the manual marker count as an override to the API
            const response = await fetch(`${API_BASE}/counting/${cameraId}/override`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ count: markerCount })
            });

            if (!response.ok) throw new Error('Failed to update adjusted empty chairs');

            // Update UI immediately
            this.updateAdjustedEmptyChairs(cameraId, markerCount, true);

        } catch (error) {
            console.error('Error updating adjusted empty chairs from markers:', error);
        }
    }

    // Toggle marking mode
    toggleMarkingMode(cameraId) {
        this.markingModeEnabled[cameraId] = !this.markingModeEnabled[cameraId];

        const button = document.getElementById(`marking-toggle-${cameraId}`);
        const canvas = document.getElementById(`marker-canvas-${cameraId}`);

        console.log(`Marking mode for ${cameraId}: ${this.markingModeEnabled[cameraId]}`);

        if (button) {
            if (this.markingModeEnabled[cameraId]) {
                button.textContent = 'Done Marking';
                button.classList.add('btn-warning');
                button.classList.remove('marking-toggle');
            } else {
                button.textContent = 'Mark Seats';
                button.classList.remove('btn-warning');
                button.classList.add('marking-toggle');
            }
        }

        if (canvas) {
            canvas.classList.toggle('marking-active', this.markingModeEnabled[cameraId]);
            console.log(`Canvas ${cameraId} marking-active class: ${canvas.classList.contains('marking-active')}`);
            console.log(`Canvas ${cameraId} dimensions: ${canvas.width}x${canvas.height}`);
            console.log(`Canvas ${cameraId} z-index: ${window.getComputedStyle(canvas).zIndex}`);
            
            // Draw a test marker to verify canvas is working
            if (this.markingModeEnabled[cameraId]) {
                const ctx = canvas.getContext('2d');
                ctx.fillStyle = 'rgba(255, 0, 0, 0.8)';
                ctx.fillRect(10, 10, 50, 50);
                console.log(`Drew test red square on canvas ${cameraId}`);
            }
        }
    }

    // Clear all markers for a camera
    clearManualMarkers(cameraId) {
        this.manualMarkers[cameraId] = [];
        this.drawMarkers(cameraId);
        this.updateManualCount(cameraId);
        // This will set adjusted empty chairs to 0 via the API
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

// Toggle show boxes for a camera
async function toggleShowBoxes(cameraId) {
    try {
        const response = await fetch(`${API_BASE}/cameras/${cameraId}/show-boxes`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        if (!response.ok) throw new Error('Failed to toggle show boxes');

        const result = await response.json();
        console.log(`Show boxes for ${cameraId}: ${result.show_boxes}`);

    } catch (error) {
        console.error('Error toggling show boxes:', error);
        // Revert checkbox state on error
        const checkbox = document.getElementById(`show-boxes-${cameraId}`);
        if (checkbox) {
            checkbox.checked = !checkbox.checked;
        }
    }
}

// Toggle manual marking mode for a camera
function toggleMarkingMode(cameraId) {
    if (cameraGrid) {
        cameraGrid.toggleMarkingMode(cameraId);
    }
}

// Clear all manual markers for a camera
function clearManualMarkers(cameraId) {
    if (cameraGrid) {
        cameraGrid.clearManualMarkers(cameraId);
    }
}
