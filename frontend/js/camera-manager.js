/**
 * Camera Manager
 * Handles camera CRUD operations and UI
 */

const CAMERA_API = 'http://localhost:8000/api';

// Cache for cameras and profiles
let camerasCache = [];
let profilesCache = {};

/**
 * Open camera manager modal
 */
async function openCameraManager() {
    const modal = document.getElementById('cameraManagerModal');
    modal.classList.add('show');

    // Load cameras and profiles
    await loadCameras();
    await loadProfiles();

    // Show camera list, hide form
    document.getElementById('cameraFormSection').style.display = 'none';
    document.querySelector('.camera-list-section').style.display = 'block';
}

/**
 * Close camera manager modal
 */
function closeCameraManager() {
    const modal = document.getElementById('cameraManagerModal');
    modal.classList.remove('show');
    hideCameraFormStatus();
}

/**
 * Load cameras from API
 */
async function loadCameras() {
    try {
        const response = await fetch(`${CAMERA_API}/cameras`);
        if (!response.ok) throw new Error('Failed to load cameras');

        camerasCache = await response.json();
        renderCameraList();

    } catch (error) {
        console.error('Error loading cameras:', error);
        showCameraFormStatus('Failed to load cameras', 'error');
    }
}

/**
 * Load available profiles
 */
async function loadProfiles() {
    try {
        const response = await fetch(`${CAMERA_API}/profiles`);
        if (!response.ok) throw new Error('Failed to load profiles');

        profilesCache = await response.json();
        updateProfileDropdown();

    } catch (error) {
        console.error('Error loading profiles:', error);
    }
}

/**
 * Update profile dropdown with available profiles
 */
function updateProfileDropdown() {
    const select = document.getElementById('cameraProfile');
    select.innerHTML = '';

    for (const [key, profile] of Object.entries(profilesCache)) {
        const option = document.createElement('option');
        option.value = key;
        option.textContent = `${key} (${profile.camera_type || 'general'})`;
        select.appendChild(option);
    }
}

/**
 * Render camera list
 */
function renderCameraList() {
    const container = document.getElementById('cameraList');

    if (camerasCache.length === 0) {
        container.innerHTML = '<p class="no-cameras">No cameras configured. Click "Add Camera" to get started.</p>';
        return;
    }

    container.innerHTML = camerasCache.map(camera => `
        <div class="camera-list-item ${camera.enabled ? '' : 'disabled'}">
            <div class="camera-item-info">
                <div class="camera-item-name">
                    <span class="camera-type-badge ${camera.type}">${camera.type.toUpperCase()}</span>
                    ${camera.name}
                </div>
                <div class="camera-item-details">
                    <span class="camera-id">${camera.id}</span>
                    <span class="camera-url">${camera.connection.rtsp_url}</span>
                </div>
            </div>
            <div class="camera-item-actions">
                <button class="btn btn-small btn-secondary" onclick="editCamera('${camera.id}')">Edit</button>
                <button class="btn btn-small ${camera.enabled ? 'btn-warning' : 'btn-success'}"
                        onclick="toggleCameraEnabled('${camera.id}', ${!camera.enabled})">
                    ${camera.enabled ? 'Disable' : 'Enable'}
                </button>
            </div>
        </div>
    `).join('');
}

/**
 * Show add camera form
 */
function showAddCameraForm() {
    document.getElementById('cameraFormTitle').textContent = 'Add New Camera';
    document.getElementById('editCameraId').value = '';
    document.getElementById('deleteCameraBtn').style.display = 'none';

    // Reset form
    resetCameraForm();

    // Show form, hide list
    document.getElementById('cameraFormSection').style.display = 'block';
    document.querySelector('.camera-list-section').style.display = 'none';
}

/**
 * Edit existing camera
 */
function editCamera(cameraId) {
    const camera = camerasCache.find(c => c.id === cameraId);
    if (!camera) return;

    document.getElementById('cameraFormTitle').textContent = 'Edit Camera';
    document.getElementById('editCameraId').value = cameraId;
    document.getElementById('deleteCameraBtn').style.display = 'inline-block';

    // Populate form
    document.getElementById('cameraName').value = camera.name;
    document.getElementById('cameraType').value = camera.type;
    document.getElementById('cameraEnabled').value = camera.enabled ? 'true' : 'false';
    document.getElementById('cameraProfile').value = camera.profile;

    // Connection settings
    const conn = camera.connection;
    const rtspUrl = conn.rtsp_url;

    // Determine connection type
    if (rtspUrl.endsWith('.mp4') || rtspUrl.endsWith('.avi') || rtspUrl.endsWith('.mkv')) {
        document.getElementById('connectionType').value = 'file';
        document.getElementById('videoFile').value = rtspUrl;
    } else if (rtspUrl.startsWith('http://') || rtspUrl.startsWith('https://')) {
        document.getElementById('connectionType').value = 'http';
        document.getElementById('rtspUrl').value = rtspUrl;
    } else {
        document.getElementById('connectionType').value = 'rtsp';
        document.getElementById('cameraIp').value = conn.ip;
        document.getElementById('rtspPort').value = conn.rtsp_port;
        document.getElementById('cameraUsername').value = conn.username;
        document.getElementById('cameraPassword').value = conn.password;

        // Extract path from URL and detect stream type
        const match = rtspUrl.match(/rtsp:\/\/[^\/]+(\/.*)/);
        if (match) {
            const path = match[1];
            document.getElementById('rtspPath').value = path;

            // Detect stream type from path
            if (path.includes('/Channels/101') || path === '/stream1') {
                document.getElementById('streamType').value = 'main';
                document.getElementById('rtspPath').readOnly = true;
            } else if (path.includes('/Channels/102')) {
                document.getElementById('streamType').value = 'sub';
                document.getElementById('rtspPath').readOnly = true;
            } else if (path.includes('/Channels/103')) {
                document.getElementById('streamType').value = 'third';
                document.getElementById('rtspPath').readOnly = true;
            } else {
                document.getElementById('streamType').value = 'custom';
                document.getElementById('rtspPath').readOnly = false;
            }
        }
    }

    document.getElementById('rtspUrl').value = rtspUrl;
    document.getElementById('cameraDescription').value = camera.position?.description || '';

    onConnectionTypeChange();

    // Show form, hide list
    document.getElementById('cameraFormSection').style.display = 'block';
    document.querySelector('.camera-list-section').style.display = 'none';
}

/**
 * Reset camera form
 */
function resetCameraForm() {
    document.getElementById('cameraName').value = '';
    document.getElementById('cameraType').value = 'rgb';
    document.getElementById('cameraEnabled').value = 'true';
    document.getElementById('cameraProfile').value = 'rgb_default';
    document.getElementById('connectionType').value = 'rtsp';
    document.getElementById('cameraIp').value = '';
    document.getElementById('rtspPort').value = '554';
    document.getElementById('cameraUsername').value = 'admin';
    document.getElementById('cameraPassword').value = '';
    document.getElementById('streamType').value = 'sub';
    document.getElementById('rtspPath').value = '/Streaming/Channels/102';
    document.getElementById('rtspPath').readOnly = true;
    document.getElementById('videoFile').value = '';
    document.getElementById('rtspUrl').value = '';
    document.getElementById('cameraDescription').value = '';

    onConnectionTypeChange();
    hideCameraFormStatus();
}

/**
 * Handle connection type change
 */
function onConnectionTypeChange() {
    const connType = document.getElementById('connectionType').value;
    const rtspFields = document.getElementById('rtspFields');
    const fileFields = document.getElementById('fileFields');

    if (connType === 'file') {
        rtspFields.style.display = 'none';
        fileFields.style.display = 'block';
    } else {
        rtspFields.style.display = 'block';
        fileFields.style.display = 'none';
    }

    updateRtspUrl();
}

/**
 * Handle stream type change (Hikvision stream paths)
 */
function onStreamTypeChange() {
    const streamType = document.getElementById('streamType').value;
    const rtspPathInput = document.getElementById('rtspPath');

    const streamPaths = {
        'main': '/Streaming/Channels/101',
        'sub': '/Streaming/Channels/102',
        'third': '/Streaming/Channels/103',
        'custom': rtspPathInput.value || '/stream1'
    };

    if (streamType !== 'custom') {
        rtspPathInput.value = streamPaths[streamType];
        rtspPathInput.readOnly = true;
    } else {
        rtspPathInput.readOnly = false;
    }

    updateRtspUrl();
}

/**
 * Handle camera type change
 */
function onCameraTypeChange() {
    const cameraType = document.getElementById('cameraType').value;
    const profileSelect = document.getElementById('cameraProfile');

    // Auto-select matching profile
    const profileMap = {
        'rgb': 'rgb_default',
        'thermal': 'thermal_default',
        'infrared': 'infrared_default'
    };

    if (profileMap[cameraType]) {
        profileSelect.value = profileMap[cameraType];
    }
}

/**
 * Update RTSP URL from form fields
 */
function updateRtspUrl() {
    const connType = document.getElementById('connectionType').value;

    if (connType === 'file') {
        const file = document.getElementById('videoFile').value;
        document.getElementById('rtspUrl').value = file;
        return;
    }

    const ip = document.getElementById('cameraIp').value;
    const port = document.getElementById('rtspPort').value;
    const username = document.getElementById('cameraUsername').value;
    const password = document.getElementById('cameraPassword').value;
    const path = document.getElementById('rtspPath').value;

    if (!ip) return;

    let url = '';
    if (connType === 'rtsp') {
        if (username && password) {
            url = `rtsp://${username}:${password}@${ip}:${port}${path}`;
        } else if (username) {
            url = `rtsp://${username}@${ip}:${port}${path}`;
        } else {
            url = `rtsp://${ip}:${port}${path}`;
        }
    } else if (connType === 'http') {
        url = `http://${ip}${path}`;
    }

    document.getElementById('rtspUrl').value = url;
}

/**
 * Cancel camera form
 */
function cancelCameraForm() {
    document.getElementById('cameraFormSection').style.display = 'none';
    document.querySelector('.camera-list-section').style.display = 'block';
    hideCameraFormStatus();
}

/**
 * Save camera (create or update)
 */
async function saveCamera() {
    const editId = document.getElementById('editCameraId').value;
    const isEdit = !!editId;

    // Gather form data
    const cameraData = {
        name: document.getElementById('cameraName').value,
        type: document.getElementById('cameraType').value,
        enabled: document.getElementById('cameraEnabled').value === 'true',
        profile: document.getElementById('cameraProfile').value,
        connection: {
            ip: document.getElementById('cameraIp').value || 'localhost',
            rtsp_port: parseInt(document.getElementById('rtspPort').value) || 554,
            http_port: 80,
            username: document.getElementById('cameraUsername').value || '',
            password: document.getElementById('cameraPassword').value || '',
            rtsp_url: document.getElementById('rtspUrl').value,
            isapi_base: ''
        },
        position: {
            description: document.getElementById('cameraDescription').value,
            floor: 1
        }
    };

    // Validation
    if (!cameraData.name) {
        showCameraFormStatus('Please enter a camera name', 'error');
        return;
    }

    if (!cameraData.connection.rtsp_url) {
        showCameraFormStatus('Please enter a stream URL or video file', 'error');
        return;
    }

    try {
        showCameraFormStatus('Saving camera...', 'info');

        const url = isEdit ? `${CAMERA_API}/cameras/${editId}` : `${CAMERA_API}/cameras`;
        const method = isEdit ? 'PUT' : 'POST';

        const response = await fetch(url, {
            method: method,
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(cameraData)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to save camera');
        }

        const result = await response.json();

        showCameraFormStatus(result.message, 'success');

        // Reload cameras list
        await loadCameras();

        // Go back to list after short delay
        setTimeout(() => {
            cancelCameraForm();
        }, 1500);

    } catch (error) {
        console.error('Error saving camera:', error);
        showCameraFormStatus('Error: ' + error.message, 'error');
    }
}

/**
 * Delete camera
 */
async function deleteCamera() {
    const editId = document.getElementById('editCameraId').value;
    if (!editId) return;

    const camera = camerasCache.find(c => c.id === editId);
    if (!confirm(`Delete camera "${camera?.name}"? This cannot be undone.`)) {
        return;
    }

    try {
        showCameraFormStatus('Deleting camera...', 'info');

        const response = await fetch(`${CAMERA_API}/cameras/${editId}`, {
            method: 'DELETE'
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to delete camera');
        }

        const result = await response.json();

        showCameraFormStatus(result.message, 'success');

        // Reload cameras list
        await loadCameras();

        // Go back to list after short delay
        setTimeout(() => {
            cancelCameraForm();
        }, 1500);

    } catch (error) {
        console.error('Error deleting camera:', error);
        showCameraFormStatus('Error: ' + error.message, 'error');
    }
}

/**
 * Toggle camera enabled/disabled
 */
async function toggleCameraEnabled(cameraId, enabled) {
    try {
        const response = await fetch(`${CAMERA_API}/cameras/${cameraId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ enabled: enabled })
        });

        if (!response.ok) throw new Error('Failed to update camera');

        // Reload cameras
        await loadCameras();

    } catch (error) {
        console.error('Error toggling camera:', error);
        showCameraFormStatus('Error: ' + error.message, 'error');
    }
}

/**
 * Show status message
 */
function showCameraFormStatus(message, type) {
    const status = document.getElementById('cameraFormStatus');
    status.textContent = message;
    status.className = 'camera-form-status ' + type;
    status.style.display = 'block';
}

/**
 * Hide status message
 */
function hideCameraFormStatus() {
    const status = document.getElementById('cameraFormStatus');
    status.style.display = 'none';
}

// Add event listeners for auto-updating RTSP URL
document.addEventListener('DOMContentLoaded', () => {
    const fields = ['cameraIp', 'rtspPort', 'cameraUsername', 'cameraPassword', 'rtspPath', 'videoFile'];

    fields.forEach(fieldId => {
        const field = document.getElementById(fieldId);
        if (field) {
            field.addEventListener('input', updateRtspUrl);
        }
    });

    // Close modal on outside click
    const modal = document.getElementById('cameraManagerModal');
    if (modal) {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                closeCameraManager();
            }
        });
    }
});
