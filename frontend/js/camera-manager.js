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
        console.log('Loaded cameras into cache:', camerasCache.map(c => ({ id: c.id, name: c.name, totalChairs: c.totalChairs })));
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
                    ${camera.detection_settings ? '<span class="camera-custom-badge" title="Custom detection settings">Custom</span>' : ''}
                </div>
                <div class="camera-item-details">
                    <span class="camera-id">${camera.id}</span>
                    <span class="camera-url">${camera.connection.rtsp_url}</span>
                    ${camera.detection_settings ? `<span class="camera-settings-summary">model: ${camera.detection_settings.detection_model || 'rgb'}, conf: ${camera.detection_settings.confidence_threshold}, img: ${camera.detection_settings.img_size}</span>` : ''}
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

    console.log(`Editing camera ${cameraId}:`, camera);
    console.log(`Camera ${cameraId} totalChairs:`, camera.totalChairs);

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
    
    // Set totalChairs - ensure we're getting the right value
    const totalChairsValue = camera.totalChairs !== undefined ? camera.totalChairs : 0;
    document.getElementById('cameraTotalChairs').value = totalChairsValue;
    console.log(`Set cameraTotalChairs input to: ${totalChairsValue}`);

    // Load detection settings if they exist
    if (camera.detection_settings) {
        document.getElementById('cameraUseCustomSettings').checked = true;
        document.getElementById('cameraDetectionSettings').style.display = 'block';
        document.getElementById('cameraDetectionModel').value = camera.detection_settings.detection_model || 'rgb';
        document.getElementById('cameraConfidence').value = camera.detection_settings.confidence_threshold || 0.25;
        document.getElementById('cameraConfidenceValue').textContent = camera.detection_settings.confidence_threshold || 0.25;
        document.getElementById('cameraIou').value = camera.detection_settings.iou_threshold || 0.45;
        document.getElementById('cameraIouValue').textContent = camera.detection_settings.iou_threshold || 0.45;
        document.getElementById('cameraImgSize').value = camera.detection_settings.img_size || 640;
        document.getElementById('cameraPreprocessing').value = camera.detection_settings.preprocessing || 'none';
        // Blob hotspot settings
        document.getElementById('blobThreshold').value = camera.detection_settings.blob_threshold || 200;
        document.getElementById('blobThresholdValue').textContent = camera.detection_settings.blob_threshold || 200;
        document.getElementById('blobMinArea').value = camera.detection_settings.blob_min_area || 2000;
        document.getElementById('blobMinAreaValue').textContent = camera.detection_settings.blob_min_area || 2000;
        document.getElementById('blobMaxArea').value = camera.detection_settings.blob_max_area || 50000;
        document.getElementById('blobMaxAreaValue').textContent = camera.detection_settings.blob_max_area || 50000;
        document.getElementById('blobAspectMin').value = camera.detection_settings.blob_aspect_ratio_min || 0.5;
        document.getElementById('blobAspectMax').value = camera.detection_settings.blob_aspect_ratio_max || 3.0;
        // Show/hide blob settings based on model
        onDetectionModelChange();
    } else {
        document.getElementById('cameraUseCustomSettings').checked = false;
        document.getElementById('cameraDetectionSettings').style.display = 'none';
    }

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
    document.getElementById('cameraTotalChairs').value = 0;

    // Reset detection settings
    document.getElementById('cameraUseCustomSettings').checked = false;
    document.getElementById('cameraDetectionSettings').style.display = 'none';
    document.getElementById('cameraDetectionModel').value = 'rgb';
    document.getElementById('cameraConfidence').value = 0.25;
    document.getElementById('cameraConfidenceValue').textContent = '0.25';
    document.getElementById('cameraIou').value = 0.45;
    document.getElementById('cameraIouValue').textContent = '0.45';
    document.getElementById('cameraImgSize').value = '640';
    document.getElementById('cameraPreprocessing').value = 'none';
    // Reset blob hotspot settings
    document.getElementById('blobThreshold').value = 200;
    document.getElementById('blobThresholdValue').textContent = '200';
    document.getElementById('blobMinArea').value = 2000;
    document.getElementById('blobMinAreaValue').textContent = '2000';
    document.getElementById('blobMaxArea').value = 50000;
    document.getElementById('blobMaxAreaValue').textContent = '50000';
    document.getElementById('blobAspectMin').value = 0.5;
    document.getElementById('blobAspectMax').value = 3.0;
    document.getElementById('blobHotspotSettings').style.display = 'none';

    onConnectionTypeChange();
    hideCameraFormStatus();
}

/**
 * Toggle custom detection settings visibility
 */
function onCustomSettingsToggle() {
    const useCustom = document.getElementById('cameraUseCustomSettings').checked;
    document.getElementById('cameraDetectionSettings').style.display = useCustom ? 'block' : 'none';
    if (useCustom) {
        onDetectionModelChange();
    }
}

/**
 * Show/hide blob hotspot settings based on detection model
 */
function onDetectionModelChange() {
    const model = document.getElementById('cameraDetectionModel').value;
    const blobSettings = document.getElementById('blobHotspotSettings');
    if (blobSettings) {
        blobSettings.style.display = model === 'blob_hotspot' ? 'block' : 'none';
    }
}

/**
 * Update camera setting value display
 */
function updateCameraSettingValue(inputId) {
    const input = document.getElementById(inputId);
    const valueSpan = document.getElementById(inputId + 'Value');
    if (input && valueSpan) {
        valueSpan.textContent = input.value;
    }
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

    // Gather form data - UPDATED VERSION
    const totalChairsInput = document.getElementById('cameraTotalChairs');
    
    console.warn('=== TOTALCHAIRS DEBUG START ===');
    console.warn('Input element exists:', !!totalChairsInput);
    console.warn('Input value:', totalChairsInput?.value);
    console.warn('Input value type:', typeof totalChairsInput?.value);
    
    let totalChairsFinal = 0;
    
    if (totalChairsInput && totalChairsInput.value !== null && totalChairsInput.value !== undefined && totalChairsInput.value !== '') {
        const parsed = parseInt(totalChairsInput.value, 10);
        console.warn('Parsed value:', parsed);
        console.warn('Is NaN:', isNaN(parsed));
        if (!isNaN(parsed) && parsed >= 0) {
            totalChairsFinal = parsed;
        }
    }
    
    console.warn('Final totalChairs value:', totalChairsFinal);
    console.warn('=== TOTALCHAIRS DEBUG END ===');
    
    // Alert for visibility
    const alertMsg = `TotalChairs Debug:\nInput value: "${totalChairsInput?.value}"\nFinal value: ${totalChairsFinal}`;
    alert(alertMsg);

    const cameraData = {
        name: document.getElementById('cameraName').value,
        type: document.getElementById('cameraType').value,
        enabled: document.getElementById('cameraEnabled').value === 'true',
        profile: document.getElementById('cameraProfile').value,
        connection: {
            ip: document.getElementById('cameraIp').value || 'localhost',
            rtsp_port: parseInt(document.getElementById('rtspPort').value, 10) || 554,
            http_port: 80,
            username: document.getElementById('cameraUsername').value || '',
            password: document.getElementById('cameraPassword').value || '',
            rtsp_url: document.getElementById('rtspUrl').value,
            isapi_base: ''
        },
        position: {
            description: document.getElementById('cameraDescription').value,
            floor: 1
        },
        totalChairs: totalChairsFinal
    };

    // Debug: Log the totalChairs value being saved
    console.log(`Saving camera ${editId || 'NEW'} with totalChairs: ${cameraData.totalChairs}`);
    console.log('Full camera data being sent:', JSON.stringify(cameraData, null, 2));

    // Add detection settings if custom settings are enabled
    if (document.getElementById('cameraUseCustomSettings').checked) {
        cameraData.detection_settings = {
            detection_model: document.getElementById('cameraDetectionModel').value,
            confidence_threshold: parseFloat(document.getElementById('cameraConfidence').value),
            iou_threshold: parseFloat(document.getElementById('cameraIou').value),
            img_size: parseInt(document.getElementById('cameraImgSize').value),
            preprocessing: document.getElementById('cameraPreprocessing').value,
            // Blob hotspot settings
            blob_threshold: parseInt(document.getElementById('blobThreshold').value),
            blob_min_area: parseInt(document.getElementById('blobMinArea').value),
            blob_max_area: parseInt(document.getElementById('blobMaxArea').value),
            blob_aspect_ratio_min: parseFloat(document.getElementById('blobAspectMin').value),
            blob_aspect_ratio_max: parseFloat(document.getElementById('blobAspectMax').value)
        };
    } else {
        cameraData.detection_settings = null;
    }

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

        // Reload main camera grid to reflect changes
        if (window.cameraGrid && typeof window.cameraGrid.loadCameras === 'function') {
            await window.cameraGrid.loadCameras();
        }

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

        // Reload main camera grid to reflect changes
        if (window.cameraGrid) {
            await window.cameraGrid.loadCameras();
        }

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
