/**
 * Settings Manager
 * Handles detection settings UI and API communication
 */

const SETTINGS_API = 'http://localhost:8000/api';

// Default settings
const DEFAULT_SETTINGS = {
    img_size: 608,
    confidence_threshold: 0.25,
    iou_threshold: 0.45,
    half_precision: true,
    denoise: false,
    clahe: false,
    equalize_histogram: false,
    inference_interval: 0.1
};

// Current settings cache
let currentSettings = { ...DEFAULT_SETTINGS };

/**
 * Open settings modal and load current settings
 */
async function openSettings() {
    const modal = document.getElementById('settingsModal');
    modal.classList.add('show');

    // Load current settings from API
    await loadSettings();
}

/**
 * Close settings modal
 */
function closeSettings() {
    const modal = document.getElementById('settingsModal');
    modal.classList.remove('show');
    hideSettingsStatus();
}

/**
 * Load settings from API
 */
async function loadSettings() {
    try {
        const response = await fetch(`${SETTINGS_API}/settings`);
        if (!response.ok) throw new Error('Failed to load settings');

        const data = await response.json();

        // Merge detection settings with preprocessing defaults
        currentSettings = {
            img_size: data.detection_settings.img_size || DEFAULT_SETTINGS.img_size,
            confidence_threshold: data.detection_settings.confidence_threshold || DEFAULT_SETTINGS.confidence_threshold,
            iou_threshold: data.detection_settings.iou_threshold || DEFAULT_SETTINGS.iou_threshold,
            half_precision: data.detection_settings.half_precision ?? DEFAULT_SETTINGS.half_precision,
            denoise: data.preprocessing_defaults.denoise || false,
            clahe: data.preprocessing_defaults.clahe || false,
            equalize_histogram: data.preprocessing_defaults.equalize_histogram || false,
            inference_interval: data.detection_settings.inference_interval || DEFAULT_SETTINGS.inference_interval
        };

        updateUIFromSettings();
        console.log('Settings loaded:', currentSettings);

    } catch (error) {
        console.error('Error loading settings:', error);
        showSettingsStatus('Failed to load settings', 'error');
    }
}

/**
 * Update UI controls from current settings
 */
function updateUIFromSettings() {
    // Image size (dropdown)
    const imgSizeInput = document.getElementById('imgSize');
    imgSizeInput.value = currentSettings.img_size;

    // Inference interval
    const inferenceInput = document.getElementById('inferenceInterval');
    const inferenceValue = document.getElementById('inferenceIntervalValue');
    inferenceInput.value = currentSettings.inference_interval;
    inferenceValue.textContent = `${currentSettings.inference_interval}s`;

    // Half precision
    const halfPrecisionInput = document.getElementById('halfPrecision');
    const halfPrecisionValue = document.getElementById('halfPrecisionValue');
    halfPrecisionInput.checked = currentSettings.half_precision;
    halfPrecisionValue.textContent = currentSettings.half_precision ? 'Enabled' : 'Disabled';

    // Confidence threshold
    const confInput = document.getElementById('confidenceThreshold');
    const confValue = document.getElementById('confidenceThresholdValue');
    confInput.value = currentSettings.confidence_threshold;
    confValue.textContent = currentSettings.confidence_threshold.toFixed(2);

    // IOU threshold
    const iouInput = document.getElementById('iouThreshold');
    const iouValue = document.getElementById('iouThresholdValue');
    iouInput.value = currentSettings.iou_threshold;
    iouValue.textContent = currentSettings.iou_threshold.toFixed(2);

    // Denoise
    const denoiseInput = document.getElementById('denoise');
    const denoiseValue = document.getElementById('denoiseValue');
    denoiseInput.checked = currentSettings.denoise;
    denoiseValue.textContent = currentSettings.denoise ? 'Enabled' : 'Disabled';
    if (currentSettings.denoise) {
        denoiseValue.classList.add('setting-warning');
    } else {
        denoiseValue.classList.remove('setting-warning');
    }

    // CLAHE
    const claheInput = document.getElementById('clahe');
    const claheValue = document.getElementById('claheValue');
    claheInput.checked = currentSettings.clahe;
    claheValue.textContent = currentSettings.clahe ? 'Enabled' : 'Disabled';

    // Histogram equalization
    const eqInput = document.getElementById('equalizeHistogram');
    const eqValue = document.getElementById('equalizeHistogramValue');
    eqInput.checked = currentSettings.equalize_histogram;
    eqValue.textContent = currentSettings.equalize_histogram ? 'Enabled' : 'Disabled';
}

/**
 * Get settings from UI controls
 */
function getSettingsFromUI() {
    return {
        img_size: parseInt(document.getElementById('imgSize').value),
        confidence_threshold: parseFloat(document.getElementById('confidenceThreshold').value),
        iou_threshold: parseFloat(document.getElementById('iouThreshold').value),
        half_precision: document.getElementById('halfPrecision').checked,
        denoise: document.getElementById('denoise').checked,
        clahe: document.getElementById('clahe').checked,
        equalize_histogram: document.getElementById('equalizeHistogram').checked,
        inference_interval: parseFloat(document.getElementById('inferenceInterval').value)
    };
}

/**
 * Apply settings live (without restart)
 */
async function applySettings() {
    const settings = getSettingsFromUI();

    try {
        showSettingsStatus('Applying settings...', 'info');

        const response = await fetch(`${SETTINGS_API}/settings/apply`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(settings)
        });

        if (!response.ok) throw new Error('Failed to apply settings');

        const result = await response.json();
        currentSettings = settings;

        showSettingsStatus(
            `Settings applied! Live: ${result.applied_live.join(', ')}. Restart needed for: ${result.requires_restart.join(', ')}`,
            'success'
        );

        console.log('Settings applied:', result);

    } catch (error) {
        console.error('Error applying settings:', error);
        showSettingsStatus('Failed to apply settings: ' + error.message, 'error');
    }
}

/**
 * Save settings (may require restart)
 */
async function saveSettings() {
    const settings = getSettingsFromUI();

    try {
        showSettingsStatus('Saving settings...', 'info');

        const response = await fetch(`${SETTINGS_API}/settings`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(settings)
        });

        if (!response.ok) throw new Error('Failed to save settings');

        const result = await response.json();
        currentSettings = settings;

        showSettingsStatus(
            'Settings saved! Restart the server for full effect.',
            'success'
        );

        console.log('Settings saved:', result);

    } catch (error) {
        console.error('Error saving settings:', error);
        showSettingsStatus('Failed to save settings: ' + error.message, 'error');
    }
}

/**
 * Reset settings to defaults
 */
function resetSettings() {
    if (!confirm('Reset all settings to defaults?')) return;

    currentSettings = { ...DEFAULT_SETTINGS };
    updateUIFromSettings();
    showSettingsStatus('Settings reset to defaults. Click Apply or Save to confirm.', 'info');
}

/**
 * Show status message in settings modal
 */
function showSettingsStatus(message, type) {
    const status = document.getElementById('settingsStatus');
    status.textContent = message;
    status.className = 'settings-status ' + type;
    status.style.display = 'block';
}

/**
 * Hide status message
 */
function hideSettingsStatus() {
    const status = document.getElementById('settingsStatus');
    status.style.display = 'none';
}

// Initialize settings event listeners when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Inference interval slider
    const inferenceInput = document.getElementById('inferenceInterval');
    if (inferenceInput) {
        inferenceInput.addEventListener('input', (e) => {
            document.getElementById('inferenceIntervalValue').textContent = `${e.target.value}s`;
        });
    }

    // Confidence threshold slider
    const confInput = document.getElementById('confidenceThreshold');
    if (confInput) {
        confInput.addEventListener('input', (e) => {
            document.getElementById('confidenceThresholdValue').textContent = parseFloat(e.target.value).toFixed(2);
        });
    }

    // IOU threshold slider
    const iouInput = document.getElementById('iouThreshold');
    if (iouInput) {
        iouInput.addEventListener('input', (e) => {
            document.getElementById('iouThresholdValue').textContent = parseFloat(e.target.value).toFixed(2);
        });
    }

    // Half precision toggle
    const halfPrecisionInput = document.getElementById('halfPrecision');
    if (halfPrecisionInput) {
        halfPrecisionInput.addEventListener('change', (e) => {
            document.getElementById('halfPrecisionValue').textContent = e.target.checked ? 'Enabled' : 'Disabled';
        });
    }

    // Denoise toggle
    const denoiseInput = document.getElementById('denoise');
    if (denoiseInput) {
        denoiseInput.addEventListener('change', (e) => {
            const value = document.getElementById('denoiseValue');
            value.textContent = e.target.checked ? 'Enabled' : 'Disabled';
            if (e.target.checked) {
                value.classList.add('setting-warning');
            } else {
                value.classList.remove('setting-warning');
            }
        });
    }

    // CLAHE toggle
    const claheInput = document.getElementById('clahe');
    if (claheInput) {
        claheInput.addEventListener('change', (e) => {
            document.getElementById('claheValue').textContent = e.target.checked ? 'Enabled' : 'Disabled';
        });
    }

    // Histogram equalization toggle
    const eqInput = document.getElementById('equalizeHistogram');
    if (eqInput) {
        eqInput.addEventListener('change', (e) => {
            document.getElementById('equalizeHistogramValue').textContent = e.target.checked ? 'Enabled' : 'Disabled';
        });
    }

    // Close modal on outside click
    const modal = document.getElementById('settingsModal');
    if (modal) {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                closeSettings();
            }
        });
    }

    // Close modal on Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            const modal = document.getElementById('settingsModal');
            if (modal && modal.classList.contains('show')) {
                closeSettings();
            }
        }
    });
});
