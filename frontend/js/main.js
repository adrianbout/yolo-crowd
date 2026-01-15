/**
 * Main Application
 * Initializes and coordinates all components
 */

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('SmartChairCounter Dashboard Loading...');

    // Initialize components
    initCameraGrid();
    initROIEditor();
    initWebSocket();

    // Request initial status
    setTimeout(() => {
        if (wsClient && wsClient.isConnected) {
            wsClient.requestStatus();
        }
    }, 1000);

    // HTTP polling fallback for status updates (like quick_view)
    // This ensures counter updates even if WebSocket is not working
    setInterval(updateStatusHTTP, 500);
    updateStatusHTTP();

    console.log('SmartChairCounter Dashboard Loaded');
});

// HTTP polling fallback for status updates (like quick_view)
async function updateStatusHTTP() {
    try {
        const response = await fetch(`${API_BASE}/status`);
        const data = await response.json();

        // Update total count
        const totalCountElement = document.getElementById('totalCount');
        if (totalCountElement) {
            totalCountElement.textContent = data.system.total_count || 0;
        }

        // Update last update time
        const lastUpdateElement = document.getElementById('lastUpdate');
        if (lastUpdateElement) {
            const now = new Date();
            lastUpdateElement.textContent = `Last updated: ${now.toLocaleTimeString()}`;
        }

        // Update active cameras count
        const activeCamerasElement = document.getElementById('activeCameras');
        if (activeCamerasElement && data.cameras) {
            const connectedCount = Object.values(data.cameras)
                .filter(cam => cam.connected).length;
            activeCamerasElement.textContent = connectedCount;
        }

        // Update system status
        const systemStatusElement = document.getElementById('systemStatus');
        if (systemStatusElement) {
            systemStatusElement.textContent = data.system.running ? 'Running' : 'Stopped';
        }

        // Update uptime
        const uptimeElement = document.getElementById('uptime');
        if (uptimeElement && data.system.uptime_seconds) {
            const hours = Math.floor(data.system.uptime_seconds / 3600);
            const minutes = Math.floor((data.system.uptime_seconds % 3600) / 60);
            uptimeElement.textContent = `Uptime: ${hours}h ${minutes}m`;
        }

        // Update individual camera counts (YOLO raw counts)
        if (data.cameras && cameraGrid) {
            const countsByCamera = {};
            Object.entries(data.cameras).forEach(([cameraId, camData]) => {
                if (camData.count !== undefined) {
                    countsByCamera[cameraId] = camData.count;
                }
            });
            cameraGrid.updateCounts(countsByCamera);

            // Update median counts
            cameraGrid.updateMedianCounts(data.cameras);
        }

    } catch (error) {
        console.error('Error fetching status via HTTP:', error);
    }
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (wsClient) {
        wsClient.disconnect();
    }

    if (cameraGrid) {
        cameraGrid.stopFeedUpdates();
    }
});

// Global utility functions
async function refreshData() {
    console.log('Refreshing data...');

    // Reload cameras
    if (cameraGrid) {
        await cameraGrid.loadCameras();
    }

    // Request fresh counts and status
    if (wsClient && wsClient.isConnected) {
        wsClient.requestCounts();
        wsClient.requestStatus();
    }

    // Show feedback
    showNotification('Data refreshed', 'success');
}

async function resetCounts() {
    if (!confirm('Are you sure you want to reset all counts to zero?')) {
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/counting/reset`, {
            method: 'POST'
        });

        if (!response.ok) throw new Error('Failed to reset counts');

        showNotification('Counts reset successfully', 'success');

        // Refresh data
        setTimeout(refreshData, 500);

    } catch (error) {
        console.error('Error resetting counts:', error);
        showNotification('Failed to reset counts', 'error');
    }
}

function toggleROIMode() {
    showNotification('Click "Edit ROI" on any camera to define counting zones', 'info');
}

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;

    // Add styles
    notification.style.position = 'fixed';
    notification.style.top = '20px';
    notification.style.right = '20px';
    notification.style.padding = '15px 25px';
    notification.style.borderRadius = '5px';
    notification.style.boxShadow = '0 3px 10px rgba(0,0,0,0.3)';
    notification.style.zIndex = '10000';
    notification.style.fontWeight = '600';
    notification.style.animation = 'slideIn 0.3s ease';

    if (type === 'success') {
        notification.style.backgroundColor = '#27ae60';
        notification.style.color = 'white';
    } else if (type === 'error') {
        notification.style.backgroundColor = '#e74c3c';
        notification.style.color = 'white';
    } else {
        notification.style.backgroundColor = '#3498db';
        notification.style.color = 'white';
    }

    document.body.appendChild(notification);

    // Remove after 3 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 3000);
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }

    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Error handling
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
});

window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
});
