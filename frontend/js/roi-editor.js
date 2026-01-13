/**
 * ROI Editor
 * Interactive polygon drawing for defining counting zones
 */

class ROIEditor {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.currentCameraId = null;
        this.currentCameraName = null;

        this.image = new Image();
        this.polygons = [];
        this.currentPolygon = [];
        this.isDrawing = false;

        this.setupEventListeners();
    }

    setupEventListeners() {
        this.canvas.addEventListener('click', (e) => this.handleClick(e));
        this.canvas.addEventListener('dblclick', (e) => this.handleDoubleClick(e));
        this.canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
    }

    async open(cameraId, cameraName) {
        this.currentCameraId = cameraId;
        this.currentCameraName = cameraName;

        document.getElementById('roiCameraName').textContent = cameraName;
        document.getElementById('roiEditorModal').classList.add('show');

        // Load camera frame
        await this.loadCameraFrame(cameraId);

        // Load existing ROI
        await this.loadExistingROI(cameraId);

        this.redraw();
    }

    close() {
        document.getElementById('roiEditorModal').classList.remove('show');
        this.currentCameraId = null;
        this.currentPolygon = [];
        this.polygons = [];
        this.isDrawing = false;
    }

    async loadCameraFrame(cameraId) {
        return new Promise((resolve, reject) => {
            this.image.onload = () => {
                // Set canvas size to image size
                this.canvas.width = this.image.width;
                this.canvas.height = this.image.height;
                resolve();
            };

            this.image.onerror = reject;

            // Load frame without ROIs drawn
            this.image.src = `${API_BASE}/cameras/${cameraId}/frame?draw_rois=false&t=${Date.now()}`;
        });
    }

    async loadExistingROI(cameraId) {
        try {
            const response = await fetch(`${API_BASE}/roi/${cameraId}`);
            const roiData = await response.json();

            if (roiData.enabled && roiData.polygons && roiData.polygons.length > 0) {
                this.polygons = roiData.polygons.map(p => ({
                    name: p.name,
                    points: p.points,
                    description: p.description || ''
                }));
            }
        } catch (error) {
            console.error('Error loading existing ROI:', error);
        }
    }

    handleClick(e) {
        if (!this.isDrawing) {
            this.isDrawing = true;
            this.currentPolygon = [];
        }

        const rect = this.canvas.getBoundingClientRect();
        const x = Math.round((e.clientX - rect.left) * (this.canvas.width / rect.width));
        const y = Math.round((e.clientY - rect.top) * (this.canvas.height / rect.height));

        this.currentPolygon.push([x, y]);
        this.redraw();
    }

    handleDoubleClick(e) {
        e.preventDefault();

        if (this.currentPolygon.length >= 3) {
            // Finish current polygon
            const polyName = `ROI_${this.polygons.length + 1}`;
            this.polygons.push({
                name: polyName,
                points: this.currentPolygon,
                description: `Counting zone ${this.polygons.length + 1}`
            });

            this.currentPolygon = [];
            this.isDrawing = false;
            this.redraw();

            this.showStatus('Polygon completed. You can draw another or save.', 'success');
        }
    }

    handleMouseMove(e) {
        if (!this.isDrawing || this.currentPolygon.length === 0) return;

        const rect = this.canvas.getBoundingClientRect();
        const x = Math.round((e.clientX - rect.left) * (this.canvas.width / rect.width));
        const y = Math.round((e.clientY - rect.top) * (this.canvas.height / rect.height));

        this.redraw();

        // Draw line from last point to cursor
        this.ctx.strokeStyle = '#3498db';
        this.ctx.lineWidth = 2;
        this.ctx.setLineDash([5, 5]);
        this.ctx.beginPath();
        const lastPoint = this.currentPolygon[this.currentPolygon.length - 1];
        this.ctx.moveTo(lastPoint[0], lastPoint[1]);
        this.ctx.lineTo(x, y);
        this.ctx.stroke();
        this.ctx.setLineDash([]);
    }

    redraw() {
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw image
        this.ctx.drawImage(this.image, 0, 0);

        // Draw completed polygons
        this.polygons.forEach((polygon, idx) => {
            this.drawPolygon(polygon.points, '#27ae60', true, polygon.name);
        });

        // Draw current polygon being drawn
        if (this.currentPolygon.length > 0) {
            this.drawPolygon(this.currentPolygon, '#3498db', false);
        }
    }

    drawPolygon(points, color, filled = false, label = null) {
        if (points.length === 0) return;

        this.ctx.strokeStyle = color;
        this.ctx.fillStyle = color;
        this.ctx.lineWidth = 2;

        // Draw polygon
        this.ctx.beginPath();
        this.ctx.moveTo(points[0][0], points[0][1]);

        for (let i = 1; i < points.length; i++) {
            this.ctx.lineTo(points[i][0], points[i][1]);
        }

        if (filled) {
            this.ctx.closePath();
            this.ctx.globalAlpha = 0.2;
            this.ctx.fill();
            this.ctx.globalAlpha = 1.0;
        }

        this.ctx.stroke();

        // Draw points
        points.forEach((point, idx) => {
            this.ctx.fillStyle = color;
            this.ctx.beginPath();
            this.ctx.arc(point[0], point[1], 4, 0, 2 * Math.PI);
            this.ctx.fill();
        });

        // Draw label
        if (label && points.length > 0) {
            this.ctx.fillStyle = color;
            this.ctx.font = '14px Arial';
            this.ctx.fillText(label, points[0][0] + 10, points[0][1] - 10);
        }
    }

    clear() {
        this.polygons = [];
        this.currentPolygon = [];
        this.isDrawing = false;
        this.redraw();
        this.showStatus('ROI cleared', 'success');
    }

    async save() {
        if (this.polygons.length === 0) {
            this.showStatus('No ROI defined. Draw at least one polygon.', 'error');
            return;
        }

        try {
            const roiData = {
                enabled: true,
                polygons: this.polygons,
                notes: `ROI updated via UI at ${new Date().toISOString()}`
            };

            const response = await fetch(`${API_BASE}/roi/${this.currentCameraId}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(roiData)
            });

            if (!response.ok) throw new Error('Failed to save ROI');

            const result = await response.json();
            this.showStatus('ROI saved successfully!', 'success');

            setTimeout(() => {
                this.close();
            }, 1500);

        } catch (error) {
            console.error('Error saving ROI:', error);
            this.showStatus('Failed to save ROI', 'error');
        }
    }

    showStatus(message, type) {
        const statusElement = document.getElementById('roiStatus');
        statusElement.textContent = message;
        statusElement.className = `roi-status ${type}`;

        setTimeout(() => {
            statusElement.className = 'roi-status';
        }, 3000);
    }
}

// Global ROI editor instance
let roiEditor = null;

// Initialize ROI editor
function initROIEditor() {
    roiEditor = new ROIEditor('roiCanvas');
}

// Global functions for buttons
function openROIEditor(cameraId, cameraName) {
    roiEditor.open(cameraId, cameraName);
}

function closeROIEditor() {
    roiEditor.close();
}

function saveROI() {
    roiEditor.save();
}

function clearROI() {
    if (confirm('Are you sure you want to clear all ROI polygons?')) {
        roiEditor.clear();
    }
}

function cancelROI() {
    roiEditor.close();
}
