/**
 * Floor plan calibration.
 *
 * Click a landmark in the camera view, then the same spot on the plan. Four
 * pairs determine the perspective between them; the backend solves it and
 * reports how well the points agreed.
 *
 * Clicks are converted to the image's own pixel coordinates rather than
 * screen coordinates, so a calibration done on a laptop still holds when the
 * same plan is opened on a wall display.
 */

const API = `${window.location.origin}/api`;

const state = {
    plan: null,          // { image, width, height }
    cameraId: null,
    pairs: [],           // [{ cam:[x,y], plan:[x,y] }]
    pendingCam: null,    // camera point waiting for its plan counterpart
    heatOn: false,
    heatTimer: null,
    mode: 'quad',        // 'quad' (drag a grid) | 'points' (click pairs)
    camQuad: null,       // 4 corners in camera pixels, clockwise from top-left
    planQuad: null,      // the same four corners on the plan
    drag: null,
    handles: { cam: null, plan: null },
};

// Corner order is fixed so the two quads correspond: a handle labelled 2 on
// the camera is the same physical corner as handle 2 on the plan.
const CORNERS = ['1', '2', '3', '4'];

const el = id => document.getElementById(id);

// ---------------------------------------------------------------- utilities

function toast(message, isError) {
    const t = el('toast');
    t.textContent = message;
    t.classList.toggle('error', !!isError);
    t.hidden = false;
    clearTimeout(toast._t);
    toast._t = setTimeout(() => { t.hidden = true; }, isError ? 6000 : 3000);
}

/**
 * Where a click landed, in the image's own pixels.
 *
 * Measured against the image element itself, never its container. The two
 * differ whenever the picture's aspect ratio does not match the panel's, and
 * a portrait plan in a wide panel differs by a lot - which is exactly how
 * markers ended up offset from the clicks that placed them.
 */
function imagePoint(img, event, clamp) {
    const rect = img.getBoundingClientRect();
    if (!rect.width || !rect.height || !img.naturalWidth) return null;

    let x = (event.clientX - rect.left) / rect.width * img.naturalWidth;
    let y = (event.clientY - rect.top) / rect.height * img.naturalHeight;

    if (clamp) {
        // Dragging past the edge pins to the edge; a stray click there is
        // simply not a point.
        x = Math.min(Math.max(x, 0), img.naturalWidth);
        y = Math.min(Math.max(y, 0), img.naturalHeight);
    } else if (x < 0 || y < 0 || x > img.naturalWidth || y > img.naturalHeight) {
        return null;
    }

    return [Math.round(x * 10) / 10, Math.round(y * 10) / 10];
}

/** Place a numbered marker over an image, positioned as a percentage so it survives resizing. */
function addMarker(container, img, point, label, pending) {
    const dot = document.createElement('div');
    dot.className = 'marker' + (pending ? ' pending' : '');
    dot.style.left = (point[0] / img.naturalWidth * 100) + '%';
    dot.style.top = (point[1] / img.naturalHeight * 100) + '%';
    dot.textContent = label;
    container.appendChild(dot);
}

// ------------------------------------------------------------------ drawing

function renderMarkers() {
    const camWrap = el('cameraMarkers');
    const planWrap = el('planMarkers');
    const camImg = el('cameraImage');
    const planImg = el('planImage');
    camWrap.innerHTML = '';
    planWrap.innerHTML = '';

    state.pairs.forEach((pair, i) => {
        if (camImg.naturalWidth) addMarker(camWrap, camImg, pair.cam, i + 1, false);
        if (planImg.naturalWidth) addMarker(planWrap, planImg, pair.plan, i + 1, false);
    });

    if (state.pendingCam && camImg.naturalWidth) {
        addMarker(camWrap, camImg, state.pendingCam, state.pairs.length + 1, true);
    }
}

function renderPairs() {
    const list = el('pairList');
    const pairs = effectivePairs();
    el('pairCount').textContent = pairs.length;

    if (!pairs.length) {
        list.innerHTML = '<p class="hint">No points yet. Four is the minimum; five or six absorbs a mis-click.</p>';
    } else {
        list.innerHTML = '';
        pairs.forEach((pair, i) => {
            const row = document.createElement('div');
            row.className = 'pair';
            row.innerHTML =
                `<span class="idx">${i + 1}</span>` +
                `<span class="coord">camera <b>${pair.cam[0]}, ${pair.cam[1]}</b></span>` +
                `<span class="coord">plan <b>${pair.plan[0]}, ${pair.plan[1]}</b></span>`;
            const drop = document.createElement('button');
            drop.className = 'drop';
            drop.type = 'button';
            drop.title = `Remove point ${i + 1}`;
            drop.setAttribute('aria-label', `Remove point ${i + 1}`);
            drop.textContent = '×';
            drop.addEventListener('click', () => {
                state.pairs.splice(i, 1);
                refreshEditor();
            });
            // A quad always has exactly four corners - they move, they do not
            // get deleted.
            if (state.mode !== 'quad') row.appendChild(drop);
            list.appendChild(row);
        });
    }

    const enough = pairs.length >= 4;
    el('saveBtn').disabled = !enough || !state.cameraId;

    const quad = state.mode === 'quad';
    el('undoBtn').disabled = quad || (!state.pairs.length && !state.pendingCam);
    el('clearBtn').disabled = !state.cameraId;
    el('stepBar').hidden = quad;

    if (!quad) {
        el('step1').className = 'step' + (state.pendingCam ? ' done' : ' active');
        el('step2').className = 'step' + (state.pendingCam ? ' active' : '');
        el('step3').className = 'step' + (enough ? ' active' : '');
    }
}

/**
 * The point pairs the current mode has produced.
 *
 * Both modes feed the same four-or-more correspondences to the same endpoint;
 * they differ only in how a person produces them.
 */
function effectivePairs() {
    if (state.mode === 'quad') {
        if (!state.camQuad || !state.planQuad) return [];
        return state.camQuad.map((c, i) => ({ cam: c, plan: state.planQuad[i] }));
    }
    return state.pairs;
}

function refreshEditor() {
    if (state.mode === 'quad') {
        renderQuad();
    } else {
        clearGrids();
        // renderMarkers() empties the same containers, which would leave the
        // cached handles pointing at detached nodes.
        resetHandles();
        renderMarkers();
    }
    renderPairs();
}

// --------------------------------------------------------------- plan + data

async function loadPlan() {
    let data;
    try {
        const res = await fetch(`${API}/floorplan`);
        data = await res.json();
    } catch (e) {
        toast('Cannot reach the backend. Is it running?', true);
        return;
    }

    if (!data.has_image) {
        el('emptyState').hidden = false;
        el('workspace').hidden = true;
        return;
    }

    state.plan = data;
    el('emptyState').hidden = true;
    el('workspace').hidden = false;

    // Cache-busted so a replaced plan actually reappears.
    el('planImage').src = `${API}/floorplan/image?t=${Date.now()}`;
    el('planImage').addEventListener('load', refreshEditor, { once: true });
}

async function loadCameras() {
    let cams;
    try {
        const res = await fetch(`${API}/cameras`);
        cams = await res.json();
    } catch (e) {
        toast('Could not load the camera list.', true);
        return;
    }
    const list = Array.isArray(cams) ? cams : (cams.cameras || []);
    const select = el('cameraSelect');
    select.innerHTML = '<option value="">Select a camera…</option>';
    list.forEach(c => {
        const opt = document.createElement('option');
        opt.value = c.id;
        opt.textContent = c.name ? `${c.name} (${c.id})` : c.id;
        select.appendChild(opt);
    });
}

async function selectCamera(cameraId) {
    state.cameraId = cameraId || null;
    state.pairs = [];
    state.pendingCam = null;
    state.camQuad = null;
    state.planQuad = null;
    resetHandles();
    el('quality').hidden = true;

    const img = el('cameraImage');
    if (!cameraId) {
        img.removeAttribute('src');
        el('cameraEmpty').hidden = false;
        el('calStatus').textContent = 'Pick a camera to begin.';
        el('calStatus').className = 'status-line';
        refreshEditor();
        return;
    }

    el('cameraEmpty').hidden = true;
    // draw_rois=false so boxes and zone outlines do not hide the landmarks
    // being clicked.
    img.src = `${API}/cameras/${encodeURIComponent(cameraId)}/frame?draw_rois=false&t=${Date.now()}`;
    img.addEventListener('load', refreshEditor, { once: true });
    img.addEventListener('error', () => {
        el('calStatus').textContent = 'No frame from that camera - is it enabled and connected?';
        el('calStatus').className = 'status-line warn';
    }, { once: true });

    // Reopen an existing survey rather than starting blank, so a calibration
    // can be corrected a point at a time.
    try {
        const res = await fetch(`${API}/floorplan/cameras/${encodeURIComponent(cameraId)}/calibration`);
        const cal = await res.json();
        if (cal.calibrated && cal.camera_points && cal.camera_points.length) {
            state.pairs = cal.camera_points.map((c, i) => ({ cam: c, plan: cal.plan_points[i] }));
            // Exactly four points round-trip as a quad, so a grid survey can be
            // reopened and nudged. More than four came from point mode and
            // cannot be shown as one.
            if (cal.camera_points.length === 4) {
                state.camQuad = cal.camera_points.map(p => p.slice());
                state.planQuad = cal.plan_points.map(p => p.slice());
            } else if (state.mode === 'quad') {
                setMode('points');
            }
            el('calStatus').textContent =
                `Already calibrated from ${cal.point_count} points, error ${cal.error_px}px. Adjust and save again to update.`;
            el('calStatus').className = 'status-line good';
        } else {
            el('calStatus').textContent = 'Not calibrated yet. Match at least 4 points.';
            el('calStatus').className = 'status-line';
        }
    } catch (e) {
        el('calStatus').textContent = 'Not calibrated yet. Match at least 4 points.';
        el('calStatus').className = 'status-line';
    }
    refreshEditor();
}

// ------------------------------------------------------------------- actions

function onCameraClick(event) {
    if (state.mode === 'quad') return;   // corners are dragged, not clicked
    if (!state.cameraId) { toast('Pick a camera first.'); return; }
    const point = imagePoint(el('cameraImage'), event);
    if (!point) return;
    if (state.pendingCam) {
        // Re-clicking replaces the pending point instead of stacking them.
        state.pendingCam = point;
        toast('Moved the pending point. Now click its match on the plan.');
    } else {
        state.pendingCam = point;
    }
    refreshEditor();
}

function onPlanClick(event) {
    if (state.mode === 'quad') return;
    if (!state.cameraId) { toast('Pick a camera first.'); return; }
    if (!state.pendingCam) { toast('Click the landmark in the camera view first.'); return; }
    const point = imagePoint(el('planImage'), event);
    if (!point) return;
    state.pairs.push({ cam: state.pendingCam, plan: point });
    state.pendingCam = null;
    refreshEditor();
}

async function saveCalibration() {
    const pairs = effectivePairs();
    if (!state.cameraId || pairs.length < 4) return;
    const btn = el('saveBtn');
    btn.disabled = true;
    btn.textContent = 'Saving…';

    try {
        const res = await fetch(`${API}/floorplan/cameras/${encodeURIComponent(state.cameraId)}/calibration`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                camera_points: pairs.map(p => p.cam),
                plan_points: pairs.map(p => p.plan),
            }),
        });
        const data = await res.json();
        if (!res.ok) {
            toast(data.detail || 'Could not save that calibration.', true);
            return;
        }

        const q = el('quality');
        q.hidden = false;
        q.className = 'quality ' + (data.quality || '').split(/[ ,-]/)[0];
        q.textContent = `Saved. Fit is ${data.quality} — points land an average of ${data.error_px}px from where you put them.`;
        el('calStatus').textContent = `Calibrated from ${data.point_count} points.`;
        el('calStatus').className = 'status-line good';
        toast('Calibration saved.');
        if (state.heatOn) loadHeatmap();
    } catch (e) {
        toast('Could not reach the backend to save.', true);
    } finally {
        btn.textContent = 'Save calibration';
        btn.disabled = effectivePairs().length < 4;
    }
}

function undo() {
    if (state.pendingCam) state.pendingCam = null;
    else state.pairs.pop();
    el('quality').hidden = true;
    refreshEditor();
}

function clearAll() {
    state.pairs = [];
    state.pendingCam = null;
    state.camQuad = null;
    state.planQuad = null;
    resetHandles();
    el('quality').hidden = true;
    refreshEditor();
}

// ------------------------------------------------------------------- upload

function pickFile() { el('fileInput').click(); }

async function uploadPlan(file) {
    if (!file) return;
    toast('Uploading plan…');
    try {
        // Sent as the raw body: the backend reads the format from the bytes,
        // which avoids a multipart dependency on the server.
        const res = await fetch(`${API}/floorplan/image`, {
            method: 'POST',
            headers: { 'Content-Type': file.type || 'application/octet-stream' },
            body: file,
        });
        const data = await res.json();
        if (!res.ok) { toast(data.detail || 'Upload failed.', true); return; }

        toast(data.recalibration_needed
            ? `Plan replaced at a new size (${data.width}×${data.height}). Existing calibrations no longer line up.`
            : `Plan uploaded (${data.width}×${data.height}).`,
            !!data.recalibration_needed);

        await loadPlan();
        if (state.cameraId) selectCamera(state.cameraId);
    } catch (e) {
        toast('Could not upload the plan.', true);
    }
}

// ------------------------------------------------------------------ heatmap

/** Blue to red ramp, matching the overlay drawn on the camera feeds. */
function heatColor(v) {
    const stops = [
        [0.00, [44, 127, 184]],
        [0.40, [127, 205, 187]],
        [0.65, [255, 220, 100]],
        [1.00, [231, 76, 60]],
    ];
    for (let i = 1; i < stops.length; i++) {
        if (v <= stops[i][0]) {
            const [p0, c0] = stops[i - 1], [p1, c1] = stops[i];
            const t = (v - p0) / (p1 - p0 || 1);
            return c0.map((c, k) => Math.round(c + (c1[k] - c) * t));
        }
    }
    return stops[stops.length - 1][1];
}

async function loadHeatmap() {
    const canvas = el('heatCanvas');
    try {
        const res = await fetch(`${API}/floorplan/heatmap`);
        if (!res.ok) return;
        const data = await res.json();
        const hm = data.heatmap;
        if (!hm || !hm.grid || !hm.grid.length) {
            canvas.classList.remove('on');
            el('calStatus').textContent = data.calibrated_count
                ? 'No traffic recorded yet on the calibrated cameras.'
                : 'No camera is calibrated yet, so the plan has nothing to show.';
            el('calStatus').className = 'status-line warn';
            return;
        }

        canvas.width = hm.width;
        canvas.height = hm.height;
        const ctx = canvas.getContext('2d');
        const img = ctx.createImageData(hm.width, hm.height);

        for (let y = 0; y < hm.height; y++) {
            for (let x = 0; x < hm.width; x++) {
                const v = Math.max(0, Math.min(1, hm.grid[y][x]));
                const o = (y * hm.width + x) * 4;
                if (v < 0.02) { img.data[o + 3] = 0; continue; }
                const [r, g, b] = heatColor(v);
                img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b;
                // Opacity tracks intensity, so quiet floor stays readable.
                img.data[o + 3] = Math.round(40 + v * 175);
            }
        }
        ctx.putImageData(img, 0, 0);
        canvas.classList.add('on');
    } catch (e) {
        /* leave the last frame up rather than blanking the plan */
    }
}

function toggleHeat() {
    state.heatOn = !state.heatOn;
    const btn = el('heatBtn');
    btn.setAttribute('aria-pressed', String(state.heatOn));
    btn.textContent = state.heatOn ? 'Hide heatmap' : 'Show heatmap';
    el('heatLegend').hidden = !state.heatOn;

    clearInterval(state.heatTimer);
    if (state.heatOn) {
        loadHeatmap();
        state.heatTimer = setInterval(loadHeatmap, 3000);
    } else {
        el('heatCanvas').classList.remove('on');
    }
}

// --------------------------------------------------------------------- boot

document.addEventListener('DOMContentLoaded', () => {
    el('cameraImage').addEventListener('click', onCameraClick);
    el('planImage').addEventListener('click', onPlanClick);
    el('cameraSelect').addEventListener('change', e => selectCamera(e.target.value));
    el('saveBtn').addEventListener('click', saveCalibration);
    el('undoBtn').addEventListener('click', undo);
    el('clearBtn').addEventListener('click', clearAll);
    el('heatBtn').addEventListener('click', toggleHeat);
    el('modeQuadBtn').addEventListener('click', () => setMode('quad'));
    el('modePointsBtn').addEventListener('click', () => setMode('points'));
    el('uploadBtn').addEventListener('click', pickFile);
    el('emptyUploadBtn').addEventListener('click', pickFile);
    el('refreshFrameBtn').addEventListener('click', () => {
        if (state.cameraId) {
            el('cameraImage').src =
                `${API}/cameras/${encodeURIComponent(state.cameraId)}/frame?draw_rois=false&t=${Date.now()}`;
        }
    });
    el('fileInput').addEventListener('change', e => {
        uploadPlan(e.target.files[0]);
        e.target.value = '';   // so re-picking the same file fires again
    });
    window.addEventListener('resize', () => {
        if (state.drag) return;              // never reshuffle mid-drag
        if (state.mode === 'quad') { syncQuad('cam'); syncQuad('plan'); }
        else renderMarkers();
    });

    loadPlan();
    loadCameras();
});


// --------------------------------------------------------------- grid mode

/**
 * Projective transform taking the unit square to a quadrilateral.
 *
 * Used only for drawing: it lets the guide grid follow the same perspective
 * the corners imply, so the lines lie flat on the floor instead of dividing
 * the quad into equal-looking slabs. If the grid tracks the floor's own
 * lines - tiles, kerbs, the base of a wall - the corners are placed well,
 * which is a far easier judgement than reading four coordinates.
 */
function squareToQuad(q) {
    const [x0, y0] = q[0], [x1, y1] = q[1], [x2, y2] = q[2], [x3, y3] = q[3];
    const dx1 = x1 - x2, dy1 = y1 - y2;
    const dx2 = x3 - x2, dy2 = y3 - y2;
    const sx = x0 - x1 + x2 - x3, sy = y0 - y1 + y2 - y3;

    let g = 0, h = 0;
    const den = dx1 * dy2 - dx2 * dy1;
    if (Math.abs(den) > 1e-10) {
        g = (sx * dy2 - dx2 * sy) / den;
        h = (dx1 * sy - sx * dy1) / den;
    }
    return [
        x1 - x0 + g * x1, x3 - x0 + h * x3, x0,
        y1 - y0 + g * y1, y3 - y0 + h * y3, y0,
        g, h, 1,
    ];
}

function applyH(H, u, v) {
    const w = H[6] * u + H[7] * v + H[8] || 1e-10;
    return [(H[0] * u + H[1] * v + H[2]) / w, (H[3] * u + H[4] * v + H[5]) / w];
}

/** A starting quad covering the middle of the image, for the operator to drag out. */
function defaultQuad(img) {
    const w = img.naturalWidth, h = img.naturalHeight;
    const mx = w * 0.18, my = h * 0.22;
    return [
        [mx, my], [w - mx, my], [w - mx, h - my], [mx, h - my],
    ].map(p => [Math.round(p[0] * 10) / 10, Math.round(p[1] * 10) / 10]);
}

function drawGrid(canvas, img, quad, colour) {
    if (!img.naturalWidth) return;
    // Assigning width/height reallocates the buffer and clears it, so it is
    // done only when the size actually changes - otherwise every pointermove
    // throws away a full-resolution canvas and the drag stutters.
    if (canvas.width !== img.naturalWidth || canvas.height !== img.naturalHeight) {
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
    }
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!quad) { canvas.classList.remove('on'); return; }

    const H = squareToQuad(quad);
    // Scaled to the image so the grid reads the same on a 417px plan and a
    // 1920px frame.
    const unit = Math.max(canvas.width, canvas.height) / 500;
    const STEPS = 4;

    ctx.lineWidth = Math.max(1, unit);
    ctx.strokeStyle = colour + '66';
    for (let i = 1; i < STEPS; i++) {
        const t = i / STEPS;
        ctx.beginPath();
        for (let j = 0; j <= 24; j++) {
            const [x, y] = applyH(H, j / 24, t);
            j ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
        }
        ctx.stroke();
        ctx.beginPath();
        for (let j = 0; j <= 24; j++) {
            const [x, y] = applyH(H, t, j / 24);
            j ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
        }
        ctx.stroke();
    }

    // Outline last and heavier, so the footprint stays legible over the grid.
    ctx.lineWidth = Math.max(2, unit * 2);
    ctx.strokeStyle = colour;
    ctx.beginPath();
    quad.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.closePath();
    ctx.stroke();

    ctx.fillStyle = colour + '1f';
    ctx.fill();
    canvas.classList.add('on');
}

function clearGrids() {
    ['cameraGrid', 'planGrid'].forEach(id => {
        const c = el(id);
        c.classList.remove('on');
        const ctx = c.getContext('2d');
        ctx && ctx.clearRect(0, 0, c.width, c.height);
    });
}

/**
 * Build the four handles for one side, once.
 *
 * Handles are created and then kept. Rebuilding them during a drag - which an
 * earlier version did on every pointermove - detaches the element holding the
 * pointer capture, so the corner lurched once and then stopped following the
 * cursor entirely.
 */
function buildHandles(which) {
    const img = which === 'cam' ? el('cameraImage') : el('planImage');
    const container = which === 'cam' ? el('cameraMarkers') : el('planMarkers');
    container.innerHTML = '';

    const handles = CORNERS.map((label, idx) => {
        const h = document.createElement('div');
        h.className = 'handle';
        h.textContent = label;
        h.tabIndex = 0;
        h.setAttribute('role', 'slider');
        h.setAttribute('aria-label',
            `Corner ${label} on the ${which === 'cam' ? 'camera view' : 'floor plan'}`);

        h.addEventListener('pointerdown', ev => {
            ev.preventDefault();
            ev.stopPropagation();
            const q = which === 'cam' ? state.camQuad : state.planQuad;
            const at = imagePoint(img, ev, true);
            if (!q || !at) return;
            h.setPointerCapture(ev.pointerId);
            h.classList.add('dragging');
            // Where inside the handle it was grabbed, so it moves with the
            // cursor instead of snapping its centre under it.
            state.drag = {
                which, idx, el: h,
                offset: [q[idx][0] - at[0], q[idx][1] - at[1]],
            };
        });

        h.addEventListener('pointermove', ev => {
            const d = state.drag;
            if (!d || d.el !== h) return;
            const at = imagePoint(img, ev, true);
            if (!at) return;
            const q = which === 'cam' ? state.camQuad : state.planQuad;
            q[idx] = [
                Math.min(Math.max(at[0] + d.offset[0], 0), img.naturalWidth),
                Math.min(Math.max(at[1] + d.offset[1], 0), img.naturalHeight),
            ];
            // Positions and grid only - never a rebuild while dragging.
            syncQuad(which);
        });

        const end = ev => {
            const d = state.drag;
            if (!d || d.el !== h) return;
            h.classList.remove('dragging');
            try { h.releasePointerCapture(ev.pointerId); } catch (e) { /* already released */ }
            state.drag = null;
            renderPairs();
        };
        h.addEventListener('pointerup', end);
        h.addEventListener('pointercancel', end);
        h.addEventListener('lostpointercapture', end);

        // Keyboard nudging, for placing a corner exactly.
        h.addEventListener('keydown', ev => {
            const step = ev.shiftKey ? 10 : 1;
            const moves = {
                ArrowLeft: [-step, 0], ArrowRight: [step, 0],
                ArrowUp: [0, -step], ArrowDown: [0, step],
            };
            const mv = moves[ev.key];
            if (!mv) return;
            ev.preventDefault();
            const q = which === 'cam' ? state.camQuad : state.planQuad;
            q[idx] = [
                Math.min(Math.max(q[idx][0] + mv[0], 0), img.naturalWidth),
                Math.min(Math.max(q[idx][1] + mv[1], 0), img.naturalHeight),
            ];
            syncQuad(which);
            renderPairs();
        });

        container.appendChild(h);
        return h;
    });

    state.handles[which] = handles;
}

/** Move the existing handles to match the quad, and redraw its grid. */
function syncQuad(which) {
    const img = which === 'cam' ? el('cameraImage') : el('planImage');
    const quad = which === 'cam' ? state.camQuad : state.planQuad;
    const handles = state.handles[which];
    if (!img.naturalWidth || !quad || !handles) return;

    handles.forEach((h, i) => {
        h.style.left = (quad[i][0] / img.naturalWidth * 100) + '%';
        h.style.top = (quad[i][1] / img.naturalHeight * 100) + '%';
    });
    drawGrid(
        el(which === 'cam' ? 'cameraGrid' : 'planGrid'),
        img, quad,
        which === 'cam' ? '#e67e22' : '#3498db'
    );
}

function renderQuad() {
    const camImg = el('cameraImage'), planImg = el('planImage');

    if (camImg.naturalWidth && !state.camQuad) state.camQuad = defaultQuad(camImg);
    if (planImg.naturalWidth && !state.planQuad) state.planQuad = defaultQuad(planImg);

    if (camImg.naturalWidth && state.camQuad) {
        if (!state.handles.cam) buildHandles('cam');
        syncQuad('cam');
    }
    if (planImg.naturalWidth && state.planQuad) {
        if (!state.handles.plan) buildHandles('plan');
        syncQuad('plan');
    }
}

/** Drop the handle elements so the next render rebuilds them. */
function resetHandles() {
    state.handles = { cam: null, plan: null };
    el('cameraMarkers').innerHTML = '';
    el('planMarkers').innerHTML = '';
}

function setMode(mode) {
    state.mode = mode;
    state.drag = null;
    resetHandles();
    el('modeQuadBtn').setAttribute('aria-pressed', String(mode === 'quad'));
    el('modePointsBtn').setAttribute('aria-pressed', String(mode === 'points'));
    el('modeNote').innerHTML = mode === 'quad'
        ? 'Drag the four corners so the grid covers the same patch of floor in both pictures. ' +
          'The grid lines should follow the floor&rsquo;s own lines &mdash; tiles, kerbs, wall bases.'
        : 'Click a landmark in the camera view, then the same spot on the plan. ' +
          'Repeat for at least four points; five or six absorbs a mis-click.';
    el('quality').hidden = true;
    refreshEditor();
}
