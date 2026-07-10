// Calibration E2E JS - Full Implementation with Draggable Canvas Bounding Box Editor

// DOM Elements
const videoSelect = document.getElementById('video-select');
const videoFileInput = document.getElementById('video-file');
const dropZone = document.getElementById('drop-zone');
const uploadStatus = document.getElementById('upload-status');
const zoneConfigPanel = document.getElementById('zone-config-panel');
const runPanel = document.getElementById('run-panel');

const currentVideoTitle = document.getElementById('current-video-title');
const currentVideoMeta = document.getElementById('current-video-meta');
const calibrationBadge = document.getElementById('calibration-badge');
const processingBadge = document.getElementById('processing-badge');

const canvas = document.getElementById('calibration-canvas');
const ctx = canvas.getContext('2d');
const canvasInstructions = document.getElementById('canvas-instructions');

const btnSuggest = document.getElementById('btn-suggest');
const btnSave = document.getElementById('btn-save');
const btnProcess = document.getElementById('btn-process');
const saveStatus = document.getElementById('save-status');
const processStatus = document.getElementById('process-status');

const marginOffset = document.getElementById('margin-offset');
const marginValue = document.getElementById('margin-value');

// Progress & Results Elements
const progressContainer = document.getElementById('progress-container');
const progressPercent = document.getElementById('progress-percent');
const progressBarFill = document.getElementById('progress-bar-fill');
const resultsPanel = document.getElementById('results-panel');
const btnViewVideo = document.getElementById('btn-view-video');
const btnViewLog = document.getElementById('btn-view-log');

const sumIntrusions = document.getElementById('sum-intrusions');
const sumTime = document.getElementById('sum-time');
const sumPeople = document.getElementById('sum-people');

// Application State
let state = {
    videoName: null,
    videoWidth: 0,
    videoHeight: 0,
    fps: 0,
    frameCount: 0,
    duration: 0,
    firstFrameImg: null,
    box: null, // Coordinates in original video space [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
};

// Interaction State
let isDragging = false;
let dragMode = null; // 'handle', 'edge', 'box'
let dragIndex = -1;  // Index of handle (0-3) or edge (0-3: top, right, bottom, left)
let dragStartPos = { x: 0, y: 0 };
let boxStartCoords = null; // Stored state of box when drag begins
let pollingInterval = null;

const HANDLE_RADIUS = 8;
const EDGE_THRESHOLD = 8;

// Initialization
document.addEventListener('DOMContentLoaded', () => {
    loadVideosList();
    setupUploadHandlers();
    setupCanvasInteractions();
    
    // Wire control buttons
    btnSuggest.addEventListener('click', runFloorMarkingsDetector);
    btnSave.addEventListener('click', saveCalibration);
    btnProcess.addEventListener('click', startVideoProcessing);
    
    marginOffset.addEventListener('input', () => {
        // Pixel margin display
        marginValue.textContent = `${marginOffset.value}px (~${Math.round(marginOffset.value * 0.6)}cm)`;
    });

    // ResizeObserver to dynamically resize the canvas to maximum available workspace space
    const viewport = document.getElementById('canvas-viewport');
    const resizeObserver = new ResizeObserver(() => {
        if (state.firstFrameImg) {
            renderCanvas();
        }
    });
    resizeObserver.observe(viewport);
});

// Fetch uploaded videos list
async function loadVideosList() {
    try {
        const response = await fetch('/api/videos');
        const videos = await response.json();
        
        const currentSelection = videoSelect.value;
        
        videoSelect.innerHTML = '<option value="">-- Select video --</option>';
        videos.forEach(video => {
            const opt = document.createElement('option');
            opt.value = video;
            opt.textContent = video;
            videoSelect.appendChild(opt);
        });
        
        if (currentSelection && videos.includes(currentSelection)) {
            videoSelect.value = currentSelection;
        } else if (videos.length > 0) {
            videoSelect.value = videos[0];
            loadVideoDetails(videos[0]);
        }
    } catch (err) {
        console.error('Error fetching videos list:', err);
    }
}

// Setup Selection Event Listeners
function setupUploadHandlers() {
    videoSelect.addEventListener('change', () => {
        if (videoSelect.value) {
            loadVideoDetails(videoSelect.value);
        } else {
            resetState();
        }
    });
}

// Upload file using XMLHttpRequest for progress logging
function uploadVideoFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    uploadStatus.classList.remove('hide');
    uploadStatus.className = 'status-msg info';
    uploadStatus.textContent = 'Starting upload...';

    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/upload', true);

    xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
            const percentComplete = Math.round((e.loaded / e.total) * 100);
            uploadStatus.textContent = `Uploading... ${percentComplete}%`;
        }
    };

    xhr.onload = async () => {
        if (xhr.status === 200) {
            const result = JSON.parse(xhr.responseText);
            uploadStatus.className = 'status-msg success';
            uploadStatus.textContent = 'Upload completed & frame extracted successfully!';
            
            await loadVideosList();
            videoSelect.value = result.video_name;
            setVideoState(result);
            
            setTimeout(() => { uploadStatus.classList.add('hide'); }, 4000);
        } else {
            uploadStatus.className = 'status-msg error';
            uploadStatus.textContent = `Upload failed: ${xhr.statusText}`;
        }
    };

    xhr.onerror = () => {
        uploadStatus.className = 'status-msg error';
        uploadStatus.textContent = 'Network error occurred during upload.';
    };

    xhr.send(formData);
}

// Get video metadata and first frame image path
async function loadVideoDetails(videoName) {
    try {
        const response = await fetch(`/api/video-info/${encodeURIComponent(videoName)}`);
        if (response.ok) {
            const result = await response.json();
            setVideoState(result);
        } else {
            console.error('Error loading video details:', response.statusText);
        }
    } catch (err) {
        console.error('Network error loading video details:', err);
    }
}

// Update app state and load baseline first frame image
function setVideoState(videoData) {
    state.videoName = videoData.video_name;
    state.videoWidth = videoData.width;
    state.videoHeight = videoData.height;
    state.fps = videoData.fps;
    state.frameCount = videoData.frame_count;
    state.duration = videoData.duration;

    currentVideoTitle.textContent = state.videoName;
    currentVideoMeta.textContent = `${state.videoWidth}x${state.videoHeight} | ${state.fps.toFixed(2)} FPS | ${state.frameCount} frames | ~${state.duration.toFixed(1)}s`;

    canvasInstructions.classList.add('hide');
    
    // Clear results
    resultsPanel.classList.add('hide');
    progressContainer.classList.add('hide');
    processStatus.classList.add('hide');
    saveStatus.classList.add('hide');
    if (pollingInterval) {
        clearInterval(pollingInterval);
    }
    
    const img = new Image();
    img.src = videoData.first_frame_url;
    img.onload = () => {
        state.firstFrameImg = img;
        
        // Scale and render the canvas inside the viewport
        renderCanvas();
        
        // Enable panels
        zoneConfigPanel.classList.remove('disabled');
        runPanel.classList.remove('disabled');
        
        // Check calibration status from backend
        checkCalibrationStatus();
    };
    img.onerror = () => {
        console.error('Failed to load first frame image:', videoData.first_frame_url);
    };
}

// Coordinate Scaling Helpers
// Converts coordinates from original video space back to displayed canvas size
function videoToCanvas(pt) {
    if (!canvas.width || !state.videoWidth) return [0, 0];
    return [
        pt[0] * (canvas.width / state.videoWidth),
        pt[1] * (canvas.height / state.videoHeight)
    ];
}

// Converts coordinates from displayed canvas size to original video space
function canvasToVideo(pt) {
    if (!canvas.width || !state.videoWidth) return [0, 0];
    return [
        pt[0] * (state.videoWidth / canvas.width),
        pt[1] * (state.videoHeight / canvas.height)
    ];
}

// Draws the image and the active danger zone boundary
function drawEditor() {
    if (!state.firstFrameImg) return;

    // 1. Draw frame image
    ctx.drawImage(state.firstFrameImg, 0, 0, canvas.width, canvas.height);
    
    if (!state.box) return;
    
    // 2. Draw danger zone filled overlay
    ctx.fillStyle = 'rgba(244, 63, 94, 0.22)'; // Semi-transparent rose red
    ctx.strokeStyle = '#f43f5e';                // Solid red border
    ctx.lineWidth = 3;
    
    ctx.beginPath();
    const startPt = videoToCanvas(state.box[0]);
    ctx.moveTo(startPt[0], startPt[1]);
    for (let i = 1; i < state.box.length; i++) {
        const pt = videoToCanvas(state.box[i]);
        ctx.lineTo(pt[0], pt[1]);
    }
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    
    // 3. Draw border dashed highlights if dragging/hovering
    if (isDragging) {
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
        ctx.lineWidth = 1;
        ctx.setLineDash([6, 4]);
        ctx.stroke();
        ctx.setLineDash([]);
    }
    
    // 4. Draw corner handles
    ctx.fillStyle = '#ffffff';
    ctx.strokeStyle = '#f43f5e';
    ctx.lineWidth = 2;
    
    state.box.forEach((pt, idx) => {
        const p = videoToCanvas(pt);
        ctx.beginPath();
        ctx.arc(p[0], p[1], HANDLE_RADIUS, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
    });
}

// Scales and renders canvas viewport
function renderCanvas() {
    if (!state.firstFrameImg) return;

    const viewport = document.getElementById('canvas-viewport');
    const viewportWidth = viewport.clientWidth - 24;
    const viewportHeight = viewport.clientHeight - 24;
    
    const videoAspect = state.videoWidth / state.videoHeight;
    const viewportAspect = viewportWidth / viewportHeight;
    
    let canvasWidth = viewportWidth;
    let canvasHeight = viewportHeight;
    
    if (videoAspect > viewportAspect) {
        canvasHeight = canvasWidth / videoAspect;
    } else {
        canvasWidth = canvasHeight * videoAspect;
    }
    
    canvas.width = canvasWidth;
    canvas.height = canvasHeight;
    
    // Explicitly set the CSS width and height of the canvas matching the viewport space
    canvas.style.width = `${canvasWidth}px`;
    canvas.style.height = `${canvasHeight}px`;
    
    drawEditor();
}

// Helper to compute distance from a point to a line segment
function getDistanceToSegment(p, a, b) {
    const dx = b[0] - a[0];
    const dy = b[1] - a[1];
    if (dx === 0 && dy === 0) return Math.hypot(p.x - a[0], p.y - a[1]);
    
    const t = ((p.x - a[0]) * dx + (p.y - a[1]) * dy) / (dx * dx + dy * dy);
    
    if (t < 0) return Math.hypot(p.x - a[0], p.y - a[1]);
    if (t > 1) return Math.hypot(p.x - b[0], p.y - b[1]);
    
    const projX = a[0] + t * dx;
    const projY = a[1] + t * dy;
    return Math.hypot(p.x - projX, p.y - projY);
}

// Set up interactive Canvas box editing mouse/touch behaviors
function setupCanvasInteractions() {
    
    function getMousePos(e) {
        const rect = canvas.getBoundingClientRect();
        return {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };
    }
    
    canvas.addEventListener('mousedown', (e) => {
        if (!state.box) return;
        
        const m = getMousePos(e);
        const mVideo = canvasToVideo([m.x, m.y]);
        
        // 1. Check corner handle hits
        for (let i = 0; i < state.box.length; i++) {
            const handleCanvas = videoToCanvas(state.box[i]);
            const dist = Math.hypot(m.x - handleCanvas[0], m.y - handleCanvas[1]);
            if (dist <= HANDLE_RADIUS + 4) {
                isDragging = true;
                dragMode = 'handle';
                dragIndex = i;
                dragStartPos = { x: m.x, y: m.y };
                boxStartCoords = JSON.parse(JSON.stringify(state.box));
                canvas.style.cursor = 'grabbing';
                return;
            }
        }
        
        // 2. Check edge hits using 2D segment distance
        const p = state.box.map(pt => videoToCanvas(pt));
        for (let i = 0; i < p.length; i++) {
            const next = (i + 1) % p.length;
            const dist = getDistanceToSegment(m, p[i], p[next]);
            if (dist <= EDGE_THRESHOLD) {
                startEdgeDrag(i, m);
                return;
            }
        }
        
        // 3. Check inside box click for translation (using bounding box limits of all points)
        let xMin = state.videoWidth, xMax = 0, yMin = state.videoHeight, yMax = 0;
        state.box.forEach(pt => {
            xMin = Math.min(xMin, pt[0]);
            xMax = Math.max(xMax, pt[0]);
            yMin = Math.min(yMin, pt[1]);
            yMax = Math.max(yMax, pt[1]);
        });
        
        if (mVideo[0] >= xMin && mVideo[0] <= xMax && mVideo[1] >= yMin && mVideo[1] <= yMax) {
            isDragging = true;
            dragMode = 'box';
            dragStartPos = { x: m.x, y: m.y };
            boxStartCoords = JSON.parse(JSON.stringify(state.box));
            canvas.style.cursor = 'move';
        }
    });
    
    function startEdgeDrag(idx, m) {
        isDragging = true;
        dragMode = 'edge';
        dragIndex = idx;
        dragStartPos = { x: m.x, y: m.y };
        boxStartCoords = JSON.parse(JSON.stringify(state.box));
        canvas.style.cursor = 'move';
    }
    
    canvas.addEventListener('mousemove', (e) => {
        if (!state.box) return;
        
        const m = getMousePos(e);
        
        if (!isDragging) {
            // Hover cursor styling
            let hover = false;
            // Check corners
            for (let i = 0; i < state.box.length; i++) {
                const handleCanvas = videoToCanvas(state.box[i]);
                const dist = Math.hypot(m.x - handleCanvas[0], m.y - handleCanvas[1]);
                if (dist <= HANDLE_RADIUS + 4) {
                    canvas.style.cursor = 'grab';
                    hover = true;
                    break;
                }
            }
            
            if (!hover) {
                // Check edges using segment distance
                const p = state.box.map(pt => videoToCanvas(pt));
                for (let i = 0; i < p.length; i++) {
                    const next = (i + 1) % p.length;
                    const dist = getDistanceToSegment(m, p[i], p[next]);
                    if (dist <= EDGE_THRESHOLD) {
                        canvas.style.cursor = 'move';
                        hover = true;
                        break;
                    }
                }
            }
            
            if (!hover) {
                // Check inside (using bounding box of all points)
                const mVideo = canvasToVideo([m.x, m.y]);
                let xMin = state.videoWidth, xMax = 0, yMin = state.videoHeight, yMax = 0;
                state.box.forEach(pt => {
                    xMin = Math.min(xMin, pt[0]);
                    xMax = Math.max(xMax, pt[0]);
                    yMin = Math.min(yMin, pt[1]);
                    yMax = Math.max(yMax, pt[1]);
                });
                
                if (mVideo[0] >= xMin && mVideo[0] <= xMax && mVideo[1] >= yMin && mVideo[1] <= yMax) {
                    canvas.style.cursor = 'pointer';
                    hover = true;
                }
            }
            
            if (!hover) {
                canvas.style.cursor = 'crosshair';
            }
            return;
        }
        
        // Calculate coordinate delta in video space
        const dCanvasX = m.x - dragStartPos.x;
        const dCanvasY = m.y - dragStartPos.y;
        
        const scaleX = state.videoWidth / canvas.width;
        const scaleY = state.videoHeight / canvas.height;
        
        const dVideoX = dCanvasX * scaleX;
        const dVideoY = dCanvasY * scaleY;
        
        let newBox = JSON.parse(JSON.stringify(boxStartCoords));
        
        if (dragMode === 'handle') {
            // Dragging a specific corner. Updates only that corner to allow parallelograms and diagonal boundaries.
            const idx = dragIndex;
            const targetX = boxStartCoords[idx][0] + dVideoX;
            const targetY = boxStartCoords[idx][1] + dVideoY;
            
            // Constrain inside video resolution boundaries
            const clampedX = Math.max(0, Math.min(state.videoWidth - 1, targetX));
            const clampedY = Math.max(0, Math.min(state.videoHeight - 1, targetY));
            
            newBox[idx] = [clampedX, clampedY];
        }
        else if (dragMode === 'edge') {
            // Dragging an edge. Shift both endpoints of the edge by dVideoX and dVideoY.
            const idx = dragIndex;
            const nextIdx = (idx + 1) % state.box.length;
            const ptsToShift = [idx, nextIdx];
            
            ptsToShift.forEach(pIdx => {
                newBox[pIdx][0] = Math.max(0, Math.min(state.videoWidth - 1, boxStartCoords[pIdx][0] + dVideoX));
                newBox[pIdx][1] = Math.max(0, Math.min(state.videoHeight - 1, boxStartCoords[pIdx][1] + dVideoY));
            });
        }
        else if (dragMode === 'box') {
            // Translate the entire box
            // Calculate limits to prevent shifting box out of boundaries
            let minX = state.videoWidth, maxX = 0, minY = state.videoHeight, maxY = 0;
            boxStartCoords.forEach(pt => {
                minX = Math.min(minX, pt[0]);
                maxX = Math.max(maxX, pt[0]);
                minY = Math.min(minY, pt[1]);
                maxY = Math.max(maxY, pt[1]);
            });
            
            // Constrain deltas
            let shiftX = dVideoX;
            let shiftY = dVideoY;
            
            if (minX + shiftX < 0) shiftX = -minX;
            if (maxX + shiftX >= state.videoWidth) shiftX = state.videoWidth - 1 - maxX;
            if (minY + shiftY < 0) shiftY = -minY;
            if (maxY + shiftY >= state.videoHeight) shiftY = state.videoHeight - 1 - maxY;
            
            newBox = boxStartCoords.map(pt => [pt[0] + shiftX, pt[1] + shiftY]);
        }
        
        // Check minimum size to prevent the box from collapsing or turning inside out
        const width = Math.abs(newBox[1][0] - newBox[0][0]);
        const height = Math.abs(newBox[3][1] - newBox[0][1]);
        
        if (width > 20 && height > 20) {
            state.box = newBox;
            drawEditor();
        }
    });
    
    window.addEventListener('mouseup', () => {
        if (isDragging) {
            isDragging = false;
            dragMode = null;
            dragIndex = -1;
            canvas.style.cursor = 'crosshair';
            drawEditor();
        }
    });
}

// Create a default centered pentagonal bounding box (5 vertices)
function setDefaultBox() {
    const w = state.videoWidth;
    const h = state.videoHeight;
    
    state.box = [
        [w * 0.5, h * 0.25],  // Top Peak
        [w * 0.75, h * 0.45], // Top-Right
        [w * 0.65, h * 0.75], // Bottom-Right
        [w * 0.35, h * 0.75], // Bottom-Left
        [w * 0.25, h * 0.45]  // Top-Left
    ];
    drawEditor();
}

// Call backend yellow floor markings detector
async function runFloorMarkingsDetector() {
    if (!state.videoName) return;
    
    btnSuggest.disabled = true;
    btnSuggest.textContent = 'Analyzing...';
    
    const margin = marginOffset.value;
    
    try {
        const response = await fetch(`/api/suggest-zone/${encodeURIComponent(state.videoName)}?margin_px=${margin}`);
        if (response.ok) {
            const data = await response.json();
            if (data.suggested_coordinates) {
                let coords = data.suggested_coordinates;
                if (coords.length === 4) {
                    const p0 = coords[0];
                    const p1 = coords[1];
                    const mid = [(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2];
                    coords = [p0, mid, p1, coords[2], coords[3]];
                }
                state.box = coords;
                drawEditor();
                showSaveStatus("Yellow floor marking boundary suggested successfully!", "success");
            } else {
                showSaveStatus("No yellow markings found. Fallback to default center box.", "error");
                setDefaultBox();
            }
        } else {
            showSaveStatus("Server error during suggestion scan.", "error");
            setDefaultBox();
        }
    } catch (err) {
        showSaveStatus(`Connection error: ${err.message}`, "error");
        setDefaultBox();
    } finally {
        btnSuggest.disabled = false;
        btnSuggest.textContent = 'Auto-Detect Floor Lines';
    }
}

// Save calibrated coordinates and zone name
async function saveCalibration() {
    if (!state.videoName) return;
    if (!state.box) {
        showSaveStatus("Please draw or calibrate a danger zone first.", "error");
        return;
    }
    
    const zoneNameInput = document.getElementById('zone-name');
    const zoneName = zoneNameInput.value.trim();
    if (!zoneName) {
        showSaveStatus("Danger zone name is required.", "error");
        return;
    }
    
    btnSave.disabled = true;
    showSaveStatus("Saving...", "info");
    
    try {
        const response = await fetch('/api/save-zone', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                video_name: state.videoName,
                zone_name: zoneName,
                coordinates: state.box
            })
        });
        
        if (response.ok) {
            showSaveStatus("Zone saved successfully!", "success");
            calibrationBadge.className = 'badge badge-active';
            calibrationBadge.textContent = 'Calibrated';
        } else {
            const err = await response.json();
            showSaveStatus(`Failed to save: ${err.detail || response.statusText}`, "error");
        }
    } catch (err) {
        showSaveStatus(`Save connection failed: ${err.message}`, "error");
    } finally {
        btnSave.disabled = false;
    }
}

// Triggers the YOLO processing pipeline
async function startVideoProcessing() {
    if (!state.videoName) return;
    
    btnProcess.disabled = true;
    processStatus.classList.remove('hide');
    processStatus.className = 'status-msg info';
    processStatus.textContent = 'Queuing processing task...';
    
    progressContainer.classList.remove('hide');
    progressBarFill.style.width = '0%';
    progressPercent.textContent = '0%';
    resultsPanel.classList.add('hide');
    
    try {
        const response = await fetch(`/api/process-video?video_name=${encodeURIComponent(state.videoName)}`, {
            method: 'POST'
        });
        
        if (response.ok) {
            processStatus.textContent = 'YOLO tracking pipeline started in background. Processing frames...';
            
            // Start Polling for updates
            if (pollingInterval) clearInterval(pollingInterval);
            pollingInterval = setInterval(pollProcessingStatus, 1000);
        } else {
            const err = await response.json();
            processStatus.className = 'status-msg error';
            processStatus.textContent = `Start failed: ${err.detail || response.statusText}`;
            btnProcess.disabled = false;
        }
    } catch (err) {
        processStatus.className = 'status-msg error';
        processStatus.textContent = `Failed to connect: ${err.message}`;
        btnProcess.disabled = false;
    }
}

// Polls processing status and updates UI
async function pollProcessingStatus() {
    try {
        const response = await fetch(`/api/status/${encodeURIComponent(state.videoName)}`);
        if (!response.ok) return;
        
        const data = await response.json();
        
        if (data.status === 'processing') {
            const percent = data.progress || 0;
            progressBarFill.style.width = `${percent}%`;
            progressPercent.textContent = `${percent}%`;
            processingBadge.className = 'badge badge-alert';
            processingBadge.textContent = 'Processing';
        } 
        else if (data.status === 'completed') {
            clearInterval(pollingInterval);
            progressBarFill.style.width = '100%';
            progressPercent.textContent = '100%';
            btnProcess.disabled = false;
            
            processingBadge.className = 'badge badge-active';
            processingBadge.textContent = 'Processed';
            
            processStatus.className = 'status-msg success';
            processStatus.textContent = 'Processing successfully finished!';
            
            // Set results resources
            btnViewVideo.href = data.output_video;
            btnViewLog.href = data.log_file;
            
            // Fetch event log content to populate statistics summary card if possible
            fetchEventStatistics(data.log_file);
        } 
        else if (data.status === 'failed') {
            clearInterval(pollingInterval);
            btnProcess.disabled = false;
            processingBadge.className = 'badge badge-alert';
            processingBadge.textContent = 'Failed';
            processStatus.className = 'status-msg error';
            processStatus.textContent = `Pipeline failed: ${data.error || 'Unknown error'}`;
        }
    } catch (err) {
        console.error('Polling status error:', err);
    }
}

// Load statistics counts from event log JSON file
async function fetchEventStatistics(logFileUrl) {
    try {
        const response = await fetch(logFileUrl);
        if (response.ok) {
            const logData = await response.json();
            
            // Extract statistics from JSON structure
            const summary = logData.summary || {};
            const events = logData.events || [];
            
            sumIntrusions.textContent = summary.total_intrusions !== undefined ? summary.total_intrusions : events.length;
            sumTime.textContent = summary.total_intrusion_time_sec !== undefined ? `${summary.total_intrusion_time_sec.toFixed(1)}s` : '-';
            sumPeople.textContent = summary.unique_people_count !== undefined ? summary.unique_people_count : '-';
            
            resultsPanel.classList.remove('hide');
        }
    } catch (err) {
        console.error('Failed to parse event log for summary:', err);
        // Fallback display
        sumIntrusions.textContent = 'Check';
        sumTime.textContent = 'Logs';
        sumPeople.textContent = 'File';
        resultsPanel.classList.remove('hide');
    }
}

// Check saved config files
async function checkCalibrationStatus() {
    try {
        const response = await fetch(`/api/load-zone/${encodeURIComponent(state.videoName)}`);
        if (response.status === 200) {
            const data = await response.json();
            calibrationBadge.className = 'badge badge-active';
            calibrationBadge.textContent = 'Calibrated';
            document.getElementById('zone-name').value = data.zone_name;
            
            // Load the coordinates saved
            let coords = data.coordinates;
            if (coords && coords.length === 4) {
                const p0 = coords[0];
                const p1 = coords[1];
                const mid = [(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2];
                coords = [p0, mid, p1, coords[2], coords[3]];
            }
            state.box = coords;
            drawEditor();
        } else {
            calibrationBadge.className = 'badge badge-inactive';
            calibrationBadge.textContent = 'Uncalibrated';
            
            // Use centered box by default
            setDefaultBox();
        }
        
        checkProcessingStatus();
    } catch (err) {
        console.error('Error loading calibration status:', err);
        setDefaultBox();
    }
}

// Checks processing pipeline status
async function checkProcessingStatus() {
    try {
        const response = await fetch(`/api/status/${encodeURIComponent(state.videoName)}`);
        const status = await response.json();
        
        if (status.status === 'completed') {
            processingBadge.className = 'badge badge-active';
            processingBadge.textContent = 'Processed';
            
            btnViewVideo.href = status.output_video;
            btnViewLog.href = status.log_file;
            fetchEventStatistics(status.log_file);
        } else if (status.status === 'processing') {
            processingBadge.className = 'badge badge-alert';
            processingBadge.textContent = 'Processing';
            
            progressContainer.classList.remove('hide');
            progressBarFill.style.width = `${status.progress}%`;
            progressPercent.textContent = `${status.progress}%`;
            
            // Resume polling
            if (pollingInterval) clearInterval(pollingInterval);
            pollingInterval = setInterval(pollProcessingStatus, 1000);
        } else {
            processingBadge.className = 'badge badge-inactive';
            processingBadge.textContent = 'Not Processed';
        }
    } catch (err) {
        console.error('Error checking processing status:', err);
    }
}

// Show save statuses
function showSaveStatus(msg, type) {
    saveStatus.classList.remove('hide');
    saveStatus.className = `status-msg ${type}`;
    saveStatus.textContent = msg;
    
    if (type === 'success') {
        setTimeout(() => { saveStatus.classList.add('hide'); }, 4000);
    }
}

// Reset state
function resetState() {
    state = {
        videoName: null,
        videoWidth: 0,
        videoHeight: 0,
        fps: 0,
        frameCount: 0,
        duration: 0,
        firstFrameImg: null,
        box: null
    };

    currentVideoTitle.textContent = 'No Video Loaded';
    currentVideoMeta.textContent = 'Please upload or select a video to begin calibration.';
    
    calibrationBadge.className = 'badge badge-inactive';
    calibrationBadge.textContent = 'Uncalibrated';
    
    processingBadge.className = 'badge badge-inactive';
    processingBadge.textContent = 'Not Processed';
    
    canvasInstructions.classList.remove('hide');
    zoneConfigPanel.classList.add('disabled');
    runPanel.classList.add('disabled');
    resultsPanel.classList.add('hide');
    progressContainer.classList.add('hide');
    processStatus.classList.add('hide');
    
    if (pollingInterval) {
        clearInterval(pollingInterval);
    }
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// Resize handlers
window.addEventListener('resize', () => {
    if (state.firstFrameImg) {
        renderCanvas();
    }
});
