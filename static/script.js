document.addEventListener('DOMContentLoaded', () => {
    // Views
    const uploadView = document.getElementById('upload-view');
    const previewView = document.getElementById('preview-view');
    const loadingView = document.getElementById('loading-view');
    const resultsView = document.getElementById('results-view');
    
    // Upload Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const btnChooseFile = document.getElementById('btn-choose-file');
    const imagePreview = document.getElementById('image-preview');
    const fileName = document.getElementById('file-name');
    
    // Action Elements
    const btnRemove = document.getElementById('btn-remove');
    const btnAnalyze = document.getElementById('btn-analyze');
    const btnReset = document.getElementById('btn-reset');
    
    // Error Elements
    const errorContainer = document.getElementById('error-container');
    const errorMessage = document.getElementById('error-message');
    
    // Debug
    const btnDebugToggle = document.getElementById('btn-debug-toggle');
    const debugContent = document.getElementById('debug-content');
    const jsonOutput = document.getElementById('json-output');

    let selectedFile = null;

    // --- Upload Handlers ---
    
    btnChooseFile.addEventListener('click', () => fileInput.click());
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    function handleFileSelect(file) {
        if (!file.type.startsWith('image/')) {
            showError("Invalid image", "Please upload a JPG, JPEG, or PNG image.");
            return;
        }
        
        selectedFile = file;
        fileName.textContent = file.name;
        
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            // Also set it for the results view immediately
            document.getElementById('result-image').src = e.target.result;
            document.getElementById('xai-img-orig').src = e.target.result;
            document.getElementById('xai-img-orig-side').src = e.target.result;
            switchView(previewView);
        };
        reader.readAsDataURL(file);
        hideError();
    }

    btnRemove.addEventListener('click', () => {
        selectedFile = null;
        fileInput.value = '';
        switchView(uploadView);
    });

    // --- API Integration ---

    btnAnalyze.addEventListener('click', async () => {
        if (!selectedFile) return;
        
        switchView(loadingView);
        hideError();
        
        const formData = new FormData();
        formData.append("file", selectedFile);
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || "API Request Failed");
            }
            
            const data = await response.json();
            populateDashboard(data);
            switchView(resultsView);
            
        } catch (error) {
            showError("Analysis failed", "Unable to connect to Snake AI backend or process image. " + error.message);
            switchView(previewView);
        }
    });

    btnReset.addEventListener('click', () => {
        selectedFile = null;
        fileInput.value = '';
        switchView(uploadView);
        // Hide bounding box
        document.getElementById('bounding-box').classList.add('hidden');
    });

    // --- Dashboard Population ---
    
    function populateDashboard(data) {
        // Debug JSON
        jsonOutput.textContent = JSON.stringify(data, null, 2);
        
        const mainGrid = document.getElementById('main-results-grid');
        const noSnakeState = document.getElementById('no-snake-state');
        const xaiSection = document.getElementById('xai-main-section');
        
        // 1. Detection & Verification
        const det = data.detection;
        const warning = document.getElementById('classification-only-warning');
        
        const verifyStatus = document.getElementById('verification-status');
        const verifyReason = document.getElementById('verification-reason');
        
        if (det.mode === "verification_rejection") {
            // OOD REJECTION STATE
            mainGrid.classList.add('hidden');
            xaiSection.classList.add('hidden');
            noSnakeState.classList.remove('hidden');
            noSnakeState.style.display = 'block';
            
            // Re-use No Snake State but update text
            noSnakeState.querySelector('h2').textContent = "NO SNAKE DETECTED";
            noSnakeState.querySelector('p').textContent = det.reason || "The system could not verify a snake in this image.";
            
            populateAdvisory(data.advisory);
            return;
        }
        
        if (det.snake_detected === false) {
            // EXPLICIT NO SNAKE STATE (From YOLO)
            mainGrid.classList.add('hidden');
            xaiSection.classList.add('hidden');
            noSnakeState.classList.remove('hidden');
            noSnakeState.style.display = 'block';
            
            populateAdvisory(data.advisory);
            return;
        }
        
        // SNAKE DETECTED OR CLASSIFICATION ONLY STATE
        mainGrid.classList.remove('hidden');
        noSnakeState.classList.add('hidden');
        noSnakeState.style.display = 'none';
        
        // Populate Verification
        verifyStatus.textContent = det.status.replace(/_/g, " ");
        verifyStatus.style.color = "var(--success-color)";
        if (data.verification_metrics) {
            verifyReason.textContent = `MSP: ${data.verification_metrics.msp} | Entropy: ${data.verification_metrics.entropy} | Margin: ${data.verification_metrics.margin}`;
        }
        if (det.mode === "classification_only") {
            warning.classList.remove('hidden');
            document.getElementById('bounding-box').classList.add('hidden');
        } else {
            warning.classList.add('hidden');
            drawBoundingBox(det.bounding_box);
        }

        // 2. Species
        const sp = data.species;
        if (sp.status === "not_evaluated" || !sp.top_prediction) {
            document.getElementById('top-species').textContent = "Not Evaluated";
            document.getElementById('top-conf').textContent = "--%";
            document.getElementById('top-conf-bar').style.width = "0%";
            document.getElementById('alt-list').innerHTML = "";
        } else {
            document.getElementById('top-species').textContent = sp.top_prediction;
            const topConf = (sp.confidence * 100).toFixed(2);
            document.getElementById('top-conf').textContent = `${topConf}%`;
            document.getElementById('top-conf-bar').style.width = `${topConf}%`;
            
            const altList = document.getElementById('alt-list');
            altList.innerHTML = '';
            sp.alternatives.forEach(alt => {
                const li = document.createElement('li');
                li.innerHTML = `<strong>${alt.species}</strong> - ${(alt.confidence * 100).toFixed(2)}%`;
                altList.appendChild(li);
            });
        }

        // 3. Venom Status
        const ven = data.venom;
        const venomText = document.getElementById('venom-status-text');
        venomText.className = ""; // reset
        if (ven.status === "venomous") {
            venomText.textContent = "☠ VENOMOUS";
            venomText.classList.add("venomous-text");
        } else if (ven.status === "non-venomous") {
            venomText.textContent = "✅ NON-VENOMOUS";
            venomText.classList.add("non-venomous-text");
        } else {
            venomText.textContent = "⚠ UNKNOWN";
            venomText.classList.add("unknown-text");
        }
        document.getElementById('venom-source').textContent = ven.source || "None";

        // 4. Uncertainty
        const unc = data.uncertainty;
        document.getElementById('uncertainty-level').textContent = unc.level.toUpperCase();
        document.getElementById('advisory-allowed-text').textContent = `Advisory allowed: ${unc.advisory_allowed ? 'YES' : 'NO'}`;

        // 5. XAI Grad-CAM
        if (data.xai && data.xai.heatmap && data.xai.method !== "disabled") {
            xaiSection.classList.remove('hidden');
            const heatmapSrc = `data:image/jpeg;base64,${data.xai.heatmap}`;
            document.getElementById('xai-img-heatmap').src = heatmapSrc;
            document.getElementById('xai-img-heatmap-side').src = heatmapSrc;
        } else {
            xaiSection.classList.add('hidden');
        }

        // 6. Advisory
        populateAdvisory(data.advisory);
    }
    
    function populateAdvisory(adv) {
        const advSection = document.getElementById('advisory-section');
        if (adv && adv.available) {
            advSection.classList.remove('hidden');
            document.getElementById('advisory-type').textContent = `🚨 ${adv.type.replace(/_/g, ' ')}`;
            document.getElementById('advisory-message').textContent = adv.message;
            
            const advBox = document.getElementById('advisory-box');
            advBox.className = "advisory-box"; // reset
            if (adv.type.includes("EMERGENCY")) advBox.classList.add("emergency");
            else advBox.classList.add("cautionary");
        } else if (adv && !adv.available && adv.type === "INSUFFICIENT_EVIDENCE") {
            advSection.classList.remove('hidden');
            document.getElementById('advisory-type').textContent = `⚠ ${adv.type.replace(/_/g, ' ')}`;
            document.getElementById('advisory-message').textContent = adv.message;
            const advBox = document.getElementById('advisory-box');
            advBox.className = "advisory-box cautionary";
        } else {
            advSection.classList.add('hidden');
        }
    }
    
    // --- Bounding Box Logic ---
    function drawBoundingBox(bbox) {
        const bboxDiv = document.getElementById('bounding-box');
        if (!bbox || bbox.length !== 4) {
            bboxDiv.classList.add('hidden');
            return;
        }
        
        const img = document.getElementById('result-image');
        
        // We must wait for the image to be fully rendered to get its display dimensions
        // But it should be already loaded from the FileReader. We use setTimeout to ensure layout is done.
        setTimeout(() => {
            const displayWidth = img.clientWidth;
            const displayHeight = img.clientHeight;
            const naturalWidth = img.naturalWidth;
            const naturalHeight = img.naturalHeight;
            
            // Calculate scale factors
            const scaleX = displayWidth / naturalWidth;
            const scaleY = displayHeight / naturalHeight;
            
            // bbox = [x1, y1, x2, y2]
            const [x1, y1, x2, y2] = bbox;
            
            const boxLeft = x1 * scaleX;
            const boxTop = y1 * scaleY;
            const boxWidth = (x2 - x1) * scaleX;
            const boxHeight = (y2 - y1) * scaleY;
            
            bboxDiv.style.left = `${boxLeft}px`;
            bboxDiv.style.top = `${boxTop}px`;
            bboxDiv.style.width = `${boxWidth}px`;
            bboxDiv.style.height = `${boxHeight}px`;
            
            bboxDiv.classList.remove('hidden');
        }, 100);
    }
    
    // Handle window resize for bounding box
    window.addEventListener('resize', () => {
        if (resultsView.classList.contains('active')) {
            try {
                const data = JSON.parse(jsonOutput.textContent);
                drawBoundingBox(data.detection.bounding_box);
            } catch(e) {}
        }
    });

    // --- UI Helpers ---

    function switchView(view) {
        [uploadView, previewView, loadingView, resultsView].forEach(v => v.classList.remove('active'));
        [uploadView, previewView, loadingView, resultsView].forEach(v => v.classList.add('hidden'));
        view.classList.remove('hidden');
        view.classList.add('active');
    }

    function showError(title, msg) {
        errorContainer.classList.remove('hidden');
        document.getElementById('error-title').textContent = title;
        errorMessage.textContent = msg;
    }

    function hideError() {
        errorContainer.classList.add('hidden');
    }

    // XAI Tabs
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.xai-pane');
    
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));
            
            btn.classList.add('active');
            document.getElementById(btn.dataset.target).classList.add('active');
        });
    });

    // Debug Panel
    btnDebugToggle.addEventListener('click', () => {
        debugContent.classList.toggle('hidden');
    });
});
