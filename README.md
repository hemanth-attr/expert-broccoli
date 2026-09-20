# 🐍 Multi-Agent Explainable Real-Time Snake Identification & Emergency Decision-Support System

## 📖 Project Overview

This project is a **research-grade, AI-powered computer vision system** designed to identify snake species from images and video streams. It acts as an **Emergency Decision-Support System**, utilizing a multi-agent architecture to identify snake species, assess uncertainty, determine venomous status, generate Explainable AI (XAI) heatmaps, and provide strictly controlled medical advisories. 

Unlike traditional black-box classifiers, this system is designed for **safety, explainability, and uncertainty awareness**, making it robust against out-of-distribution data (e.g., classifying a random object as a snake) and providing visual interpretability to build trust in its predictions.

---

## 🏗️ System Architecture

The project has transitioned from a monolithic prototype into a modular **Multi-Agent System** orchestrated by a central Coordinator. 

### Core Agents

1. **Detection Agent (`agents/detection_agent.py`)**
   - **Role:** Locates the main subject in the image.
   - **Current Implementation:** Uses a lightweight `YOLOv8n` model. Because standard COCO does not include a "snake" class, this agent is currently marked as `experimental` to avoid miscropping the background and degrading classification accuracy.
   - **Output:** Bounding box coordinates and a detection confidence score.

2. **Species Agent (`agents/species_agent.py`)**
   - **Role:** Performs the core visual classification.
   - **Current Implementation:** Utilizes a `MobileNetV3 Small` architecture built in PyTorch (`mobilenet_snake_classifier_v2.pth`). 
   - **Output:** The Top-1 prediction and a list of Top-K alternative predictions with raw confidence scores, mapped to 17 distinct snake classes via `models/class_mapping.json`.

3. **Venom Agent (`agents/venom_agent.py`)**
   - **Role:** Translates the predicted species into a clinical venomous status.
   - **Current Implementation:** Eliminates unsafe string-matching by strictly cross-referencing the predicted species against a verified knowledge base (`advisory/knowledge_base.json`).
   - **Output:** A strict boolean/status (`venomous`, `non-venomous`, or `unknown`).

4. **Advisory Agent (`agents/advisory_agent.py`)**
   - **Role:** The critical safety layer. It issues medical guidance based on the venom status and the model's confidence.
   - **Safety Mechanism:** It does **not** use generative AI to invent medical advice. It rigidly fetches predefined, clinician-safe advisories (e.g., `EMERGENCY_MEDICAL_ADVISORY` or `CAUTIONARY_ADVISORY`) from the knowledge base.

5. **XAI Agent (`agents/xai_agent.py`)**
   - **Role:** Provides visual interpretability.
   - **Current Implementation:** Utilizes `pytorch-grad-cam` to extract class activation maps from the final convolutional layer of MobileNetV3. 
   - **Output:** A base64-encoded heatmap overlay showing exactly which pixels influenced the model's decision.

6. **Agent Coordinator (`agents/coordinator.py`)**
   - **Role:** The central nervous system. It orchestrates the synchronous data flow between all agents.
   - **Uncertainty Layer:** The Coordinator evaluates the confidence of the Species Agent. If the confidence is below 40%, it safely overrides the prediction to `UNKNOWN`, disables the detection flag, and forces the Advisory Agent to return an `INSUFFICIENT_EVIDENCE` warning rather than guessing a snake species.

---

## 🔄 End-to-End Data Flow

1. **Input**: An image or base64 frame is received via the API.
2. **Detection**: The Detection Agent analyzes the image for bounding boxes (currently experimental).
3. **Classification**: The Species Agent analyzes the raw image, returning the Top-3 classes and probabilities.
4. **Uncertainty Check**: The Coordinator checks if the Top-1 probability exceeds the safety threshold (`>= 0.40`). If not, the species is forced to `UNKNOWN`.
5. **Venom Lookup**: The Venom Agent checks the `knowledge_base.json` to map the species to a venomous status.
6. **XAI Generation**: The XAI Agent generates a Grad-CAM heatmap based on the specific classification tensor.
7. **Advisory Dispatch**: The Advisory Agent uses the Uncertainty Level and Venom Status to dispatch a strict medical response.
8. **Output**: The Coordinator packages all agent responses into a structured, unified JSON schema.

---

## 🚀 API Endpoints

The system is served using a high-performance **FastAPI** server (`api.py`).

### 1. Synchronous Inference (`POST /predict`)
Accepts `multipart/form-data` image uploads and returns a detailed JSON response representing the full pipeline.

**Example Request:**
```bash
curl -X POST "http://localhost:8000/predict" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@snake_dataset/COBRA (131).jpg"
```

### 2. Real-Time Video Streaming (`WS /stream`)
A WebSocket endpoint designed to ingest a continuous stream of base64-encoded frames. It features automatic **frame skipping** (processing every 3rd frame) to optimize for low latency and high FPS, simulating real-world edge deployment.

---

## 🔬 Quantitative Evaluation Framework

To validate the system for research defense, a massive evaluation suite is located in the `evaluation/` directory.

### Available Scripts
- **`evaluate_pipeline.py`**: Streams an annotated dataset through the Coordinator. Uses `scikit-learn` to calculate Top-1 Accuracy, Top-3 Accuracy, Weighted F1-Score, Venom Accuracy, and simulates Expected Calibration Error (ECE) thresholds.
- **`ablation_study.py`**: Programmatically isolates individual agents (e.g., Baseline MobileNetV3 vs. MobileNetV3 + Uncertainty) to measure the exact performance and latency delta introduced by each component.
- **`evaluate_latency.py`**: Benchmarks the pipeline over 50 iterations, reporting Average Latency, Median Latency, P95, P99, and Estimated FPS, broken down specifically by agent (Detection vs Classification vs XAI).

### Automated API Testing
- **`tests/test_api.py`**: Uses `pytest` to hit the FastAPI endpoints with corrupted images, invalid mimetypes, and malformed WebSocket streams to ensure graceful failure.

---

## ⚠️ Known Limitations & Research Context

When framing this system for research, the following limitations must be acknowledged:

1. **Dataset Bias & Coverage:** The model is trained on 17 distinct classes. It is geographically limited and may not perform accurately on species outside this distribution.
2. **Detection Limitations:** Because standard YOLO weights lack a snake class, the Detection Agent acts merely as a generic object cropper. In real-world deployment, this must be swapped for a YOLO model fine-tuned on a bounding-box snake dataset.
3. **No Clinical Validation:** The advisories in `knowledge_base.json` are meant for system demonstration and decision-support modeling. They have not been validated by toxicologists or medical boards and should not be used in a live clinical setting.
4. **XAI Interpretability:** Grad-CAM heatmaps highlight pixels that influence the model, but they do not prove that the model understands biological taxonomy. If a heatmap highlights the background instead of the snake's scales, it indicates the model is relying on spurious correlations.

---

## ⚙️ How to Run Locally

### 1. Install Dependencies
```bash
pip install torch torchvision fastapi uvicorn python-multipart opencv-python pytorch-grad-cam ultralytics scikit-learn pytest
```

### 2. Start the Server
```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Run the Evaluation Suite
*(Note: Requires a populated dataset and `annotations.json` in `snake_dataset/`)*
```bash
python -m evaluation.evaluate_pipeline
python -m evaluation.ablation_study
python -m evaluation.evaluate_latency
```

### 4. Run Automated Tests
```bash
pytest tests/
```
