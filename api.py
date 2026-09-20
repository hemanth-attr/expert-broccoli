import io
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from agents.coordinator import AgentCoordinator
import json
import base64
import os

app = FastAPI(title="Autonomous Venomous Snake Identification & Advisory API")

# Mount static directory for frontend assets
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the main frontend page
@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")

# Initialize the Agent Coordinator
coordinator = AgentCoordinator()

@app.post("/predict")
async def predict_specimen(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Delegate the entire workflow to the Coordinator
    response_data = coordinator.process_image(pil_img)
    return JSONResponse(response_data)

@app.websocket("/stream")
async def stream_video(websocket: WebSocket):
    await websocket.accept()
    try:
        frame_count = 0
        while True:
            # Receive frame as base64 string
            data = await websocket.receive_text()
            frame_count += 1
            
            # Skip frames to simulate real-time optimization
            if frame_count % 3 != 0:
                await websocket.send_json({"status": "skipped"})
                continue

            try:
                # Decode base64 to image
                img_data = base64.b64decode(data)
                pil_img = Image.open(io.BytesIO(img_data)).convert("RGB")
                
                # Process frame
                # In a real system, you might turn off XAI for speed
                response_data = coordinator.process_image(pil_img)
                
                # Send result back
                await websocket.send_json(response_data)
            except Exception as e:
                await websocket.send_json({"error": str(e)})

    except WebSocketDisconnect:
        print("Client disconnected from video stream.")