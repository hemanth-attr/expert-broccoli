import pytest
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_predict_no_file():
    response = client.post("/predict")
    assert response.status_code == 422 # Unprocessable Entity (missing file)

def test_predict_invalid_file_type():
    # Send a text file instead of an image
    files = {'file': ('test.txt', b'this is not an image', 'text/plain')}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "must be an image" in response.json()["detail"]

def test_predict_corrupt_image():
    # Send corrupted image bytes
    files = {'file': ('corrupt.jpg', b'corrupted_bytes_that_are_not_image', 'image/jpeg')}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "Invalid image file" in response.json()["detail"]

def test_websocket_stream_rejects_invalid_base64():
    with client.websocket_connect("/stream") as websocket:
        # Send 3 frames because the server skips frames 1 and 2
        websocket.send_text("this_is_not_valid_base64_image")
        websocket.receive_json() # skipped
        
        websocket.send_text("this_is_not_valid_base64_image")
        websocket.receive_json() # skipped
        
        websocket.send_text("this_is_not_valid_base64_image")
        data = websocket.receive_json() # This one is processed
        
        assert "error" in data
        assert "status" not in data # The first frame should error out, not skip
