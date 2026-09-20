import os
from PIL import Image
from agents.detection_agent import DetectionAgent

def debug_image(image_path):
    print("=" * 50)
    print(f"IMAGE: {os.path.basename(image_path)}")
    
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return

    pil_img = Image.open(image_path).convert("RGB")
    print(f"IMAGE SIZE: {pil_img.size[0]} x {pil_img.size[1]}")
    
    agent = DetectionAgent()
    print(f"DETECTOR MODEL: yolov8n.pt")
    
    # We must introspect the model classes
    if hasattr(agent.model, 'names'):
        classes = agent.model.names
        print(f"MODEL CLASSES (First 20): {dict(list(classes.items())[:20])}")
        
        # Check if 'snake' is explicitly in the names
        has_snake = any("snake" in v.lower() for v in classes.values())
        print(f"CONTAINS 'SNAKE' CLASS? {has_snake}")
    else:
        print("MODEL CLASSES: Unknown (no names attribute)")

    # Run detection
    print("\nRunning detection...")
    results = agent.model(pil_img, verbose=False)
    
    raw_count = len(results[0].boxes) if len(results) > 0 else 0
    print(f"RAW DETECTIONS: {raw_count}")
    
    for i, box in enumerate(results[0].boxes):
        conf = float(box.conf[0].item())
        cls_id = int(box.cls[0].item())
        cls_name = classes[cls_id] if hasattr(agent.model, 'names') else str(cls_id)
        xyxy = box.xyxy[0].cpu().numpy().tolist()
        print(f"  Box {i}: Class='{cls_name}' (ID={cls_id}), Conf={conf:.4f}, xyxy={xyxy}")

    print(f"\nCONFIDENCE THRESHOLD (Python hardcoded): 0.3")
    
    cropped, result = agent.detect_and_crop(pil_img)
    print("\nFINAL:")
    print(f"snake_detected: {result['snake_detected']}")
    print(f"confidence: {result['confidence']}")
    print(f"bounding_box: {result['bounding_box']}")

if __name__ == "__main__":
    dataset_files = os.listdir("snake_dataset")
    test_files = [os.path.join("snake_dataset", f) for f in dataset_files if f.endswith(".jpg")][:3]
    
    # explicitly test the image that caused issues
    if os.path.exists("snake_dataset/COBRA (131).jpg"):
        test_files.insert(0, "snake_dataset/COBRA (131).jpg")
    
    for img in test_files:
        debug_image(img)
