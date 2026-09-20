from ultralytics import YOLO
import numpy as np

class DetectionAgent:
    def __init__(self, model_name="yolov8n.pt"):
        # We use a lightweight YOLOv8 model. 
        # Note: Standard COCO does not have a explicit "snake" class, 
        # so this is a generic object detector used to find the main subject.
        # In a production system, this should be fine-tuned on a snake dataset.
        self.model = YOLO(model_name)

    def detect_and_crop(self, pil_img):
        """
        Uses YOLOv8 as a negative filter for OOD rejection.
        Since it lacks a snake class, we don't use it for positive snake detection.
        However, if it detects a 'cat' or 'person', we can use that to reject the image.
        """
        results = self.model(pil_img, verbose=False)
        
        detected_classes = []
        best_conf = 0.0
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                class_name = self.model.names[cls_id]
                detected_classes.append({"class": class_name, "confidence": conf})
                if conf > best_conf:
                    best_conf = conf

        return pil_img, {
            "snake_detected": None, # Cannot confirm snake presence
            "confidence": best_conf if detected_classes else None,
            "bounding_box": None,
            "status": "not_available",
            "yolo_detections": detected_classes
        }
