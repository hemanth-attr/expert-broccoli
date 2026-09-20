import cv2
import numpy as np
import base64
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

class XAIAgent:
    def __init__(self):
        pass

    def generate_heatmap(self, model, input_tensor, pil_img, target_class_idx):
        # target_layers: For MobileNetV3, usually features[-1]
        target_layers = [model.features[-1]]
        
        # We initialize GradCAM here. Alternatively, could be initialized once.
        # But we need the model reference.
        cam = GradCAM(model=model, target_layers=target_layers)

        # Prepare RGB image for overlay
        rgb_img = cv2.resize(np.array(pil_img), (224, 224))
        rgb_img = np.float32(rgb_img) / 255.0

        targets = [ClassifierOutputTarget(target_class_idx)]
        
        # Generate CAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        cam_overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # Convert heatmap to base64 JPEG
        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(cam_overlay, cv2.COLOR_RGB2BGR))
        heatmap_base64 = base64.b64encode(buffer).decode("utf-8")
        
        return {
            "method": "Grad-CAM",
            "heatmap_available": True,
            "heatmap_base64": heatmap_base64,
            "attention_quality": "acceptable" # Placeholder for future evaluation
        }
