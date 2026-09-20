import os
from PIL import Image
from agents.coordinator import AgentCoordinator
import json

def test_image(img_path):
    print("=" * 50)
    print(f"Testing: {img_path}")
    if not os.path.exists(img_path):
        print("File not found!")
        return
        
    pil_img = Image.open(img_path).convert("RGB")
    coordinator = AgentCoordinator()
    
    response = coordinator.process_image(pil_img)
    
    print("\nFINAL RESPONSE:")
    print(json.dumps(response, indent=2))

if __name__ == "__main__":
    test_image("D:\\PROJECT 4th\\snake_dataset\\real-cat-image-not-snake.jpg")
    test_image("D:\\PROJECT 4th\\snake_dataset\\COBRA (131).jpg")
