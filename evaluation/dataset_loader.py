import os
import json
from PIL import Image

class SnakeDatasetLoader:
    def __init__(self, dataset_dir="snake_dataset", annotation_file="annotations.json"):
        """
        Loads images and their corresponding ground truth annotations.
        Expected annotation format:
        {
            "image_filename.jpg": {
                "species": "COBRA_VENOUMOUS",
                "is_snake": true
            }
        }
        """
        self.dataset_dir = dataset_dir
        self.annotation_path = os.path.join(dataset_dir, annotation_file)
        self.annotations = {}
        
        if os.path.exists(self.annotation_path):
            with open(self.annotation_path, "r") as f:
                self.annotations = json.load(f)
        else:
            print(f"Warning: Annotation file {self.annotation_path} not found. Evaluation metrics requiring ground truth will be severely limited.")

    def get_samples(self):
        """
        Generator that yields (filename, pil_image, ground_truth_dict).
        """
        if not os.path.exists(self.dataset_dir):
            print(f"Dataset directory {self.dataset_dir} not found.")
            return

        for filename in os.listdir(self.dataset_dir):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            filepath = os.path.join(self.dataset_dir, filename)
            try:
                pil_img = Image.open(filepath).convert('RGB')
                gt = self.annotations.get(filename, None)
                yield filename, pil_img, gt
            except Exception as e:
                print(f"Error loading {filename}: {e}")

if __name__ == "__main__":
    loader = SnakeDatasetLoader()
    samples = list(loader.get_samples())
    print(f"Loaded {len(samples)} samples.")
