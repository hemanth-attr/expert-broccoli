import os
import json
import torch
from torchvision import transforms, models
from PIL import Image

def main():
    # Setup paths
    model_path = 'mobilenet_snake_classifier_v2.pth'
    class_mapping_path = 'class_mapping.json'
    image_dir = 'snake_dataset'

    # Load class mapping
    with open(class_mapping_path, 'r') as f:
        class_mapping = json.load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    try:
        model_data = torch.load(model_path, map_location=device, weights_only=False)
        
        # If it's a full model, it should be a nn.Module
        if isinstance(model_data, torch.nn.Module):
            model = model_data
            print("Loaded full model object.")
        # If it's a state_dict, we need to initialize the model first
        elif isinstance(model_data, dict):
            print("Loaded state dict. Attempting to initialize MobileNetV3.")
            num_classes = max([int(k) for k in class_mapping.keys() if k.isdigit()]) + 1
            print(f"Detected {num_classes} classes from mapping.")
            
            # Assuming large variant for now. Might be small. Let's try large first.
            try:
                model = models.mobilenet_v3_large(weights=None)
                num_ftrs = model.classifier[3].in_features
                model.classifier[3] = torch.nn.Linear(num_ftrs, num_classes)
                model.load_state_dict(model_data)
                print("Successfully loaded state dict into mobilenet_v3_large.")
            except Exception as e:
                print(f"Failed loading as large variant: {e}")
                print("Trying small variant...")
                model = models.mobilenet_v3_small(weights=None)
                num_ftrs = model.classifier[3].in_features
                model.classifier[3] = torch.nn.Linear(num_ftrs, num_classes)
                model.load_state_dict(model_data)
                print("Successfully loaded state dict into mobilenet_v3_small.")
        else:
            raise ValueError(f"Unknown model data type: {type(model_data)}")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model = model.to(device)
    model.eval()

    # Define transforms
    # Using standard ImageNet values for normalization
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("\nStarting inference...\n" + "-"*40)
    
    # Process images
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, filename)
            try:
                # Load and preprocess image
                image = Image.open(image_path).convert('RGB')
                input_tensor = transform(image)
                input_batch = input_tensor.unsqueeze(0).to(device)

                # Inference
                with torch.no_grad():
                    output = model(input_batch)
                    
                # Calculate probabilities and get top prediction
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                confidence, predicted_idx = torch.max(probabilities, 0)
                
                # Get the class name
                confidence_score = confidence.item()
                predicted_idx_str = str(predicted_idx.item())
                predicted_species = class_mapping.get(predicted_idx_str, f"Unknown class {predicted_idx_str}")
                
                print(f"Image: {filename}")
                print(f"  Predicted Species: {predicted_species}")
                print(f"  Confidence Score: {confidence_score:.4f}\n")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}\n")

if __name__ == "__main__":
    main()
