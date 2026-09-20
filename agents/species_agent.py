import torch
import torch.nn as nn
from torchvision import models, transforms
import json
import torch.nn.functional as F

class SpeciesAgent:
    def __init__(self, model_path="models/mobilenet_snake_classifier_v2.pth", mapping_path="models/class_mapping.json"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        with open(mapping_path, "r") as f:
            class_mapping = json.load(f)
        self.class_names = [class_mapping[str(i)] for i in range(len(class_mapping))]
        num_classes = len(self.class_names)

        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        self.model = models.mobilenet_v3_small(weights=weights)
        in_features = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(in_features, num_classes)

        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
        self.model.to(self.device)
        
        # Keep gradients active if XAI needs them, but during normal predict we use torch.no_grad()
        for param in self.model.parameters():
            param.requires_grad = True

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, pil_image, top_k=3):
        # We process the image but don't disable gradients globally since XAI might use it later
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = F.softmax(outputs, dim=1)[0]
        
        top_k_val = min(top_k, len(self.class_names))
        
        # We must aggregate probabilities for duplicated labels (e.g., 'train' and 'valid' both map to 'UNKNOWN')
        # BEFORE selecting Top-K.
        aggregated_probs = {}
        for idx, prob in enumerate(probs):
            species_name = self.class_names[idx]
            if species_name in ["train", "valid"]:
                species_name = "UNKNOWN"
                
            if species_name not in aggregated_probs:
                aggregated_probs[species_name] = {"prob": 0.0, "class_idx": idx}
            
            aggregated_probs[species_name]["prob"] += float(prob.item())
            
        # Sort aggregated by probability descending
        sorted_species = sorted(aggregated_probs.items(), key=lambda x: x[1]["prob"], reverse=True)
        
        predictions = []
        for i in range(min(top_k_val, len(sorted_species))):
            species_name = sorted_species[i][0]
            prob = sorted_species[i][1]["prob"]
            class_idx = sorted_species[i][1]["class_idx"]
            
            predictions.append({
                "species": species_name,
                "confidence": prob,
                "class_idx": class_idx
            })
            
        return predictions, input_tensor, probs
