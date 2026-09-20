import os
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from evaluation.dataset_loader import SnakeDatasetLoader
from agents.coordinator import AgentCoordinator
import numpy as np

def evaluate_accuracy(dataset_dir="snake_dataset"):
    print(f"Starting quantitative classification evaluation on dataset: {dataset_dir}")
    
    loader = SnakeDatasetLoader(dataset_dir=dataset_dir)
    coordinator = AgentCoordinator()
    
    y_true = []
    y_pred = []
    y_pred_top3 = []
    
    total_images = 0
    missing_gt = 0
    
    for filename, pil_img, gt in loader.get_samples():
        total_images += 1
        
        # Ground truth
        if gt is None or "species" not in gt:
            missing_gt += 1
            continue
            
        true_label = gt["species"]
        
        # Inference
        result = coordinator.process_image(pil_img)
        pred_label = result["species"]["top_prediction"]
        
        # Top 3 labels
        top3_labels = [pred_label] + [alt["species"] for alt in result["species"]["alternatives"]]
        
        y_true.append(true_label)
        y_pred.append(pred_label)
        y_pred_top3.append(top3_labels)

    if len(y_true) == 0:
        print(f"\nProcessed {total_images} images, but 0 had ground truth annotations.")
        print("Please populate annotations.json to see quantitative metrics.")
        return

    # Calculate Metrics
    acc = accuracy_score(y_true, y_pred)
    
    # Top-3 Accuracy
    top3_correct = sum(1 for t, p3 in zip(y_true, y_pred_top3) if t in p3)
    top3_acc = top3_correct / len(y_true)
    
    # F1, Precision, Recall
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    print("\n" + "="*40)
    print("CLASSIFICATION METRICS")
    print("="*40)
    print(f"Total Evaluated Samples: {len(y_true)}")
    print(f"Top-1 Accuracy: {acc*100:.2f}%")
    print(f"Top-3 Accuracy: {top3_acc*100:.2f}%")
    print(f"Weighted Precision: {precision*100:.2f}%")
    print(f"Weighted Recall: {recall*100:.2f}%")
    print(f"Weighted F1-Score: {f1*100:.2f}%")
    
    # Save results to JSON
    os.makedirs("evaluation/results", exist_ok=True)
    results = {
        "samples_evaluated": len(y_true),
        "top1_accuracy": acc,
        "top3_accuracy": top3_acc,
        "weighted_precision": precision,
        "weighted_recall": recall,
        "weighted_f1": f1
    }
    
    with open("evaluation/results/classification_metrics.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nDetailed results saved to evaluation/results/classification_metrics.json")

if __name__ == "__main__":
    evaluate_accuracy()
