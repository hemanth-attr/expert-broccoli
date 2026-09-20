import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from evaluation.dataset_loader import SnakeDatasetLoader
from agents.coordinator import AgentCoordinator

def evaluate_full_pipeline(dataset_dir="snake_dataset"):
    print(f"Starting full pipeline evaluation on dataset: {dataset_dir}")
    loader = SnakeDatasetLoader(dataset_dir=dataset_dir)
    coordinator = AgentCoordinator()
    
    metrics = {
        "classification": {"y_true": [], "y_pred": [], "y_pred_top3": []},
        "venom": {"y_true": [], "y_pred": []},
        "uncertainty": {"confidences": [], "correct": []},
        "detection": {"snake_detected": [], "is_unknown": []}
    }
    
    total_images = 0
    missing_gt = 0
    
    for filename, pil_img, gt in loader.get_samples():
        total_images += 1
        
        # Inference
        result = coordinator.process_image(pil_img)
        
        # Parse result
        pred_label = result["species"]["top_prediction"]
        top3_labels = [pred_label] + [alt["species"] for alt in result["species"]["alternatives"]]
        pred_venom_status = result["venom"]["status"]
        is_unknown_prediction = pred_label == "UNKNOWN"
        snake_detected = result["detection"]["snake_detected"]
        
        metrics["detection"]["snake_detected"].append(snake_detected)
        metrics["detection"]["is_unknown"].append(is_unknown_prediction)
        
        if gt is None:
            missing_gt += 1
            continue
            
        true_label = gt.get("species")
        true_venom_status = gt.get("venomous", "unknown")
        
        # Classification
        if true_label:
            metrics["classification"]["y_true"].append(true_label)
            metrics["classification"]["y_pred"].append(pred_label)
            metrics["classification"]["y_pred_top3"].append(top3_labels)
            
            is_correct = (true_label == pred_label)
            metrics["uncertainty"]["confidences"].append(result["species"]["confidence"])
            metrics["uncertainty"]["correct"].append(is_correct)
            
        # Venom
        if true_venom_status != "unknown" and pred_venom_status != "unknown":
            metrics["venom"]["y_true"].append(true_venom_status)
            metrics["venom"]["y_pred"].append(pred_venom_status)

    if len(metrics["classification"]["y_true"]) == 0:
        print(f"\nProcessed {total_images} images, but 0 had ground truth annotations.")
        print("Please populate annotations.json to see quantitative metrics.")
        return

    # Evaluate Classification
    y_true_cls = metrics["classification"]["y_true"]
    y_pred_cls = metrics["classification"]["y_pred"]
    y_pred_top3 = metrics["classification"]["y_pred_top3"]
    
    cls_acc = accuracy_score(y_true_cls, y_pred_cls)
    top3_correct = sum(1 for t, p3 in zip(y_true_cls, y_pred_top3) if t in p3)
    cls_top3_acc = top3_correct / len(y_true_cls)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_cls, y_pred_cls, average='weighted', zero_division=0)
    
    print("\n" + "="*40 + "\nCLASSIFICATION METRICS\n" + "="*40)
    print(f"Top-1 Accuracy: {cls_acc*100:.2f}%")
    print(f"Top-3 Accuracy: {cls_top3_acc*100:.2f}%")
    print(f"Weighted F1-Score: {f1*100:.2f}%")
    
    # Evaluate Venom
    y_true_venom = metrics["venom"]["y_true"]
    y_pred_venom = metrics["venom"]["y_pred"]
    if len(y_true_venom) > 0:
        venom_acc = accuracy_score(y_true_venom, y_pred_venom)
        v_prec, v_rec, v_f1, _ = precision_recall_fscore_support(y_true_venom, y_pred_venom, average='weighted', zero_division=0)
        print("\n" + "="*40 + "\nVENOM METRICS\n" + "="*40)
        print(f"Venom Accuracy: {venom_acc*100:.2f}%")
        print(f"Venom F1-Score: {v_f1*100:.2f}%")
        
    # Evaluate Uncertainty (Calibration proxy)
    confs = np.array(metrics["uncertainty"]["confidences"])
    corrects = np.array(metrics["uncertainty"]["correct"])
    
    print("\n" + "="*40 + "\nUNCERTAINTY & CALIBRATION\n" + "="*40)
    for threshold in [0.4, 0.6, 0.8]:
        mask = confs >= threshold
        count = np.sum(mask)
        if count > 0:
            acc_at_threshold = np.mean(corrects[mask])
            print(f"Threshold >= {threshold}: {count} predictions | Accuracy = {acc_at_threshold*100:.2f}%")
        else:
            print(f"Threshold >= {threshold}: 0 predictions")
            
    print("\nDetailed results saved to evaluation/results/full_pipeline_metrics.json")
    
if __name__ == "__main__":
    evaluate_full_pipeline()
