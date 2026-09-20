import os
import json
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from evaluation.dataset_loader import SnakeDatasetLoader
from agents.coordinator import AgentCoordinator

def run_experiment(loader, coordinator, name, use_detection=True, use_uncertainty=True, use_xai=True, use_advisory=True):
    print(f"\n--- Running Experiment: {name} ---")
    y_true = []
    y_pred = []
    times = []
    
    for filename, pil_img, gt in loader.get_samples():
        if gt is None or "species" not in gt:
            continue
            
        start = time.perf_counter()
        result = coordinator.process_image(
            pil_img, 
            use_detection=use_detection, 
            use_uncertainty=use_uncertainty, 
            use_xai=use_xai, 
            use_advisory=use_advisory
        )
        end = time.perf_counter()
        
        times.append(end - start)
        y_true.append(gt["species"])
        y_pred.append(result["species"]["top_prediction"])
        
    if len(y_true) == 0:
        return None
        
    acc = accuracy_score(y_true, y_pred)
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    avg_latency = sum(times) / len(times)
    
    return {
        "Accuracy": acc,
        "F1-Score": f1,
        "Avg_Latency_ms": avg_latency * 1000
    }

def run_ablation_study(dataset_dir="snake_dataset"):
    loader = SnakeDatasetLoader(dataset_dir=dataset_dir)
    coordinator = AgentCoordinator()
    
    experiments = {
        "A: MobileNetV3 Baseline": {"use_detection": False, "use_uncertainty": False, "use_xai": False, "use_advisory": False},
        "B: MobileNetV3 + Top-K Uncertainty": {"use_detection": False, "use_uncertainty": True, "use_xai": False, "use_advisory": False},
        "C: Detection + MobileNetV3": {"use_detection": True, "use_uncertainty": False, "use_xai": False, "use_advisory": False},
        "D: Detection + Classification + Uncertainty + XAI": {"use_detection": True, "use_uncertainty": True, "use_xai": True, "use_advisory": False},
        "E: Full Multi-Agent Pipeline": {"use_detection": True, "use_uncertainty": True, "use_xai": True, "use_advisory": True},
    }
    
    results = {}
    for name, kwargs in experiments.items():
        res = run_experiment(loader, coordinator, name, **kwargs)
        if res:
            results[name] = res
            print(f"Accuracy: {res['Accuracy']*100:.2f}% | F1: {res['F1-Score']*100:.2f}% | Latency: {res['Avg_Latency_ms']:.2f} ms")
        else:
            print("Skipped (No ground truth).")
            
    if results:
        os.makedirs("evaluation/results", exist_ok=True)
        with open("evaluation/results/ablation_study.json", "w") as f:
            json.dump(results, f, indent=4)
        print("\nAblation study results saved to evaluation/results/ablation_study.json")

if __name__ == "__main__":
    run_ablation_study()
