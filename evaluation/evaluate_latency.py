import time
import io
from PIL import Image
import numpy as np
from agents.coordinator import AgentCoordinator

def evaluate_latency():
    print("Starting detailed latency evaluation...")
    
    coordinator = AgentCoordinator()
    
    # Load a sample image
    try:
        pil_img = Image.open('snake_dataset/COBRA (131).jpg').convert('RGB')
    except Exception:
        # Create a dummy image if not available
        pil_img = Image.new('RGB', (224, 224), color='white')
        
    num_runs = 50
    times = []
    
    # Warmup
    for _ in range(5):
        coordinator.process_image(pil_img)
        
    print(f"Running {num_runs} iterations for latency benchmarking...")
    for i in range(num_runs):
        start_time = time.perf_counter()
        
        # We can simulate component timing by wrapping the agent calls or just measure total
        # Since coordinator encapsulates the calls, we will measure total pipeline latency.
        # To measure exact components, we can run them individually here.
        
        t0 = time.perf_counter()
        cropped_img, _ = coordinator.detection_agent.detect_and_crop(pil_img)
        t1 = time.perf_counter()
        
        preds, t_tensor = coordinator.species_agent.predict(cropped_img, top_k=3)
        t2 = time.perf_counter()
        
        _ = coordinator.xai_agent.generate_heatmap(
            model=coordinator.species_agent.model,
            input_tensor=t_tensor,
            pil_img=cropped_img,
            target_class_idx=preds[0]["class_idx"]
        )
        t3 = time.perf_counter()
        
        times.append({
            "total": t3 - t0,
            "detection": t1 - t0,
            "classification": t2 - t1,
            "xai": t3 - t2
        })
        
    total_times = [t["total"] for t in times]
    det_times = [t["detection"] for t in times]
    cls_times = [t["classification"] for t in times]
    xai_times = [t["xai"] for t in times]
    
    avg_latency = np.mean(total_times)
    median_latency = np.median(total_times)
    p95_latency = np.percentile(total_times, 95)
    p99_latency = np.percentile(total_times, 99)
    fps = 1.0 / avg_latency if avg_latency > 0 else 0
    
    print("\n--- Benchmark Results (Total Pipeline) ---")
    print(f"Average Latency: {avg_latency*1000:.2f} ms")
    print(f"Median Latency:  {median_latency*1000:.2f} ms")
    print(f"P95 Latency:     {p95_latency*1000:.2f} ms")
    print(f"P99 Latency:     {p99_latency*1000:.2f} ms")
    print(f"Estimated FPS:   {fps:.2f} FPS")
    
    print("\n--- Component Breakdown (Average) ---")
    print(f"Detection:       {np.mean(det_times)*1000:.2f} ms")
    print(f"Classification:  {np.mean(cls_times)*1000:.2f} ms")
    print(f"XAI (Grad-CAM):  {np.mean(xai_times)*1000:.2f} ms")

if __name__ == "__main__":
    evaluate_latency()
