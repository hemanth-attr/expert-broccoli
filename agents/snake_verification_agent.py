import torch
import math

class SnakeVerificationAgent:
    def __init__(self, msp_threshold=0.5, entropy_threshold=0.50, margin_threshold=0.1):
        """
        An Out-Of-Distribution (OOD) rejection agent.
        """
        self.msp_threshold = msp_threshold
        self.entropy_threshold = entropy_threshold
        self.margin_threshold = margin_threshold

    def verify(self, probs_tensor, yolo_detections=None):
        """
        Args:
            probs_tensor: A 1D torch.Tensor of softmax probabilities.
        Returns:
            dict containing verification status and metrics
        """
        probs = probs_tensor.cpu().tolist()
        
        # 1. Maximum Softmax Probability (MSP)
        sorted_probs = sorted(probs, reverse=True)
        top1_prob = sorted_probs[0]
        top2_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
        
        # 2. Predictive Entropy
        entropy = -sum([p * math.log(p) for p in probs if p > 0])
        
        # 3. Margin
        margin = top1_prob - top2_prob
        
        is_verified = True
        reason = "Pass"
        
        # 4. Statistical OOD check
        if top1_prob < self.msp_threshold:
            is_verified = False
            reason = f"Top-1 confidence ({top1_prob:.2f}) below threshold ({self.msp_threshold})"
        elif entropy > self.entropy_threshold:
            is_verified = False
            reason = f"High predictive entropy ({entropy:.2f} > {self.entropy_threshold})"
        elif margin < self.margin_threshold:
            is_verified = False
            reason = f"Low confidence margin ({margin:.2f} < {self.margin_threshold})"
            
        status = "SNAKE_DETECTED" if is_verified else "NO_SNAKE"
        
        return {
            "is_verified": is_verified,
            "status": status,
            "metrics": {
                "msp": round(top1_prob, 4),
                "entropy": round(entropy, 4),
                "margin": round(margin, 4)
            },
            "reason": reason
        }
