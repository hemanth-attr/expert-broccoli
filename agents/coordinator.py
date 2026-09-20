from agents.species_agent import SpeciesAgent
from agents.xai_agent import XAIAgent
from agents.venom_agent import VenomAgent
from agents.advisory_agent import AdvisoryAgent
from agents.detection_agent import DetectionAgent
from agents.snake_verification_agent import SnakeVerificationAgent

class AgentCoordinator:
    def __init__(self):
        # Initialize agents
        self.detection_agent = DetectionAgent()
        self.species_agent = SpeciesAgent()
        self.xai_agent = XAIAgent()
        self.venom_agent = VenomAgent()
        self.advisory_agent = AdvisoryAgent()
        self.verification_agent = SnakeVerificationAgent()

    def process_image(self, pil_img, use_detection=True, use_uncertainty=True, use_xai=True, use_advisory=True):
        # 1. Detection Agent
        if use_detection:
            cropped_img, detection_result = self.detection_agent.detect_and_crop(pil_img)
            snake_detected = detection_result.get("snake_detected")
            status = detection_result.get("status")
        else:
            cropped_img = pil_img
            detection_result = {"snake_detected": False, "confidence": 0.0, "bounding_box": None, "status": "disabled"}
            snake_detected = False
            status = "disabled"
            
        is_classification_only = (snake_detected is None)
            
        # FORMAT DETECTION OUTPUT
        det_out = {
            "status": "not_available" if is_classification_only else ("snake_detected" if snake_detected else "no_snake"),
            "snake_detected": snake_detected,
            "confidence": detection_result.get("confidence", None),
            "bounding_box": detection_result.get("bounding_box", None),
            "experimental": True, # explicitly marking YOLO generic detector
            "mode": "classification_only" if is_classification_only else "full_pipeline"
        }

        # SHORT-CIRCUIT PIPELINE IF EXPLICITLY NO SNAKE (allow None to pass through)
        if snake_detected is False:
            return {
                "request_id": "req-1234",
                "detection": det_out,
                "species": {
                    "status": "not_evaluated",
                    "top_prediction": None,
                    "confidence": None,
                    "alternatives": []
                },
                "venom": {
                    "status": "unknown",
                    "source": None
                },
                "uncertainty": {
                    "level": "high",
                    "advisory_allowed": False
                },
                "xai": {
                    "method": "Grad-CAM",
                    "heatmap": ""
                },
                "advisory": {
                    "type": "INSUFFICIENT_EVIDENCE",
                    "available": False,
                    "message": "No snake detected in the image."
                }
            }

        # 2. Species Agent (Generates raw probabilities)
        predictions, input_tensor, probs_tensor = self.species_agent.predict(cropped_img, top_k=3)
        top_prediction = predictions[0]

        # 3. Snake Verification Agent (OOD Check)
        verification_result = self.verification_agent.verify(probs_tensor)
        is_verified = verification_result["is_verified"]
        
        if not is_verified:
            # If it's a cat or other OOD image, short-circuit here.
            return {
                "request_id": "req-1234",
                "detection": {
                    "status": verification_result["status"],
                    "snake_detected": False,
                    "confidence": None,
                    "bounding_box": None,
                    "experimental": True,
                    "mode": "verification_rejection",
                    "reason": verification_result["reason"]
                },
                "species": {
                    "status": "not_evaluated",
                    "top_prediction": None,
                    "confidence": None,
                    "alternatives": []
                },
                "venom": {
                    "status": "unknown",
                    "source": "not_evaluated"
                },
                "uncertainty": {
                    "level": "high",
                    "advisory_allowed": False
                },
                "xai": {
                    "method": "disabled",
                    "heatmap": ""
                },
                "advisory": {
                    "type": "INSUFFICIENT_EVIDENCE",
                    "available": False,
                    "message": "The system could not verify that this image contains a snake."
                },
                "verification_metrics": verification_result["metrics"]
            }

        # 4. Venom Agent
        # Safety gate: Only run venom classification if species is valid and NOT "UNKNOWN"
        if top_prediction["species"] != "UNKNOWN":
            venom_result = self.venom_agent.classify_venom(top_prediction)
        else:
            venom_result = {"is_venomous": False, "status_known": False, "source": None}

        # 5. Uncertainty Layer & Unknown / No-Snake Handling
        confidence = top_prediction["confidence"]
        is_unknown = False
        
        if use_uncertainty:
            # If the highest confidence is very low, the model is likely guessing
            if confidence < 0.40:
                is_unknown = True
                top_prediction["species"] = "UNKNOWN"
                
            if confidence >= 0.80:
                uncertainty_level = "low"
            elif confidence >= 0.60:
                uncertainty_level = "moderate"
            else:
                uncertainty_level = "high"
                venom_result["is_venomous"] = False
                venom_result["status_known"] = False
        else:
            uncertainty_level = "disabled"

        # If species is UNKNOWN, reset venom status
        if is_unknown or top_prediction["species"] == "UNKNOWN":
            is_unknown = True
            venom_result["is_venomous"] = False
            venom_result["status_known"] = False

        uncertainty_result = {
            "level": uncertainty_level,
            "expert_verification_recommended": True if uncertainty_level in ["moderate", "high"] else False
        }

        # 6. XAI Agent
        if use_xai and not is_unknown:
            xai_result = self.xai_agent.generate_heatmap(
                model=self.species_agent.model,
                input_tensor=input_tensor,
                pil_img=cropped_img,
                target_class_idx=top_prediction["class_idx"]
            )
        else:
            xai_result = {"method": "disabled", "heatmap_base64": ""}

        # 7. Advisory Agent
        if use_advisory:
            if is_unknown:
                advisory_result = {
                    "available": False,
                    "type": "INSUFFICIENT_EVIDENCE",
                    "message": "Species could not be identified with sufficient confidence. Cannot provide medical advisory."
                }
            else:
                advisory_result = self.advisory_agent.get_advisory(venom_result, uncertainty_result, top_prediction)
                advisory_result["available"] = True
        else:
            advisory_result = {"available": False, "type": "disabled"}

        # Assemble Final Response
        if is_unknown:
            v_status = "unknown"
            v_source = None
        elif venom_result.get("is_venomous"):
            v_status = "venomous"
            v_source = "knowledge_base"
        elif not venom_result.get("is_venomous") and venom_result.get("status_known"):
            v_status = "non-venomous"
            v_source = "knowledge_base"
        else:
            v_status = "unknown"
            v_source = None
            
        venom_formatted = {
            "status": v_status,
            "source": v_source
        }

        uncertainty_formatted = {
            "level": uncertainty_result["level"],
            "advisory_allowed": advisory_result.get("available", False)
        }

        xai_formatted = {
            "method": xai_result.get("method", "Grad-CAM"),
            "heatmap": xai_result.get("heatmap_base64", "")
        }

        response = {
            "request_id": "req-1234",
            "detection": det_out,
            "species": {
                "status": "identified" if not is_unknown else "uncertain",
                "top_prediction": top_prediction["species"],
                "confidence": round(top_prediction["confidence"], 4),
                "alternatives": predictions[1:]
            },
            "venom": venom_formatted,
            "uncertainty": uncertainty_formatted,
            "xai": xai_formatted,
            "advisory": advisory_result,
            "verification_metrics": verification_result["metrics"]
        }

        return response
