import json

class AdvisoryAgent:
    def __init__(self, kb_path="advisory/knowledge_base.json"):
        with open(kb_path, "r") as f:
            self.kb = json.load(f)
        self.rules = self.kb.get("advisory_rules", {})

    def get_advisory(self, venom_result, uncertainty_result, top_prediction=None):
        level = uncertainty_result["level"]
        species_name = top_prediction["species"] if top_prediction else "UNKNOWN"
        is_venomous = venom_result["is_venomous"]
        
        if level == "high" or species_name == "UNKNOWN":
            return self.rules.get("unknown_or_insufficient_confidence", {
                "type": "UNKNOWN",
                "message": "Seek medical attention if bitten. Cannot determine snake type."
            })
        
        if is_venomous:
            if level == "low":
                return self.rules.get("venomous_high_confidence")
            else:
                return self.rules.get("venomous_low_confidence")
        else:
            if level == "low":
                return self.rules.get("non_venomous_high_confidence")
            else:
                return self.rules.get("non_venomous_low_confidence")

