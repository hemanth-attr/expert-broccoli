import json
import os

class VenomAgent:
    def __init__(self, kb_path="advisory/knowledge_base.json"):
        with open(kb_path, "r") as f:
            self.kb = json.load(f)
        
        self.venomous_set = set(self.kb.get("venomous_species", []))
        self.non_venomous_set = set(self.kb.get("non_venomous_species", []))

    def classify_venom(self, top_prediction):
        """
        Determine if the species is venomous based on the knowledge base.
        Returns a dictionary with the venomous status and confidence.
        """
        species = top_prediction["species"]
        confidence = top_prediction["confidence"]
        
        is_venomous = False
        status_known = False

        if species in self.venomous_set:
            is_venomous = True
            status_known = True
        elif species in self.non_venomous_set:
            is_venomous = False
            status_known = True
        else:
            # Fallback to string matching if the KB is incomplete
            if "VENOMOUS" in species.upper() and "NON" not in species.upper():
                is_venomous = True
                status_known = True
            elif "NON-VENOMOUS" in species.upper() or "NON-VENOUMOUS" in species.upper():
                is_venomous = False
                status_known = True

        return {
            "is_venomous": is_venomous,
            "status_known": status_known,
            "confidence": confidence if status_known else 0.0
        }
