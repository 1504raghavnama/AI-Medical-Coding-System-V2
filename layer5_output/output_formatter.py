
import json
from datetime import datetime

def format_output(clinical_note, pipeline_results):
    output = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "clinical_note": clinical_note,
        "suggested_codes": [],
        "total_suggestions": 0
    }
    for suggestion in pipeline_results:
        entity = suggestion["entity"]
        candidates = suggestion["candidates"]
        if candidates:
            best = candidates[0]
            output["suggested_codes"].append({
                "entity": entity,
                "primary_code": best["code"],
                "description": best["description"],
                "confidence": best["score"],
                "alternatives": [
                    {"code": c["code"], "description": c["description"], "confidence": c["score"]}
                    for c in candidates[1:]
                ]
            })
    output["total_suggestions"] = len(output["suggested_codes"])
    return output
