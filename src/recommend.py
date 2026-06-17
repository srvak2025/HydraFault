from __future__ import annotations


TARGET_MEANINGS = {
    "cooler": {
        "100": "full efficiency",
        "20": "reduced efficiency",
        "3": "close to total failure",
    },
    "valve": {
        "100": "optimal switching behavior",
        "90": "small valve lag",
        "80": "severe valve lag",
        "73": "close to total valve failure",
    },
    "pump": {
        "0": "no leakage",
        "1": "weak leakage",
        "2": "severe leakage",
    },
    "accumulator": {
        "130": "optimal pressure",
        "115": "slightly reduced pressure",
        "100": "severely reduced pressure",
        "90": "close to total failure",
    },
    "stable": {
        "0": "unstable condition",
        "1": "stable condition",
    },
}


def describe_label(target: str, label: str) -> str:
    return TARGET_MEANINGS.get(target, {}).get(str(label), str(label))


def make_recommendation(target: str, predicted_label: str, top_features: list[str] | None = None) -> str:
    label = str(predicted_label)
    meaning = describe_label(target, label)

    top_feature_text = ""
    if top_features:
        cleaned = ", ".join(f.replace("_", " ") for f in top_features[:3])
        top_feature_text = f" Key evidence came from: {cleaned}."

    if target == "cooler":
        if label == "100":
            action = "Cooling system appears healthy. Continue normal monitoring."
        elif label == "20":
            action = "Inspect cooler efficiency, fluid temperature behavior, and heat-exchanger performance."
        else:
            action = "Treat as critical. Inspect cooling path, heat exchanger, and thermal control immediately."

    elif target == "valve":
        if label == "100":
            action = "Valve behavior appears normal. Continue monitoring switching response."
        elif label in {"90", "80"}:
            action = "Inspect valve response delay, actuator behavior, and control signal timing."
        else:
            action = "Treat as critical valve degradation. Inspect valve assembly before continued operation."

    elif target == "pump":
        if label == "0":
            action = "Pump leakage appears normal. Continue normal monitoring."
        elif label == "1":
            action = "Inspect for early internal pump leakage and pressure loss under load."
        else:
            action = "Severe pump leakage likely. Prioritize pump inspection and leakage diagnostics."

    elif target == "accumulator":
        if label == "130":
            action = "Accumulator pressure appears optimal. Continue normal monitoring."
        elif label == "115":
            action = "Inspect accumulator pressure stability and check for early pressure loss."
        elif label == "100":
            action = "Severe accumulator pressure degradation likely. Schedule maintenance soon."
        else:
            action = "Critical accumulator condition likely. Inspect accumulator before continued operation."

    elif target == "stable":
        if label == "1":
            action = "System appears stable."
        else:
            action = "System appears unstable. Inspect sensor instability, pressure variation, and thermal drift."

    else:
        action = "Inspect the subsystem associated with the predicted condition."

    return f"Predicted condition: {meaning}. {action}{top_feature_text}"
