def compute_confidence(metrics, reason):
    """
    Determine a confidence score in [0, 1], a confidence label,
    and short explanations based on evidence strength.
    """

    used_papers = metrics.get("used_papers", 0)
    paper_dominance = metrics.get("paper_dominance", 1.0)
    multi_source_ratio = metrics.get("multi_source_sentence_ratio", 0.0)

    score = 1.0
    signals = []

    # --- Evidence sufficiency (absolute, not relative) ---
    if used_papers == 0:
        score -= 0.7
        signals.append(
            "The answer is not directly supported by retrieved research papers."
        )
    elif used_papers < 2:
        score -= 0.4
        signals.append(
            "The answer relies on very limited research evidence."
        )
    elif used_papers < 4:
        score -= 0.2
        signals.append(
            "The answer is supported by a small number of research papers."
        )

    # --- Evidence robustness ---
    if paper_dominance > 0.6:
        score -= 0.3
        signals.append(
            "The synthesis relies heavily on a single paper."
        )
    elif paper_dominance > 0.4:
        score -= 0.15
        signals.append(
            "One paper contributes more heavily than others."
        )

    if multi_source_ratio == 0:
        score -= 0.4
        signals.append(
            "None of the claims are supported by multiple independent sources."
        )
    elif multi_source_ratio < 0.3:
        score -= 0.2
        signals.append(
            "Only a small fraction of claims are supported by multiple independent sources."
        )
    elif multi_source_ratio > 0.6:
        score += 0.05  # small bonus for strong corroboration

    # --- Reason-based cap ---
    if reason == "insufficient_evidence":
        score = min(score, 0.4)

    score = max(0.0, min(1.0, round(score, 2)))

    # --- Label ---
    if score >= 0.75:
        label = "High"
    elif score >= 0.45:
        label = "Medium"
    else:
        label = "Low"

    # --- Explanation ---
    if label == "High":
        explanation = [
            "The answer is supported by multiple independent sources with sufficient and balanced evidence."
        ]
    else:
        explanation = signals[:3]

    return score, label, explanation