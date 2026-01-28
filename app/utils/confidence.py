def compute_confidence(metrics, reason):
    """
    Determine a confidence score in [0, 1], a confidence label,
    and short explanations based on evidence strength.
    """

    # --- Score ---
    paper_coverage = metrics.get("paper_coverage", 0.0)
    paper_dominance = metrics.get("paper_dominance", 1.0)
    multi_source_ratio = metrics.get("multi_source_sentence_ratio", 0.0)

    score = 1.0
    score -= 0.5 * (1 - paper_coverage)
    score -= 0.3 * paper_dominance
    score -= 0.2 * (1 - multi_source_ratio)

    score = max(0.0, round(score, 2))

    if reason == "insufficient_evidence":
        score = min(score, 0.4)

    # --- Label ---
    if score >= 0.75:
        label = "High"
    elif score >= 0.45:
        label = "Medium"
    else:
        label = "Low"

    # --- Explanations ---
    signals = []

    if paper_coverage < 0.5:
        signals.append(
            "Only a small portion of the retrieved papers contributed to the answer."
        )

    if paper_dominance > 0.5:
        signals.append(
            "The synthesis relies heavily on a single paper."
        )

    if multi_source_ratio == 0:
        signals.append(
            "None of the claims are supported by multiple independent sources."
        )
    elif multi_source_ratio < 0.3:
        signals.append(
            "Only a small fraction of claims are supported by multiple independent sources."
        )

    if label == "High":
        explanation = [
            "The answer is supported by multiple independent sources with good coverage."
        ]
    else:
        explanation = signals[:3]  # safe even if <2

    return score, label, explanation
