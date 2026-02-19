import numpy as np


def determine_grounding(metrics):

    used_papers = metrics.get("used_papers", 0)
    dominance = metrics.get("paper_dominance", 1.0)
    multi_ratio = metrics.get("multi_source_sentence_ratio", 0.0)

    if used_papers == 0:
        return 0.0, "not_answered"
    
    if used_papers >= 3:
        base = 0.6
    elif used_papers == 2:
        base = 0.5
    else:
        base = 0.35

    dominance_penalty = max(0, dominance - 0.5) * 0.5
    corroboration_bonus = multi_ratio * 0.35

    score = base + corroboration_bonus - dominance_penalty
    score = max(0.05, min(0.9, score))

    return score



def evaluate_evidence_structure(chunks, floor=0.25):
    """
    Classify the overall evidence distribution based on chunk scores
    and source diversity.

    Args:
        chunks (list[dict]): Retrieved chunks, each with:
            - "score": cross-encoder score
            - "paper_id": identifier of source paper

    Returns:
        str: Evidence label
    """

    if not chunks:
        return "absent"

    scores = np.array([c["final_score"] for c in chunks])
    paper_ids = [c["paper_id"] for c in chunks]

    mean = scores.mean()
    std = scores.std()
    max_score = scores.max()

    # --- Z-normalization ---
    if std < 1e-6:
        z = np.zeros_like(scores)
    else:
        z = (scores - mean) / std

    max_z = z.max()

    strong_indices = np.where(z > 1.0)[0]
    moderate_indices = np.where(z > 0.5)[0]

    strong_hits = len(strong_indices)
    moderate_hits = len(moderate_indices)

    distinct_strong_sources = len(
        set(paper_ids[i] for i in strong_indices)
    )

    # --- Compute dominance ratio ---
    if strong_hits > 0:
        strong_source_counts = {}
        for i in strong_indices:
            pid = paper_ids[i]
            strong_source_counts[pid] = strong_source_counts.get(pid, 0) + 1

        max_source_hits = max(strong_source_counts.values())
        dominance_ratio = max_source_hits / strong_hits
    else:
        dominance_ratio = 1.0

    # --- Effective density (include moderate signals) ---
    pure_moderate_hits = max(0, moderate_hits - strong_hits)
    effective_hits = strong_hits + 0.5 * pure_moderate_hits

    density_score = min(effective_hits / 10.0, 1.0)
    diversity_score = min(distinct_strong_sources / 3.0, 1.0)
    balance_score = 1.0 - dominance_ratio

    structure_score = (
        0.4 * density_score +
        0.4 * diversity_score +
        0.2 * balance_score
    )
    structure_score = max(0.0, min(1.0, structure_score))

    flags = {
        "absent": max_score < floor,
        "isolated": strong_hits == 1 and max_z > 2,
        "mono_source_strong": strong_hits >= 5 and distinct_strong_sources == 1,
        "low_density": effective_hits < 3,
        "low_diversity": distinct_strong_sources < 2
    }

    if flags['absent']:
        metrics = {
            "mean_score": mean,
            "std": std,
            "max_score": max_score
        }

    elif flags['isolated']:
        metrics = {
            "mean_score": mean,
            "std": std,
            "max_score": max_score,
            "strong_hits": strong_hits,
            "strong_hit_chunks": [chunks[strong_indices]]
        }

    else:
        metrics = {
        "mean_score": mean,
        "std": std,
        "max_score": max_score,
        "strong_hits": strong_hits,
        "moderate_hits": moderate_hits,
        "distinct_strong_sources": distinct_strong_sources,
        "dominance_ratio": dominance_ratio,
        "density_score": density_score,
        "diversity_score": diversity_score,
        "balance_score": balance_score
    }

    return structure_score, flags, metrics 
    



def assign_confidence_label(pipeline_status, evidence_score=None, grounding_score=None):

    if pipeline_status != "success" or evidence_score is None or grounding_score is None:
        return None, None, "Not applicable", ["Confidence information is not available."]

    label = ""

    if evidence_score >= 0.75:
        label += "Strong evidence"
    elif evidence_score >= 0.5:
        label += "Moderate evidence"
    else:
        label += "Weak evidence"

    label += " · "

    if grounding_score >= 0.75:
        label += "Strong integration"
    elif grounding_score >= 0.5:
        label += "Moderate integration"
    else:
        label += "Weak integration"

    return label