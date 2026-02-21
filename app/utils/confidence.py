import numpy as np



def evaluate_evidence_structure(chunks, floor=0.25):
    """
    Evaluate the structural quality of retrieved evidence.

    This function computes a continuous structure score based on:
        - Density of high-relevance passages
        - Diversity of supporting sources
        - Balance of contribution across sources

    Parameters
    ----------
    chunks : list[dict]
        Retrieved passages, each containing:
            - final_score (float): Relevance score.
            - paper_id (str): Identifier of source document.

    floor : float, optional
        Minimum relevance threshold required to consider
        evidence present.

    Returns
    -------
    structure_score : float
        Continuous evidence structure score in [0, 1].

    flags : dict
        Diagnostic indicators describing structural properties
        (e.g., absent evidence, low diversity, source dominance).

    metrics : dict
        Detailed intermediate statistics used for analysis
        and debugging.
    """

    if not chunks:
        return None

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
        "single_source_dominance": strong_hits >= 5 and distinct_strong_sources == 1,
        "low_density": effective_hits < 3,
        "low_diversity": distinct_strong_sources < 2,
        "multiple_strong_sources": strong_hits >= 3 and distinct_strong_sources >= 2,
        "well_balanced": dominance_ratio < 0.6 and distinct_strong_sources >= 2,
        "high_density": effective_hits >= 6,
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
    


def explain_evidence(flags, max_items=3):
    """
    Generate severity-aware explanations for evidence structure,
    returning separate lists for weaknesses and strengths.
    """
    explanations = {
        "weaknesses": [],
        "strengths": []
    }

    # --- Critical states ---
    if flags.get("absent"):
        explanations["weaknesses"].append("No sufficiently relevant sources were identified.")
        return explanations

    if flags.get("isolated"):
        explanations["weaknesses"].append("Support relies on a single highly prominent passage.")
        return explanations

    # --- Weaknesses ---
    if flags.get("single_source_dominance"):
        explanations["weaknesses"].append("Strong support is concentrated in one source.")

    if flags.get("low_diversity"):
        explanations["weaknesses"].append("Few independent sources contribute strong support.")

    if flags.get("low_density"):
        explanations["weaknesses"].append("Only a limited number of highly relevant passages were found.")

    # --- Strengths ---
    if flags.get("high_density"):
        explanations["strengths"].append("Numerous highly relevant passages reinforce the claim.")

    if flags.get("multiple_strong_sources"):
        explanations["strengths"].append("Multiple independent sources strongly support the claim.")

    if flags.get("well_balanced"):
        explanations["strengths"].append("Support is well distributed across sources.")

    # --- Optional: limit items ---
    explanations["weaknesses"] = explanations["weaknesses"][:max_items]
    explanations["strengths"] = explanations["strengths"][:max_items]

    return explanations



def evaluate_grounding_quality(metrics):
    """
    Evaluate the grounding quality of a generated answer.

    The grounding score reflects how well the answer integrates
    and distributes citations across multiple sources.

    Parameters
    ----------
    metrics : dict
        Dictionary containing grounding-related metrics:
            - used_papers (int): Number of distinct cited papers.
            - paper_dominance (float): Proportion of citations
              attributed to the most frequently cited paper.
            - multi_source_sentence_ratio (float): Fraction of
              sentences supported by multiple sources.

    Returns
    -------
    grounding_score : float
        Continuous grounding quality score in [0, 1].

    flags : dict
        Diagnostic boolean flags describing grounding properties.
    """

    used_papers = metrics.get("used_papers", 0)
    dominance = metrics.get("paper_dominance", 1.0)
    multi_ratio = metrics.get("multi_source_sentence_ratio", 0.0)

    flags = {
        "no_citations": used_papers == 0,
        "single_source_reliance": used_papers == 1,
        "multi_source_grounding": used_papers >= 3,
        "high_source_dominance": dominance > 0.7,
        "balanced_source_usage": dominance <= 0.6,
        "cross_source_corroboration": multi_ratio >= 0.3,
        "no_corroboration": multi_ratio == 0,
        "low_corroboration": multi_ratio <= 0.2
    }

    if used_papers == 0:
        return 0.0, flags

    # Base score from source count
    if used_papers >= 3:
        base = 0.6
    elif used_papers == 2:
        base = 0.5
    else:
        base = 0.35

    dominance_penalty = max(0, dominance - 0.5) * 0.5
    corroboration_bonus = multi_ratio * 0.4

    score = base + corroboration_bonus - dominance_penalty
    score = max(0.0, min(1.0, score))

    return score, flags



def explain_grounding(flags, max_items=3):
    """
    Generate concise, severity-aware explanations for grounding quality.

    - Strength signals ordered simple → sophisticated.
    - Max 3 bullets by default.
    - If critical weakness exists, limit strength signals to 1.
    """

    explanations = {
        "weaknesses": [],
        "strengths": []
    }

    # --- Critical ---
    if flags.get("no_citations"):
        explanations["weaknesses"].append("The answer does not cite supporting sources.")
        return explanations
    
    # --- Weaknesses (ordered by severity) ---
    if flags.get("no_corroboration"):
        explanations["weaknesses"].append("Claims are not corroborated across multiple sources.")

    elif flags.get("low_corroboration"):
        explanations["weaknesses"].append("Only limited cross-source corroboration is present.")

    if flags.get("high_source_dominance"):
        explanations["weaknesses"].append("Most citations come from one dominant source.")

    if flags.get("single_source_reliance"):
        explanations["weaknesses"].append("The answer relies primarily on a single source.")

    # --- Strengths (simple → sophisticated) ---
    if flags.get("multi_source_grounding"):
        explanations["strengths"].append("The answer integrates multiple independent sources.")

    if flags.get("balanced_source_usage"):
        explanations["strengths"].append("Citations are distributed across sources.")

    if flags.get("cross_source_corroboration"):
        explanations["strengths"].append("Several claims are supported by multiple sources.")

    # --- Optional: limit items ---
    explanations["weaknesses"] = explanations["weaknesses"][:max_items]
    explanations["strengths"] = explanations["strengths"][:max_items]

    return explanations



def evaluate_confidence_profile(pipeline_status, 
                                evidence_score=None, evidence_flags=None, 
                                grounding_score=None, grounding_flags=None,
                                reason=None):


    if pipeline_status != "success" or evidence_score is None or grounding_score is None:
        return {
        "status": "Not applicable",
        "reason": reason
    }


    if evidence_score >= 0.75:
        evidence_level = "Strong"
    elif evidence_score >= 0.5:
        evidence_level = "Moderate"
    else:
        evidence_level = "Weak"

    evidence_strength = {
        "level": evidence_level,
        "score": evidence_score,
        "explanation": explain_evidence(evidence_flags) if evidence_flags is not None else []
    }

    if grounding_score >= 0.75:
        grounding_level = "Strong"
    elif grounding_score >= 0.5:
        grounding_level = "Moderate"
    else:
        grounding_level = "Weak"    

    grounding_quality = {
        "level": grounding_level,
        "score": grounding_score,
        "explanation": explain_grounding(grounding_flags) if grounding_flags is not None else []
    }

    return {
        "evidence": evidence_strength,
        "grounding": grounding_quality,
        "status": "Success"
    }