import numpy as np
from scipy.special import expit


def semantic_norm(score, a, b):
    import math
    return 1 / (1 + math.exp(-a * (score - b)))


def compute_contribution(abs_relevance, z, z_thr=0, kz=1.5):
    rel_relevance = expit(kz * (z - z_thr))
    return abs_relevance * rel_relevance



def evaluate_semantic_alignment(chunks, params, top_N):
    """
    Evaluate the semantic alignment of the retrieved evidence to the query, 
    based on previously evaluated chunk scores.

    Parameters
    ----------
    chunks : list[dict]
        Retrieved passages, each containing:
            - final_score (float): Relevance score.
            - paper_id (str): Identifier of source document.

    Returns
    -------
    semantic_alignment : float
        Continuous semantic alignment score in [0, 1].

    flags : dict
        Diagnostic indicators based on the score.
    """
    if not chunks:
        return None

    scores = np.array([c["final_score"] for c in chunks])
    a, b = params["normalization_params"]["a"], params["normalization_params"]["b"]
    
    if scores is None or a is None or b is None:
        return None 
    
    max_score = scores.max()
    top_scores = sorted(scores, reverse=True)[:top_N]
    mean_score_topN = sum(top_scores) / len(top_scores)

    norm_max = semantic_norm(max_score, a, b)
    norm_mean = semantic_norm(mean_score_topN, a, b)
    semantic_alignment = 0.5 * norm_max + 0.5 * norm_mean

    return semantic_alignment


def explain_semantic(semantic_alignment):
    """
    Generate explanations for semantics alignment of evidence to query
    """

    if semantic_alignment < 0.25:
        return "Top retrieved passages show little relevance to the query"
    
    elif semantic_alignment < 0.5: 
        return "Some top passages are relevant, but alignment with the query is inconsistent"

    elif semantic_alignment < 0.75:
        return "Top passages are generally relevant to the query"
    
    return "Top passages closely match the query"


def evaluate_evidence_structure(chunks, params):
    """
    Evaluate the structural quality of retrieved evidence.

    A continuous score is computed, based on:
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
    evidence_structure : float
        Continuous evidence structure score in [0, 1].

    flags : dict
        Diagnostic indicators describing structural properties
        (e.g., absent evidence, low diversity, source dominance).

    metrics : dict
        Detailed intermediate statistics used for analysis
        and debugging.

    strong_hit_chunks : list
    """

    if not chunks:
        return None, None, None
    
    a,b, std_global = (params["normalization_params"]["a"], 
                       params["normalization_params"]["b"], 
                       params["normalization_params"]["std_global"])
    
    q10_contributions, q90_contributions = (params["contributions_per_query"]["q10"],
                                            params["contributions_per_query"]["q90"])
    
    min_contribution_threshold = params["chunk_contributions"]["q25"]

    q90_distinctsources = params["effective_sources_per_query"]["q90"]

    
    scores = np.array([c["final_score"] for c in chunks])
    paper_ids = [c["paper_id"] for c in chunks]

    mean_score, std_score = scores.mean(), scores.std()

    # Z-normalization
    if std_score < 1e-6:
        z = np.zeros_like(scores)
    else:
        std = max(std_score, std_global*0.5) 
        z = (scores - mean_score) / std

    # Hits
    abs_relevance = np.array([semantic_norm(s, a, b) for s in scores])
    contributions = compute_contribution(abs_relevance, z)

    source_weights = {}
    for i, contrib in enumerate(contributions):
        if contrib > min_contribution_threshold:
            pid = paper_ids[i]
            source_weights[pid] = source_weights.get(pid, 0) + contrib

    distinct_effective_sources = len(source_weights)
    relevant_mass = sum(contributions)

    if source_weights:
        dominance_ratio = max(source_weights.values()) / relevant_mass
    else:
        dominance_ratio = 1.0

    # --------------------------
    # Evidence structure metrics
    # --------------------------
    density_score = min(relevant_mass / q90_contributions, 1.0)
    diversity_score = min(distinct_effective_sources / q90_distinctsources, 1.0)
    balance_score = 1.0 - dominance_ratio

    evidence_structure = 0.4 * density_score + 0.3 * diversity_score + 0.3 * balance_score
    evidence_structure = max(0.0, min(1.0, evidence_structure))

    flags = {
        "absent": relevant_mass < q10_contributions,

        "low_density": density_score < 0.4,
        "high_density": density_score >= 0.7,

        # diversity
        "low_diversity": diversity_score < 0.4,
        "multiple_relevant_sources": diversity_score >= 0.7,

        # dominance / balance
        "single_source_dominance": balance_score < 0.4,
        "well_balanced": balance_score >= 0.7,
    }

    metrics = {
        "evidence density": round(density_score,2),
        "source diversity": round(diversity_score,2),
        "source balance": round(balance_score,2)
    }

    return evidence_structure, flags, metrics



def explain_evidence(metrics, flags):


    # --- Critical states ---    
    if flags.get("absent"):
        return "No sufficiently relevant evidence was identified"


    density_bullet = "Evidence density — " + f"{metrics['evidence density']}\n"
    
    if flags.get("high_density"):
        density_bullet += "A substantial amount of relevant evidence was identified"

    elif flags.get("low_density"):
        density_bullet += "Only a limited amount of relevant evidence was identified"
    else:
        density_bullet += "A moderate amount of relevant evidence was identified"


    diversity_bullet = "Source diversity — " + f"{metrics['source diversity']}\n"

    if flags.get("multiple_relevant_sources"):
        diversity_bullet += "Relevant evidence comes from multiple independent sources"
    elif flags.get("low_diversity"):
        diversity_bullet += "Relevant evidence comes from only a few distinct sources"
    else:
        diversity_bullet += "Relevant evidence spans a limited number of sources"


    balance_bullet = "Source balance — " + f"{metrics['source balance']}\n"

    if flags.get("well_balanced"):
        balance_bullet += "Relevant evidence is well distributed across different sources"
    elif flags.get("single_source_dominance"):
        balance_bullet += "Most relevant evidence originates from the same source"
    else:
        balance_bullet += "Some sources contribute more heavily than others"


    return [density_bullet, diversity_bullet, balance_bullet]



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
        "multi_source_grounding": used_papers > 3,
        "high_source_dominance": dominance > 0.7,
        "moderate_source_dominance": 0.4 < dominance <= 0.7,
        "balanced_source_usage": dominance <= 0.4,
        "cross_source_corroboration": multi_ratio >= 0.3,
        "no_corroboration": multi_ratio == 0,
        "low_corroboration": multi_ratio <= 0.2
    }

    if used_papers == 0:
        return 0.0, flags

    # Base score from source count
    if used_papers > 3:
        base = 0.75
    elif used_papers == 3:
        base = 0.60
    elif used_papers == 2:
        base = 0.45
    else:
        base = 0.35

    dominance_penalty = max(0, dominance - 0.5) * 0.5
    corroboration_bonus = multi_ratio * 0.4

    score = base + corroboration_bonus - dominance_penalty
    score = max(0.0, min(1.0, score))

    return score, flags



def explain_grounding(metrics, flags):
    """
    Generate concise, severity-aware explanations for grounding quality.
    """

    # --- Critical ---
    if flags.get("no_citations"):
        return "The answer does not cite any sources"
    
    
    source_usage_bullet = f"Source usage — {metrics['used_papers']} papers\n"
    
    if flags.get("multi_source_grounding"):
        source_usage_bullet += "The synthesis uses evidence from multiple independent sources"
    elif flags.get("single_source_reliance"):
        source_usage_bullet += "The synthesis relies on a single source"
    else:
        source_usage_bullet += "The synthesis relies on a limited number of sources"


    paper_dominance_bullet = f"Source dominance — {metrics['paper_dominance']:.2f}\n"

    if flags.get("single_source_reliance"):
        paper_dominance_bullet += "All of the used evidence comes from a single source"
    elif flags.get("high_source_dominance"):
        paper_dominance_bullet += "Most of the used evidence comes from a single source"
    elif flags.get("moderate_source_dominance"):
        paper_dominance_bullet += "A relevant portion of the used evidence comes from a single source"
    else:
        paper_dominance_bullet += "Citations are fairly distributed across different sources"


    corroboration_bullet = f"Cross-source support — {metrics['multi_source_sentence_ratio']:.2f}\n"
    if flags.get("no_corroboration"):
        corroboration_bullet += "Claims are not corroborated across multiple sources"
    elif flags.get("low_corroboration"):
        corroboration_bullet += "Only limited cross-source corroboration is present"
    else:
        corroboration_bullet += "Several statements are supported by multiple sources"

    return [source_usage_bullet, paper_dominance_bullet, corroboration_bullet]



def evaluate_confidence_profile(pipeline_status, 
                                semantic_score=None,
                                evidence_score=None, evidence_metrics=None, evidence_flags=None, 
                                grounding_score=None, grounding_metrics=None, grounding_flags=None,
                                reason=None):
    """
    Compute a multi-axis confidence profile: semantic alignment, evidence structure, grounding quality.
    Each axis gets a score (0-1), a level (Weak/Moderate/Strong), and optional explanations.
    """

    if pipeline_status != "success" or semantic_score is None or evidence_score is None or grounding_score is None:
        return {
            "status": "Not applicable",
            "reason": reason
        }

    # --- Semantic alignment ---
    if semantic_score >= 0.75:
        semantic_level = "Strong"
    elif semantic_score >= 0.5:
        semantic_level = "Moderate"
    else:
        semantic_level = "Weak"

    semantic_alignment = {
        "level": semantic_level,
        "score": semantic_score,
        "explanation": explain_semantic(semantic_score) if semantic_score else []
    }


    # --- Evidence structure ---
    if evidence_score >= 0.75:
        evidence_level = "Strong"
    elif evidence_score >= 0.5:
        evidence_level = "Moderate"
    else:
        evidence_level = "Weak"

    evidence_structure = {
        "level": evidence_level,
        "score": evidence_score,
        "explanation": explain_evidence(evidence_metrics, evidence_flags) if evidence_flags else []
    }


    # --- Grounding quality ---
    if grounding_score >= 0.75:
        grounding_level = "Strong"
    elif grounding_score >= 0.5:
        grounding_level = "Moderate"
    else:
        grounding_level = "Weak"

    grounding_quality = {
        "level": grounding_level,
        "score": grounding_score,
        "explanation": explain_grounding(grounding_metrics, grounding_flags) if grounding_flags else []
    }

    # --- Assemble ---
    profile = {
        "semantic": semantic_alignment,
        "evidence": evidence_structure,
        "grounding": grounding_quality,
        "status": "Success",
    }

    return profile