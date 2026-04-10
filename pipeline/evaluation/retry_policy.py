def need_retry_semantic(semantic_alignment_score, evidence_flags):
    if semantic_alignment_score < 0.25 and not evidence_flags["absent"]:
        return True

    if evidence_flags["low_density"]:
        return True

    return False 


def reason_retry_grounding(metrics):
    """
    Determine whether post-synthesis regeneration (retry) is warranted
    based on structural imbalances in how retrieved evidence was used.

    Structural imbalance cases evaluated
    -------------------------------------
    1. Source dominance despite available diversity:
       The answer relies heavily on a single paper even though
       multiple strong papers were available.

    2. Extremely low evidence usage:
       Only a very small fraction of retrieved chunks were cited
       or incorporated into the answer.

    3. No cross-source corroboration despite diversity:
       Multiple strong papers were available and cited, but no
       individual claim integrates evidence from more than one source.

    Parameters
    ----------
    metrics : dict
        Grounding-related metrics computed after synthesis. Expected keys:
            - used_papers (int): Number of distinct cited papers.
            - paper_dominance (float): Proportion of citations attributed
              to the most frequently cited paper.
            - chunk_coverage (float): Fraction of retrieved chunks that
              were cited or used in the answer.
            - multi_source_sentence_ratio (float): Fraction of sentences
              supported by multiple sources.

    distinct_hit_papers : int
        Number of distinct papers containing highly relevant passages
        identified during evidence structure evaluation. This reflects
        the diversity that was available prior to synthesis.

    Returns
    -------
    str or None
        The dominant structural imbalance type triggering retry:
            - "source_dominance"
            - "low_evidence_usage"
            - "no_corroboration"
        Returns None if no severe imbalance is detected.
    
    Notes
    -----
    This function is intended to be used only when grounding quality
    is already below an acceptable threshold. It does not optimize for
    ideal structure, but instead detects substantial grounding failures
    that justify regeneration.
    """

    failures = {}

    # -----------------------------------------
    # 1. Source dominance despite diversity
    # -----------------------------------------
    if (metrics["available_papers"] > 3 and metrics["paper_dominance"] > 0.7):
        failures["source_dominance"] = metrics["paper_dominance"]

    # -----------------------------------------
    # 2. Extremely low evidence usage
    # -----------------------------------------
    if metrics["chunk_coverage"] < 0.3:
        failures["low_evidence_usage"] = 1 - metrics["chunk_coverage"]

    # -----------------------------------------
    # 3. No cross-source use despite diversity
    # -----------------------------------------
    if (
        metrics["available_papers"] > 3 and metrics["multi_source_sentence_ratio"] == 0):
        failures["no_corroboration"] = 0.75
    
    if not failures:
        return None

    return max(failures.items(), key=lambda x: x[1])[0]