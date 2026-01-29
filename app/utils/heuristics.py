
def determine_reason(synthesis_output, chunk_lookup):
    """
    Determine whether a synthesized answer should be marked as having
    insufficient evidence, based on simple structural heuristics.

    The function checks for:
    - Presence of an answer
    - Presence of citations for every sentence
    - Minimum diversity of cited papers
    - Minimum number of cited evidence chunks

    Returns:
        "insufficient_evidence" if the heuristics are not met,
        otherwise "none".
    """
    
    answer = synthesis_output.get("answer", [])

    # insufficient evidence if the answer is empty or there is some sentence without citations
    if not answer:
        return "insufficient_evidence"

    if any(not s.get("citations") for s in answer):
        return "insufficient_evidence"


    # insufficient evidence if less than three distinct papers are cited
    cited_chunks = {
        cid
        for sentence in answer
        for cid in sentence.get("citations", [])
    }

    cited_papers = {
        chunk_lookup[cid]["paper_id"]
        for cid in cited_chunks
        if cid in chunk_lookup
    }

    if len(cited_papers) < 3:
        return "insufficient_evidence"

    return "none"



def determine_retry_reason(metrics, threshold=0.3):
    
    failures = {
        # High when one paper dominates despite low overall coverage
        "source_diversity": (
            metrics["paper_dominance"] - (1 - metrics["paper_coverage"])
        ),

        # High when most sentences rely on a single source
        "corroboration": 1 - metrics["multi_source_sentence_ratio"],

        # High when many retrieved chunks were ignored
        "evidence_utilization": 1 - metrics["chunk_coverage"],
    }

    # Find worst failure mode
    retry_reason, severity = max(
        failures.items(),
        key=lambda x: x[1]
    )

    # If even the worst failure is mild → no retry
    if severity < threshold:
        return None

    return retry_reason
