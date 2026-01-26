
def determine_reason(synthesis_output, chunk_lookup):
    """
    Qualitative failure explanation.
    Useful for user-facing messages.
    
    :param synthesis_output: Description
    :param chunk_lookup: Description
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



def should_retry(metrics):
    if (
        metrics["retrieved_papers"] > 3
        and metrics["paper_dominance"] > 0.7
        and metrics["paper_coverage"] < 0.5
    ):
        return True
    return False
