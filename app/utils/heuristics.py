def determine_reason(synthesis_output, chunk_lookup):
    answer = synthesis_output.get("answer", [])

    if not answer:
        return "out_of_scope"

    cited_chunks = {
        cid
        for sentence in answer
        for cid in sentence.get("citations", [])
    }

    if not cited_chunks:
        return "insufficient_evidence"

    cited_papers = {
        chunk_lookup[cid]["paper_id"]
        for cid in cited_chunks
        if cid in chunk_lookup
    }

    if len(cited_papers) < 2:
        return "insufficient_evidence"

    return "none"


#def should_retry()
