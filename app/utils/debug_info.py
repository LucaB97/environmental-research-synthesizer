def get_debug_info(retrieved_chunks, used_chunks_ids):

    used_chunks = []

    for c in retrieved_chunks:
        if c['chunk_id'] in used_chunks_ids:
            used_chunks.append(c)

    retrieved = len(retrieved_chunks)
    used = len(used_chunks)

    unique_papers_used = {item['paper_id'] for item in used_chunks}

    return {
        "retrieved_chunks": retrieved_chunks,
        "used_chunks": used_chunks,
        "retrieval_metrics": {
            "retrieved": retrieved,
            "used": used,
            "coverage": used/retrieved,
            "unique_papers_used": len(unique_papers_used),
        }
    }