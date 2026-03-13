def deduplicate(chunks):
    seen = set()
    unique = []

    for chunk in chunks:
        if chunk["chunk_id"] not in seen:
            unique.append(chunk)
            seen.add(chunk["chunk_id"])

    return unique


def needs_retry(semantic_alignment_score, evidence_flags):
    if semantic_alignment_score < 0.25 and not evidence_flags["absent"]:
        return True

    if evidence_flags["low_density"]:
        return True

    return False 


def assign_limitations(semantic_alignment_score, absent=False, low_density=False):
    
    if semantic_alignment_score < 0.25:
        return ["The literature does not address this question directly"]
    
    if absent:
        return ["No sufficiently relevant evidence was identified"]
    
    if low_density:
        return ["The retrieved evidence is too narrow and context-specific to support synthesis across studies"]