def assign_limitations(semantic_alignment_score=None, absent=False):
    
    if semantic_alignment_score < 0.25:
        return ["The literature does not address this question directly"]
    
    if absent:
        return ["The retrieved evidence is too narrow to support synthesis across studies"]