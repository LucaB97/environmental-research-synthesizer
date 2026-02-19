from app.schemas import QueryResponse

def build_response(
    question,
    pipeline_status,
    limitations,
    meta=None,
    confidence=None,
    debug=None
):
    return QueryResponse(
        question=question,
        pipeline_status=pipeline_status,
        answer=[],
        limitations=limitations,
        sources=[],
        meta=meta or {},
        confidence=confidence,
        debug=debug or {}
    )
