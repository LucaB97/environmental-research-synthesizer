from app.utils.citations import extract_citations

def validate_reason(synthesis_output: dict) -> str:
    """
    Enforce backend invariants on the LLM-provided reason.
    Returns the validated reason.
    """

    answer = synthesis_output.get("answer", [])
    llm_reason = synthesis_output.get("reason", "none")

    # Rule 1: no answer -> out_of_scope
    if not answer:
        return "out_of_scope"

    cited = extract_citations(answer)

    # Rule 2: no citations -> insufficient evidence
    if not cited:
        return "insufficient_evidence"

    # Rule 3: single-paper support -> insufficient evidence
    if len(set(cited)) < 2:
        return "insufficient_evidence"

    return llm_reason
