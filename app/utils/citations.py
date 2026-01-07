def extract_citations(answer_bullets):
    """
    Extract unique citation strings from synthesized answer bullets.

    Args:
        answer_bullets (list[dict]): Each dict contains 'text' and 'citations'

    Returns:
        set[str]: Unique citation identifiers (e.g. "Author et al., 2023")
    """
    citations = set()
    for bullet in answer_bullets:
        citations.update(bullet.get("citations", []))
    return citations


def build_sources_from_citations(chunks, cited_refs):
    """
    Build a deduplicated list of source papers that were actually cited.

    Args:
        chunks (list[dict]): Retrieved chunks
        cited_refs (set[str]): Citation strings extracted from answer

    Returns:
        list[dict]: Unique source papers supporting the answer
    """
    sources = {}

    for c in chunks:
        if c["paper_id"] in cited_refs:
            sources[c["paper_id"]] = {
                "paper_id": c["paper_id"],
                "title": c["title"],
                "authors": c["authors"],
                "year": c["year"],
                "journal": c.get("journal")
            }

    return list(sources.values())
