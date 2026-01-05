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
        for c in bullet.get("citations", []):
            citations.add(c)
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
        ref = f"{c['authors']}, {c['year']}"
        if ref in cited_refs and c["paper_id"] not in sources:
            sources[c["paper_id"]] = {
                "paper_id": c["paper_id"],
                "title": c["title"],
                "authors": c["authors"],
                "year": c["year"]
            }

    return list(sources.values())
