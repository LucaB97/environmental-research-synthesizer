from typing import Set, Dict, List


def format_author_year(authors, year):
    """
    Format authors into a human-readable citation label.
    """
    if not authors:
        return f"Unknown ({year})"

    # Normalize separators
    normalized = authors.replace(" and ", ",")
    parts = [a.strip() for a in normalized.split(",") if a.strip()]

    first_author = parts[0]

    if len(parts) > 1:
        return f"{first_author} et al. ({year})"
    else:
        return f"{first_author} ({year})"



def resolve_answer_citations(answer, source_lookup):
    """
    Replace chunk_id citations with human-readable labels.
    """

    resolved = []

    for sentence in answer:
        labels = []

        for sid in sentence["citations"]:
            source = source_lookup.get(sid)
            if not source:
                continue

            label = format_author_year(
                source.get("authors"),
                source.get("year")
            )
            labels.append(label)

        resolved.append({
            "text": sentence["text"],
            "citations": labels
        })

    return resolved



def build_sources_from_used_chunks(
    used_chunks: Set[str],
    chunk_lookup: Dict[str, dict]
) -> List[dict]:
    """
    Build a deduplicated list of source papers from cited chunk IDs.

    Args:
        used_chunks (set[str]): Chunk IDs cited in the answer
        chunk_lookup (dict): Maps chunk_id -> chunk metadata

    Returns:
        list[dict]: Unique source papers supporting the answer
    """
    sources = {}

    for cid in used_chunks:
        chunk = chunk_lookup.get(cid)
        if not chunk:
            continue  # safety against invalid citations

        pid = chunk["paper_id"]

        if pid not in sources:
            sources[pid] = {
                "paper_id": pid,
                "title": chunk["title"],
                "authors": chunk["authors"],
                "year": chunk["year"],
                "journal": chunk.get("journal"),
            }

    return list(sources.values())
