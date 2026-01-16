from typing import Set, Dict, List
import re


CITATION_PATTERN = re.compile(r"\([^)]*\d{4}[^)]*\)")

def remove_citations_inside_text(answer):
    """
    Remove parenthetical citation-like patterns from text.
    """
    cleaned = []

    for sentence in answer:
        text = sentence['text']
        text_no_citations = CITATION_PATTERN.sub("", text).strip()

        cleaned.append({
            "text": text_no_citations,
            "citations": sentence['citations']
        })

    return cleaned



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
    Replace chunk_id citations with human-readable labels,
    deduplicated at the paper level.

    Returns:
        resolved_answer: list of sentences with [Author, Year] citations
        sentence_citations: list of sets of paper_ids per sentence
    """

    resolved_answer = []
    sentence_citations = []

    for sentence in answer:
        seen_papers = set()
        labels = []

        for cid in sentence.get("citations", []):
            source = source_lookup.get(cid)
            if not source:
                continue

            paper_id = source.get("paper_id")
            if not paper_id or paper_id in seen_papers:
                continue

            label = format_author_year(
                source.get("authors"),
                source.get("year")
            )

            labels.append(label)
            seen_papers.add(paper_id)

        sentence_citations.append(seen_papers)

        resolved_answer.append({
            "text": sentence["text"],
            "citations": labels
        })

    return resolved_answer, sentence_citations



def build_source_entry(paper_id, source_lookup):
    for src in source_lookup.values():
        if src["paper_id"] == paper_id:
            return {
                "paper_id": paper_id,
                "title": src.get("title"),
                "authors": src.get("authors"),
                "year": src.get("year"),
                "journal": src.get("journal")
            }


# def build_sources_from_used_chunks(
#     used_chunks: Set[str],
#     chunk_lookup: Dict[str, dict]
# ) -> List[dict]:
#     """
#     Build a deduplicated list of source papers from cited chunk IDs.

#     Args:
#         used_chunks (set[str]): Chunk IDs cited in the answer
#         chunk_lookup (dict): Maps chunk_id -> chunk metadata

#     Returns:
#         list[dict]: Unique source papers supporting the answer
#     """
#     sources = {}

#     for cid in used_chunks:
#         chunk = chunk_lookup.get(cid)
#         if not chunk:
#             continue  # safety against invalid citations

#         pid = chunk["paper_id"]

#         if pid not in sources:
#             sources[pid] = {
#                 "paper_id": pid,
#                 "title": chunk["title"],
#                 "authors": chunk["authors"],
#                 "year": chunk["year"],
#                 "journal": chunk.get("journal"),
#             }

#     return list(sources.values())



