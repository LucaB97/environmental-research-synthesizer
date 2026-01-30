# imports
from enum import Enum
import re

# constants
CHUNK_ID_PATTERN = re.compile(r"\(paper_\d+__chunk_\d+\)")

# enums
class CitationStyle(Enum):
    AUTHOR_YEAR = "author_year"
    NUMERIC = "numeric"



# CITATIONS INDEX

def build_citation_index(sentence_papers):
    """
    sentence_papers: List[Set[paper_id]]
    Returns:
        citation_index: Dict[paper_id, int]
    """
    ordered_papers = []
    seen = set()

    for papers in sentence_papers:
        for pid in papers:
            if pid not in seen:
                ordered_papers.append(pid)
                seen.add(pid)

    return {pid: i + 1 for i, pid in enumerate(ordered_papers)}
    


# CITATIONS FORMATTING

def format_author_year(_paper_id, source, _citation_index):
    """
    Format citation into a human-readable label.
    """

    authors = source.get("authors")
    year = source.get("year")

    if not authors:
        return f"Unknown ({year})"

    normalized = authors.replace(" and ", ",")
    parts = [a.strip() for a in normalized.split(",") if a.strip()]
    first_author = parts[0]

    if len(parts) > 1:
        return f"{first_author} et al. ({year})"
    else:
        return f"{first_author} ({year})"


def format_numeric(paper_id, _source, citation_index):
    return str(citation_index[paper_id])


FORMATTERS = {
    CitationStyle.AUTHOR_YEAR: format_author_year,
    CitationStyle.NUMERIC: format_numeric,
}



# CITATIONS RENDERING

def resolve_answer_citations(answer, source_lookup, citation_index, citation_formatter):
    """
    Replace chunk_id citations with rendered (and deduplicated) citations.
    """

    source_by_paper_id = {
        src["paper_id"]: src
        for src in source_lookup.values()
    }

    resolved_answer = []

    for sentence in answer:
        paper_ids = []

        for cid in sentence.get("citations", []):
            source = source_lookup.get(cid)
            if not source:
                continue

            paper_id = source.get("paper_id")
            if not paper_id or paper_id in paper_ids:
                continue

            paper_ids.append(paper_id)

        # Sort deterministically by global citation order
        paper_ids.sort(key=lambda pid: citation_index[pid])

        labels = [
            citation_formatter(pid, source_by_paper_id[pid], citation_index)
            for pid in paper_ids
        ]

        resolved_answer.append({
            "text": sentence["text"],
            "citations": labels
        })

    return resolved_answer



# BUILD SOURCES
  
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
        

def build_sources(citation_index, source_lookup):
    sources = []

    for paper_id, number in sorted(
        citation_index.items(), key=lambda x: x[1]
    ):
        entry = build_source_entry(paper_id, source_lookup)
        entry["citation_number"] = number
        sources.append(entry)

    return sources



# LAYOUT

def remove_citations_inside_text(answer):
    """
    Remove parenthetical citation-like patterns from text.
    """
    cleaned = []

    for sentence in answer:
        text = sentence['text']
        text_no_citations = CHUNK_ID_PATTERN.sub("", text).strip()
        text_no_spaces_before_punctuation = re.sub(r"\s+([.,])", r"\1", text_no_citations)

        cleaned.append({
            "text": text_no_spaces_before_punctuation,
            "citations": sentence['citations']
        })

    return cleaned



# UI HELPERS (presentation-only)

def render_sentence_with_inline_citations(item, citation_style: CitationStyle):
    text = item["text"]
    citations = item.get("citations", [])

    if not citations:
        return f"- {text}"

    if citation_style == CitationStyle.NUMERIC:
        citation_str = ", ".join(citations)
        return f"- {text} [{citation_str}]"

    elif citation_style == CitationStyle.AUTHOR_YEAR:
        citation_str = "; ".join(citations)
        return f"- {text} [{citation_str}]"

    else:
        return f"- {text}"