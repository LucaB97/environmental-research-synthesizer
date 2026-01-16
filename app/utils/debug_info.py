from collections import defaultdict


def get_debug_info(retrieved_chunks, used_chunk_ids, sentence_citations=None):
    """
    Build structured debug information for evidence tracing.
    sentence_citations: List[Set[paper_id]] per sentence
    """

    chunks = []
    paper_stats = defaultdict(lambda: {
        "chunks_retrieved": 0,
        "chunks_used": 0,
        "title": None,
        "authors": None,
        "year": None,
    })

    for rank, c in enumerate(retrieved_chunks, start=1):
        used = c["chunk_id"] in used_chunk_ids

        chunks.append({
            "chunk_id": c["chunk_id"],
            "paper_id": c["paper_id"],
            "title": c.get("title"),
            "authors": c.get("authors"),
            "year": c.get("year"),
            "text": c.get("text"),
            "similarity": c.get("score"),
            "rank": rank,
            "used_in_synthesis": used,
        })

        p = paper_stats[c["paper_id"]]
        p["chunks_retrieved"] += 1
        p["chunks_used"] += int(used)
        p["title"] = c.get("title")
        p["authors"] = c.get("authors")
        p["year"] = c.get("year")

    retrieved = len(retrieved_chunks)
    used = len(used_chunk_ids)

    unique_papers_retrieved = len(paper_stats)
    unique_papers_used = sum(
        1 for p in paper_stats.values() if p["chunks_used"] > 0
    )

    max_chunks_from_one_paper = max(
        (p["chunks_used"] for p in paper_stats.values()),
        default=0
    )

    paper_dominance = (
        max_chunks_from_one_paper / used if used > 0 else 0.0
    )


    avg_citations_per_sentence = 0.0
    multi_source_sentence_ratio = 0.0

    if sentence_citations:
        total_sentences = len(sentence_citations)

        total_citations = sum(
            len(papers) for papers in sentence_citations
        )

        multi_source_sentences = sum(
            1 for papers in sentence_citations if len(papers) >= 2
        )

        if total_sentences > 0:
            avg_citations_per_sentence = (
                total_citations / total_sentences
            )
            multi_source_sentence_ratio = (
                multi_source_sentences / total_sentences
            )

    return {
        "chunks": chunks,
        "papers": [
            {
                "paper_id": pid,
                **stats
            }
            for pid, stats in paper_stats.items()
        ],
        "metrics": {
            # Retrieval
            "retrieved_chunks": retrieved,
            "used_chunks": used,
            "chunk_coverage": used / retrieved if retrieved else 0.0,
            "retrieved_papers": unique_papers_retrieved,

            # Source diversity
            "used_papers": unique_papers_used,
            "paper_coverage": (
                unique_papers_used / unique_papers_retrieved
                if unique_papers_retrieved else 0.0
            ),
            "paper_dominance": round(paper_dominance, 3),

            # Synthesis depth metrics
            "avg_citations_per_sentence": round(avg_citations_per_sentence, 3),
            "multi_source_sentence_ratio": round(multi_source_sentence_ratio, 3),
        }
    }