from collections import defaultdict

def aggregate_evidence(retrieved_chunks, used_chunk_ids):
    """
    Aggregate evidence at chunk and paper level.
    This is the single source of truth for evidence statistics.
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

    return {
        "chunks": chunks,
        "paper_stats": paper_stats,
    }



def compute_evidence_metrics(aggregation, sentence_citations):
    paper_stats = aggregation["paper_stats"]

    retrieved_chunks = sum(
        p["chunks_retrieved"] for p in paper_stats.values()
    )
    used_chunks = sum(
        p["chunks_used"] for p in paper_stats.values()
    )

    retrieved_papers = len(paper_stats)
    used_papers = sum(
        1 for p in paper_stats.values() if p["chunks_used"] > 0
    )

    max_chunks_from_one_paper = max(
        (p["chunks_used"] for p in paper_stats.values()),
        default=0
    )

    paper_dominance = (
        max_chunks_from_one_paper / used_chunks if used_chunks else 0.0
    )

    total_sentences = len(sentence_citations)
    total_citations = sum(len(c) for c in sentence_citations)

    avg_citations_per_sentence = (
        total_citations / total_sentences if total_sentences else 0.0
    )

    multi_source_sentences = sum(
        1 for c in sentence_citations if len(c) >= 2
    )

    multi_source_ratio = (
        multi_source_sentences / total_sentences if total_sentences else 0.0
    )

    return {
        "retrieved_chunks": retrieved_chunks,
        "used_chunks": used_chunks,
        "chunk_coverage": used_chunks / retrieved_chunks if retrieved_chunks else 0.0,

        "retrieved_papers": retrieved_papers,
        "used_papers": used_papers,
        "paper_coverage": (
            used_papers / retrieved_papers if retrieved_papers else 0.0
        ),

        "paper_dominance": round(paper_dominance, 3),
        "avg_citations_per_sentence": round(avg_citations_per_sentence, 2),
        "multi_source_sentence_ratio": round(multi_source_ratio, 2),
    }



def get_debug_info(aggregation, metrics):
    return {
        "chunks": aggregation["chunks"],
        "papers": [
            {"paper_id": pid, **stats}
            for pid, stats in aggregation["paper_stats"].items()
        ],
        "metrics": metrics,
    }