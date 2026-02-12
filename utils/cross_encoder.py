import numpy as np
from sentence_transformers import CrossEncoder


class RelevanceProfiler:
    """
    Reranks and profiles retrieved document chunks for a given natural language query
    using a cross-encoder.

    Attributes:
        encoder (CrossEncoder): Cross-encoder model for scoring question-chunk pairs
        floor (float): Minimum score to consider a chunk as relevant (floor safeguard)
    """
    
    def __init__(self, model_name: str, floor: float = 0.25):
        self.encoder = CrossEncoder(model_name)
        self.floor = floor

    
    def score(self, question, chunks):
        """
        Compute cross-encoder relevance scores for each chunk with respect to a query.

        Args:
            question (str): The natural language question to retrieve evidence for.
            chunks (list[dict]): List of chunk dictionaries, each with at least a "text" field.

        Returns:
            np.ndarray: Array of float relevance scores (one per chunk).
        """

        pairs = [(question, c["text"]) for c in chunks]
        scores = self.encoder.predict(pairs)
        return np.array(scores)

    
    def rerank(self, chunks, scores):
        """
        Rerank chunks in descending order of relevance and annotate metadata.

        Args:
            chunks (list[dict]): List of chunk dictionaries (will be copied internally)
            scores (list[float] or np.ndarray): Cross-encoder scores for each chunk

        Returns:
            list[dict]: List of chunk dictionaries, each augmented with:
                - 'final_score': float, cross-encoder score
                - 'final_rank': int, position in descending order of score (1 = highest)
        """
        ranked = sorted(
            zip(chunks, scores),
            key=lambda x: x[1],
            reverse=True
        )
        for rank, (chunk, score) in enumerate(ranked, start=1):
            chunk["final_score"] = float(score)
            chunk["final_rank"] = rank
            # chunk["strong_hit"] = score >= self.floor
        return [c for c, s in ranked]


    def classify_evidence(self, scores):
        """
        Classify the overall evidence distribution based on scores.

        Args:
            scores (list[float] or np.ndarray): Cross-encoder scores for retrieved chunks.

        Returns:
            str: Evidence label, one of:
                - 'absent': No chunk meets the absolute floor (no real signal)
                - 'isolated': One dominant chunk, others weak
                - 'fragmented': Small scattered signal, no clear cluster
                - 'thematic': Several moderately strong chunks
                - 'robust': Multiple strong chunks
                - 'weak': Low/moderate, indeterminate structure

        Notes:
            - Z-score normalization is applied to measure relative salience
            - Floor safeguard ensures weak/noisy distributions are classified as 'absent'
        """
        mean = scores.mean()
        std = scores.std()
        max_score = scores.max()

        # Floor safeguard
        if max_score < self.floor:
            return "absent"

        if std < 1e-6:
            z = np.zeros_like(scores)
        else:
            z = (scores - mean) / std

        max_z = z.max()
        strong_hits = np.sum(z > 1)
        moderate_hits = np.sum(z > 0.5)

        if strong_hits == 1 and max_z > 2:
            return "isolated"
        elif strong_hits >= 3:
            return "robust"
        elif moderate_hits >= 3:
            return "thematic"
        elif strong_hits <= 2 and moderate_hits <= 2:
            return "fragmented"
        else:
            return "weak"


    def rerank_and_profile(self, question, chunks):
        """
        Convenience method to combine scoring, reranking, and evidence profiling.
        """
        scores = self.score(question, chunks)
        ranked_chunks = self.rerank(chunks, scores)
        evidence_label = self.classify_evidence(scores)
        
        return {
            "ranked_chunks": ranked_chunks,
            "evidence_label": evidence_label,
            "metrics": {
                # "scores": scores.tolist(),
                "mean": float(scores.mean()),
                "std": float(scores.std()),
                "max": float(scores.max()),
                "strong_hits": int(np.sum((scores - scores.mean()) / max(scores.std(),1e-6) > 1))
            }
        }
    

    def get_strong_hits(self, ranked_chunks, z_threshold=1.0):
        """
        Return chunks that are strong hits based on Z-score among the ranked chunks.

        Args:
            ranked_chunks (list[dict]): Chunks already annotated with 'score' (from rerank/profile)
            z_threshold (float): Minimum z-score to consider a chunk a strong hit (default=1.0)

        Returns:
            list[dict]: Chunks with z-score >= z_threshold, annotated with their z-score
        """
        scores = np.array([c["score"] for c in ranked_chunks])
        mean = scores.mean()
        std = scores.std()
        if std < 1e-6:
            z_scores = np.zeros_like(scores)
        else:
            z_scores = (scores - mean) / std

        strong_hits = []
        for chunk, z in zip(ranked_chunks, z_scores):
            if z >= z_threshold:
                chunk_copy = chunk.copy()
                chunk_copy["z_score"] = float(z)
                strong_hits.append(chunk_copy)

        return strong_hits


    # def is_relevant(self, question, chunks):
    #     scores = self.score(question, chunks)
    #     strong_hits = sum(s >= self.min_score for s in scores)
    #     return strong_hits >= self.min_hits, strong_hits

    
    # def debug(self, question, chunks, show_only_relevant=False):
    #     scores = self.score(question, chunks)
        
    #     if show_only_relevant:
    #         return [
    #             {
    #                 "chunk_id": c["chunk_id"],
    #                 "paper_id": c["paper_id"],
    #                 "text": c["text"],
    #                 "score": float(score)
    #             }
    #             for c, score in zip(chunks, scores) if score >= self.min_score
    #         ]
        
    #     else:
    #         return [
    #             {
    #                 "chunk_id": c["chunk_id"],
    #                 "paper_id": c["paper_id"],
    #                 "text": c["text"],
    #                 "score": float(score),
    #                 "strong_hit": score >= self.min_score
    #             }
    #             for c, score in zip(chunks, scores)
    #         ]
