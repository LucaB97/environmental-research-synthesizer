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
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", floor: float = 0.25):
        self.model_name = model_name
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

    
    def rerank(self, question, chunks):
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
        scores = self.score(question, chunks)
        
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