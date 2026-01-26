from sentence_transformers import CrossEncoder


class RelevanceGate:
    def __init__(
        self,
        model_name: str,
        min_score: float = 0.5,
        min_hits: int = 2
    ):
        self.encoder = CrossEncoder(model_name)
        self.min_score = min_score
        self.min_hits = min_hits

    
    def score(self, question, chunks):
        pairs = [(question, c["text"]) for c in chunks]
        return self.encoder.predict(pairs)

    
    def is_relevant(self, question, chunks):
        scores = self.score(question, chunks)
        strong_hits = sum(s >= self.min_score for s in scores)
        return strong_hits >= self.min_hits

    
    def debug(self, question, chunks):
        scores = self.score(question, chunks)
        return [
            {
                "chunk_id": c["chunk_id"],
                "score": float(score),
                "paper_id": c["paper_id"]
            }
            for c, score in zip(chunks, scores)
        ]
