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
        return strong_hits >= self.min_hits, strong_hits

    
    def debug(self, question, chunks, show_only_relevant=False):
        scores = self.score(question, chunks)
        
        if show_only_relevant:
            return [
                {
                    "chunk_id": c["chunk_id"],
                    "paper_id": c["paper_id"],
                    "text": c["text"],
                    "score": float(score)
                }
                for c, score in zip(chunks, scores) if score >= self.min_score
            ]
        
        else:
            return [
                {
                    "chunk_id": c["chunk_id"],
                    "paper_id": c["paper_id"],
                    "text": c["text"],
                    "score": float(score),
                    "strong_hit": score >= self.min_score
                }
                for c, score in zip(chunks, scores)
            ]
