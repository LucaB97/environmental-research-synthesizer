import faiss

from rank_bm25 import BM25Okapi
import re



class SemanticRetriever:
    """
    Semantic retriever for document chunks using vector similarity search.

    This class wraps a FAISS index and an embedding function to retrieve
    relevant text chunks for a given natural language query. 
    It supports retrieval for downstream LLM synthesis.
    """

    def __init__(self, index, chunks, embedding_fn):
        self.index = index
        self.chunks = chunks
        self.embedding_fn = embedding_fn

    
    def search(self, query, top_k=10):
        """
        Retrieve the most semantically similar chunks for a given query.

        Args:
            query (str): Natural language query.
            top_k (int): Number of chunks to retrieve.

        Returns:
            list[dict]: Retrieved chunk dictionaries with metadata.
        """

        query_embedding = self.embedding_fn(query).astype("float32")
        faiss.normalize_L2(query_embedding)

        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), start=1):
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(score)
            chunk["rank"] = rank
            results.append(chunk)

        return results



class BM25Retriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.tokenized_corpus = [self.tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def tokenize(self, text):
        text = text.lower()
        text = re.sub(r"\W+", " ", text)
        return text.split()

    def search(self, query, top_k=10):
        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(scores[idx])
            chunk["rank"] = rank
            results.append(chunk)

        return results
    


class HybridRetriever:
    def __init__(self, semantic_retriever, bm25_retriever):
        self.semantic_retriever = semantic_retriever
        self.bm25_retriever = bm25_retriever

    def search(self, query, topk_faiss=30, topk_bm25=30):
        faiss_results = self.semantic_retriever.search(query, topk_faiss)
        bm25_results = self.bm25_retriever.search(query, topk_bm25)

        combined = {}

        # First insert FAISS results
        for chunk in faiss_results:
            chunk_id = chunk["chunk_id"]
            chunk_copy = chunk.copy()
            chunk_copy["retrieval_type"] = {"faiss"}
            combined[chunk_id] = chunk_copy

        # Then merge BM25 results
        for chunk in bm25_results:
            chunk_id = chunk["chunk_id"]

            if chunk_id not in combined:
                chunk_copy = chunk.copy()
                chunk_copy["retrieval_type"] = {"bm25"}
                combined[chunk_id] = chunk_copy
            else:
                # Retrieved by both methods
                combined[chunk_id]["retrieval_type"].add("bm25")

        return list(combined.values())