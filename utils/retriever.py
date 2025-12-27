import numpy as np
import faiss

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

        This method is recall-oriented: it may return multiple chunks
        from the same document, which is useful when providing rich
        context to an LLM.

        Args:
            query (str): Natural language query.
            top_k (int): Number of chunks to retrieve.

        Returns:
            list[dict]: Retrieved chunk dictionaries with metadata.
        """

        query_vec = self.embedding_fn(query).astype("float32")
        query_vec = np.array([query_vec])
        faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(query_vec, top_k)

        return [self.chunks[i] for i in indices[0]]
    

    def display(self, results, max_chars=500, deduplicate=True):
        """
        Display retrieved chunks in a human-readable format with citations.

        Optionally removes duplicate papers for clarity, while leaving
        the original retrieval results unchanged.

        Args:
            results (list[dict]): Retrieved chunks.
            max_chars (int): Maximum number of characters to display per chunk.
            deduplicate (bool): Whether to display only one chunk per paper.
        """

        seen = set()

        for i, r in enumerate(results, 1):
            if deduplicate and r["paper_id"] in seen:
                continue

            seen.add(r["paper_id"])

            print(f"\n[{i}] {r['title']} ({r['year']})")
            print(r["text"][:max_chars] + ("..." if len(r["text"]) > max_chars else ""))