import numpy as np
import faiss


def build_faiss_index(embeddings):
    """
    Build a FAISS index for fast similarity search using cosine similarity.

    The embeddings are L2-normalized so that inner product corresponds
    to cosine similarity.

    Args:
        embeddings (np.ndarray): Array of shape (n_chunks, embedding_dim).

    Returns:
        faiss.Index: A FAISS index containing the embeddings.
    """

    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_faiss(index, path):
    """
    Persist a FAISS index to disk.

    Args:
        index (faiss.Index): FAISS index to save.
        path (str or Path): Output file path.
    """
    
    faiss.write_index(index, str(path))


def load_faiss(path):
    """
    Load a FAISS index from disk.

    Args:
        path (str or Path): Path to the saved FAISS index.

    Returns:
        faiss.Index: Loaded FAISS index.
    """

    return faiss.read_index(str(path))