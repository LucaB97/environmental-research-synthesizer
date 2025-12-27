import numpy as np
from sentence_transformers import SentenceTransformer


_model_cache = {}

def hf_embedding(text, model_name="all-MiniLM-L6-v2"):
    """
        Generate a semantic embedding for a given text using a Hugging Face
        SentenceTransformer model.

        The model is loaded once and cached in memory to avoid repeated
        initialization during multiple calls.

        Args:
            text (str): Input text to embed.
            model_name (str): Name of the SentenceTransformer model.

        Returns:
            np.ndarray: A 1D NumPy array representing the text embedding.
        """

    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name].encode(text, convert_to_numpy=True)