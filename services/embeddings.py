import numpy as np


class HFEmbedding:
    """
    Callable HuggingFace SentenceTransformer embedding wrapper
    with internal model caching.
    """
    
    _model_cache = {}

    def __init__(self, model: str = "all-MiniLM-L6-v2", device=None):
        from sentence_transformers import SentenceTransformer
        import torch

        self.model_name = model

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if model not in HFEmbedding._model_cache:
            HFEmbedding._model_cache[model] = SentenceTransformer(
                model,
                device=self.device
            )
            
        self.model = HFEmbedding._model_cache[model]


    def __call__(self, texts):
        
        if not texts:
            return np.array([])

        if isinstance(texts, str):
            texts = [texts]

        return self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True
        )



class OpenAIEmbedding:
    """
    Callable OpenAI embedding wrapper that exposes model metadata.
    """

    def __init__(self, model="text-embedding-3-small", batch_size=200):
        self.model_name = model
        self.batch_size = batch_size

    def __call__(self, texts):
        from openai import OpenAI
        import os

        key = os.getenv("OPENAI_API_KEY")
        if key is None:
            raise ValueError("No OpenAI API key found.")
        
        client = OpenAI(api_key=key)

        if not texts:
            return np.array([])

        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = client.embeddings.create(
                model=self.model_name,
                input=batch
            )
            all_embeddings.extend([item.embedding for item in response.data])

        return np.array(all_embeddings, dtype="float32")