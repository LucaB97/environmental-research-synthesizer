class InitializationConfig:
    def __init__(
        self,
        chunk_size,
        overlap,
        embedding,
        topN,
        auto_build,
        normalize_query_lexical,
        lemmatize_query_lexical
        
    ):
        if overlap >= chunk_size:
            raise ValueError("Overlap must be smaller than chunk size.")

        if embedding not in ["hf", "openai"]:
            raise ValueError("Embedding must be 'hf' or 'openai'")
        
        if topN < 10 or topN > 30:
            raise ValueError("Top N must be in the range [10,30]")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.embedding = embedding
        self.topN = topN
        self.auto_build = auto_build
        self.normalize_query_lexical = normalize_query_lexical
        self.lemmatize_query_lexical = lemmatize_query_lexical


DEFAULT_CONFIG = InitializationConfig(
    chunk_size=500,
    overlap=100,
    embedding="hf",
    topN=15,
    auto_build=True,
    normalize_query_lexical=True, #recommended to improve system stability for similar queries
    lemmatize_query_lexical=True
)