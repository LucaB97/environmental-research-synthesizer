import os
import pandas as pd
import json
from pathlib import Path
from typing import Optional

from utils.embeddings import openai_embedding
from utils.indexing import load_faiss
from utils.retriever import SemanticRetriever
from utils.llm_clients import OpenAIClient, HFClient
from utils.cross_encoder import RelevanceGate
from app.utils.synthesis import QueryScopeClassifier, ResearchSynthesisEngine

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = PROJECT_ROOT / "data" / "metadata.csv"
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks_500t_100o.json"
FAISS_PATH = PROJECT_ROOT / "data" / "faiss_openai_500t_100o.index"


def load_system(app, profile: Optional[str] = None):
    
    if profile is None:
        profile = os.getenv("SYNTH_PROFILE", "public")

    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)

    index = load_faiss(FAISS_PATH)
    embedding_fn = openai_embedding

    # ---- LLM selection ----
    if profile == "public":
        llm = OpenAIClient()

    # elif profile == "local":
    #     print("[startup] Loading HF model, this may take a few minutes on first run...")
    #     llm = HFClient("google/flan-t5-base")

    elif profile == "gpu":
        llm = HFClient(
            "mistralai/Mistral-7B-Instruct-v0.2",
            load_in_4bit=True
        )

    else:
        raise ValueError(f"Unknown profile: {profile}")

    # ---- Core components ----
    scope_classifier = QueryScopeClassifier(llm)
    retriever = SemanticRetriever(index, chunks, embedding_fn)
    relevance_gate = RelevanceGate("cross-encoder/ms-marco-MiniLM-L-6-v2")
    synthesizer = ResearchSynthesisEngine(llm, max_attempts=3)

    # ---- Attach to app ----
    app.state.metadata = pd.read_csv(METADATA_PATH).set_index("paper_id")
    app.state.scope_classifier = scope_classifier
    app.state.retriever = retriever
    app.state.relevance_gate = relevance_gate
    app.state.synthesizer = synthesizer