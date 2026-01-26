import json
from pathlib import Path

from utils.embeddings import hf_embedding
from utils.indexing import load_faiss
from utils.retriever import SemanticRetriever
from utils.synthesizer import OpenAIClient, ResearchSynthesisEngine
from utils.cross_encoder import RelevanceGate

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks.json"
FAISS_PATH = PROJECT_ROOT / "data" / "faiss.index"


def load_system(app):
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    with open(PROJECT_ROOT / "data" / "chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)

    index = load_faiss(PROJECT_ROOT / "data" / "faiss.index")

    retriever = SemanticRetriever(index, chunks, hf_embedding)
    llm = OpenAIClient()
    synthesizer = ResearchSynthesisEngine(llm, 3)
    relevance_gate = RelevanceGate("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Store everything on app.state
    app.state.retriever = retriever
    app.state.synthesizer = synthesizer
    app.state.relevance_gate = relevance_gate