import json
from pathlib import Path

from utils.embeddings import hf_embedding
from utils.indexing import load_faiss
from utils.retriever import SemanticRetriever
from utils.synthesizer import OpenAIClient, ResearchSynthesisEngine
from app.utils.synthesis_prompt import SYNTHESIS_PROMPT_TEMPLATE
from app.schemas import QueryRequest, QueryResponse, Sentence
from app.utils.citations import resolve_answer_citations, build_sources_from_used_chunks



PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks.json"
FAISS_PATH = PROJECT_ROOT / "data" / "faiss.index"

with open(PROJECT_ROOT / "data" / "chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)
index = load_faiss(PROJECT_ROOT / "data" / "faiss.index")
retriever = SemanticRetriever(index, chunks, hf_embedding)
llm = OpenAIClient()
synthesizer = ResearchSynthesisEngine(llm, SYNTHESIS_PROMPT_TEMPLATE)

query = "How are the costs and benefits of renewable energy adoption distributed on society?"
retrieved_chunks = retriever.search(
    query,
    top_k=5
)

source_lookup = {
    c["chunk_id"]: c
    for c in retrieved_chunks
}

synthesis_output = synthesizer.synthesize(
    query,
    retrieved_chunks
)

resolved_answer = resolve_answer_citations(
    synthesis_output["answer"],
    source_lookup
)

used_chunks = {
    cid
    for sentence in synthesis_output["answer"]
    for cid in sentence["citations"]
}
sources = build_sources_from_used_chunks(used_chunks, source_lookup)

synthesis = QueryResponse(
        question=query,
        reason=synthesis_output["reason"],
        answer=[Sentence(**s) for s in resolved_answer],
        limitations=synthesis_output["limitations"],
        sources=sources,
        meta={
            "chunks_retrieved": len(retrieved_chunks)
        }
    )

print(synthesis)
