from fastapi import FastAPI
from fastapi import Request

from app.dependencies import load_system
from app.schemas import QueryRequest, QueryResponse, AnswerBullet
from app.utils.citations import extract_citations, build_sources_from_citations


app = FastAPI(
    title="Environmental Research Synthesizer",
    description="Semantic retrieval + evidence-based synthesis from academic literature",
    version="0.1.0"
)


@app.on_event("startup")
def startup_event():
    load_system(app)


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest, req: Request):
    retriever = req.app.state.retriever
    synthesizer = req.app.state.synthesizer

    retrieved_chunks = retriever.search(
        request.question,
        top_k=request.top_k
    )

    answer = synthesizer.synthesize(
        request.question,
        retrieved_chunks
    )

    if not answer["in_scope"]:
        sources = []
    else:
        cited_refs = extract_citations(answer["answer"])
        sources = build_sources_from_citations(retrieved_chunks, cited_refs)


    return QueryResponse(
        question=request.question,
        in_scope=answer["in_scope"],
        answer=[AnswerBullet(**b) for b in answer["answer"]],
        limitations=answer["limitations"],
        sources=sources,
        meta={"chunks_retrieved": len(retrieved_chunks)}
    )
