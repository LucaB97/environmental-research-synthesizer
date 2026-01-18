from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        description="Natural language research question to be answered using the indexed literature",
        example="What are the economic impacts of replacing fossil fuels with wind energy?"
    )
    top_k: int = Field(
        default=15,
        ge=1,
        le=50,
        description="Number of chunks to retrieve for synthesis"
    )


class Sentence(BaseModel):
    text: str = Field(
        ...,
        description="Concise evidence-based statement grounded in the retrieved literature",
        example="Operational and maintenance costs for wind energy are higher than for coal-fired generation."
    )
    citations: List[str] = Field(
        ...,
        description="List of source citations supporting the claim, formatted as 'Author, Year'"
    )


class Source(BaseModel):
    paper_id: str = Field(description="Unique identifier of the source paper")
    title: str = Field(description="Title of the academic paper")
    authors: str = Field(description="Authors of the paper")
    year: int = Field(description="Publication year")
    journal: Optional[str] = Field(
        None,
        description="Journal or conference where the paper was published"
    )


class QueryResponse(BaseModel):
    question: str = Field(description="Original research question")
    reason: Literal[
        "none",
        "out_of_scope",
        "insufficient_evidence",
        "generation_failed"
    ]
    answer: List[Sentence] = Field(
        description="Structured synthesis of the retrieved evidence"
    )
    limitations: List[str] = Field(
        description="Known limitations, uncertainties, or gaps in the available evidence"
    )
    sources: List[Source] = Field(
        description="Academic sources that support the synthesized answer"
    )
    meta: Dict = Field(
        default_factory=dict,
        description="Additional metadata about retrieval and synthesis"
    )
    debug: Dict = Field(
        default_factory=dict,
        description="Debug info"
    )