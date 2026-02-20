from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        description="Natural language research question to be answered using the indexed literature",
        example="What are the economic impacts of replacing fossil fuels with wind energy?"
    )
    top_k_faiss: int = Field(
        default=30,
        ge=1,
        le=100,
        description="Number of chunks to be retrieved based on semantic similarity"
    )
    top_k_bm25: int = Field(
        default=30,
        ge=1,
        le=100,
        description="Number of chunks to be retrieved based on lexical matches"
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
    citation_number: Optional[int] = Field(
        None,
        description="Numeric citation identifier used in the answer"
    )



class AxisProfile(BaseModel):
    level: Literal["Strong", "Moderate", "Weak"]
    score: float = Field(ge=0.0, le=1.0)
    explanation: List[str]


class ConfidenceProfile(BaseModel):
    evidence: AxisProfile = Field(description="Evidence structure strength")
    grounding: AxisProfile = Field(description="Grounding quality of the synthesis")
    status: Literal["Success", "Not applicable"]
    reason: Optional[str] = Field(
        default="",
        description="Reason for \"Not applicable\" status"
    )



class QueryResponse(BaseModel):
    question: str = Field(description="Original research question")
    
    pipeline_status: Literal[
        "success",
        "out_of_scope",
        "retrieval_failed",
        "generation_error"
    ] = Field(
        description="Technical execution status of the pipeline"
    )
    
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
    
    evidence_metrics: Optional[Dict] = Field(
        default=None,
        description="Evidence quality metrics; None if synthesis failed"
    )
    
    confidence: Optional[ConfidenceProfile] = Field(
        default=None,
        description="Overall confidence in the synthesized answer; None if pipeline failed"
    )
    
    debug: Dict = Field(
        default_factory=dict,
        description="Debug info"
    )
    