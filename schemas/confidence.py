from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal, Union


class AxisProfile(BaseModel):
    level: Literal["Strong", "Moderate", "Weak", "Not_applicable"]
    score: Optional[float] = Field(ge=0.0, le=1.0)
    explanation: Optional[
        Union[
            str,  # for semantic axis
            List[str]  # for evidence / grounding
        ]
    ] = None


class ConfidenceProfile(BaseModel):
    semantic: AxisProfile = Field(
        default_factory=lambda: AxisProfile(level="Not_applicable", score=None),
        description="Semantic alignment of evidence to query"
    )
    evidence: AxisProfile = Field(
        default_factory=lambda: AxisProfile(level="Not_applicable", score=None),
        description="Evidence structure strength"
    )
    grounding: AxisProfile = Field(
        default_factory=lambda: AxisProfile(level="Not_applicable", score=None),
        description="Grounding quality of the synthesis"
    )
    status: Literal["Success", "Not applicable"]
    reason: Optional[str] = Field(
        default="",
        description="Reason for \"Not applicable\" status"
    )


class GroundingMetrics(BaseModel):
    available_chunks: int
    used_chunks: int
    chunk_coverage: float

    available_papers: int
    used_papers: int
    paper_dominance: float

    avg_citations_per_sentence: float
    multi_source_sentence_ratio: float