from pydantic import BaseModel
from typing import List, Dict, Optional
from schemas.confidence import GroundingMetrics

class AnalysisTrace(BaseModel):
    queries: Optional[List] = None
    evidence_distribution: Optional[Dict] = None
    grounding_metrics: Optional[GroundingMetrics] = None
    chunks_provided_to_synthesizer: Optional[List[Dict]] = None
    paper_stats: Optional[List[Dict]] = None 