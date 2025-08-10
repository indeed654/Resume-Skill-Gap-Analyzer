from pydantic import BaseModel
from typing import List, Dict, Any

class AnalysisResponse(BaseModel):
    match_score: float
    resume_skills: List[str]
    jd_skills: List[str]
    matched_skills: List[str]
    missing_skills: List[str]
    learning_plan: Dict[str, Any]
