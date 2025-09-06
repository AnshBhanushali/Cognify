from pydantic import BaseModel
from typing import List, Optional

class ImageProcessResponse(BaseModel):
    id: str
    filename: str
    timestamp: str
    embedding_dim: int
    embedding: Optional[List[float]] = None
    predicted_label: str 
class AudioProcessResponse(BaseModel):
    id: str
    filename: str
    timestamp: str
    duration_sec: float
    transcript: str
    avg_confidence: float
    is_broken: bool
    broken_words: List[str]
    summary: Optional[str] = None
