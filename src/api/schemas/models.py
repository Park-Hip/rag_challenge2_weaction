from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = Field(default=None, description="Optional user ID for Langfuse tracing")
    session_id: Optional[str] = Field(default=None, description="Optional session ID for Langfuse tracing")

class SourceNode(BaseModel):
    text: str
    score: float
    source_file: str
    id: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceNode]
    trace_id: Optional[str] = Field(default=None, description="Langfuse Trace ID for client-side feedback logging")