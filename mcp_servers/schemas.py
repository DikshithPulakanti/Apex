# mcp_servers/schemas.py
# Pydantic models validating each MCP tool's input before the handler
# touches it. The wire-level `types.Tool.inputSchema` is still JSON Schema
# (an MCP protocol requirement, not something the SDK lets us change), but
# every handler validates through one of these first — that's the concrete,
# checkable meaning of "typed service" here.

from typing import List, Optional

from pydantic import BaseModel, Field


class CreateHypothesisInput(BaseModel):
    statement: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    supporting_concepts: List[str] = Field(default_factory=list)
    testability_score: float = Field(ge=0.0, le=1.0)
    predicted_impact: str = ''


class SearchPapersInput(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=20)
    alpha: float = Field(default=0.7, ge=0.0, le=1.0)


class SimulationInput(BaseModel):
    hypothesis_statement: str = Field(min_length=1)
    testability_score: float = Field(ge=0.0, le=1.0)
    n_simulations: int = Field(default=1000, ge=1, le=100_000)


class DraftPatentInput(BaseModel):
    hypothesis_statement: str = Field(min_length=1)
    supporting_concepts: List[str] = Field(default_factory=list)
    predicted_impact: str = ''
