"""
Pydantic models shared across routers/services.
Keep fields minimal for clarity and auditability.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class BranchContext(BaseModel):
    branch_id: int
    branch_name: Optional[str] = None


class KPIRequest(BaseModel):
    branch_id: int
    lookback_days: int = Field(default=30, ge=7, le=120)


class SalesKPI(BaseModel):
    total_revenue: float
    avg_ticket: float
    transactions: int
    wow_change: float
    top_services: List[Dict[str, Any]]
    weekday_mix: List[Dict[str, Any]]


class InventoryKPI(BaseModel):
    items: List[Dict[str, Any]]  # each item: {name, current_stock, avg_daily_use, days_to_out, status}


class KPIResponse(BaseModel):
    branch: BranchContext
    sales: SalesKPI
    inventory: InventoryKPI


class ForecastPoint(BaseModel):
    date: str
    predicted: float


class ForecastResponse(BaseModel):
    branch: BranchContext
    horizon_days: int
    points: List[ForecastPoint]
    model_used: str
    confidence: float
    notes: List[str]


class InventoryRecommendation(BaseModel):
    item_name: str
    action: str  # e.g., "reorder", "monitor", "reduce"
    rationale: str


class InventoryPlan(BaseModel):
    branch: BranchContext
    recommendations: List[InventoryRecommendation]
    assumptions: List[str]


class ChatRequest(BaseModel):
    query: str
    branch_id: Optional[int] = None
    user_role: Optional[str] = "manager"


class ChatResponse(BaseModel):
    reply: str
    reasoning: str
    data_used: Dict[str, Any]
    prompts: Dict[str, Any]  # logged prompts per AI role
    fallback_used: bool = False

