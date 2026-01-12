"""
Forecasting AI using Google Gemini (free tier).
Stub with deterministic fallback; keep payload small and auditable.
"""
import logging
from typing import Dict, Any, List

from config import get_settings
from ai.prompt_utils import log_prompt, envelope

logger = logging.getLogger(__name__)


def forecast_revenue(daily_history: List[Dict[str, Any]], horizon_days: int = 7) -> Dict[str, Any]:
    """
    daily_history: list of {date: str, revenue: float}
    Returns dict: points [{date, predicted}], model_used, prompt_used, fallback_used.
    """
    settings = get_settings()
    prompt = envelope(
        system="You predict near-term revenue for a salon using recent daily totals.",
        task=f"Predict the next {horizon_days} days of revenue as numbers only.",
        context={"daily_history": daily_history[-60:]},  # clamp context
    )
    log_prompt("forecast", prompt)

    if not settings.gemini_api_key:
        # Simple moving-average fallback
        values = [row.get("revenue", 0) for row in daily_history[-7:]] or [0]
        avg = sum(values) / len(values)
        points = []
        last_date = daily_history[-1]["date"] if daily_history else "2024-01-01"
        # Do not compute real dates to avoid tz issues; caller can map if needed
        for i in range(1, horizon_days + 1):
            points.append({"date": f"{last_date}+{i}", "predicted": round(avg, 2)})
        return {
            "points": points,
            "model_used": "moving-average-fallback",
            "prompt_used": prompt,
            "fallback_used": True,
        }

    # TODO: Implement real Gemini call.
    return {
        "points": [],
        "model_used": "gemini-placeholder",
        "prompt_used": prompt,
        "fallback_used": False,
    }

