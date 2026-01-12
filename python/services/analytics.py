"""
Analytics core (deterministic, no AI calls).
Computes sales and inventory KPIs for explainable dashboards and prompts.
"""
import logging
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def _safe_percent(numerator: float, denominator: float) -> float:
    return (numerator / denominator) if denominator else 0.0


def compute_sales_kpis(sales_rows: List[Dict[str, Any]], lookback_days: int = 30) -> Dict[str, Any]:
    """
    sales_rows expected keys: date (datetime/date/str), amount (numeric), service_name (optional).
    """
    if not sales_rows:
        return {
            "total_revenue": 0.0,
            "avg_ticket": 0.0,
            "transactions": 0,
            "wow_change": 0.0,
            "top_services": [],
            "weekday_mix": [],
        }

    df = pd.DataFrame(sales_rows)
    if "date" not in df.columns or "amount" not in df.columns:
        raise ValueError("sales_rows must include 'date' and 'amount'")

    df["date"] = pd.to_datetime(df["date"])
    cutoff = df["date"].max() - timedelta(days=lookback_days)
    df = df[df["date"] >= cutoff]

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    df = df[df["amount"] > 0]

    total_revenue = float(df["amount"].sum())
    transactions = int(len(df))
    avg_ticket = float(total_revenue / transactions) if transactions else 0.0

    # Week-over-week change (last 7 vs previous 7)
    latest_date = df["date"].max()
    week1_start = latest_date - timedelta(days=6)
    week0_start = week1_start - timedelta(days=7)

    week1 = df[(df["date"] >= week1_start) & (df["date"] <= latest_date)]["amount"].sum()
    week0 = df[(df["date"] >= week0_start) & (df["date"] < week1_start)]["amount"].sum()
    wow_change = _safe_percent(week1 - week0, week0) if week0 else 0.0

    # Top services
    top_services = []
    if "service_name" in df.columns:
        svc_counts = df["service_name"].fillna("Unknown").value_counts().head(5)
        for name, count in svc_counts.items():
            top_services.append({"service_name": name, "transactions": int(count)})

    # Weekday mix (% of revenue per weekday)
    df["weekday"] = df["date"].dt.day_name()
    weekday_sums = df.groupby("weekday")["amount"].sum()
    total_for_mix = float(weekday_sums.sum())
    weekday_mix = [
        {"weekday": day, "share": _safe_percent(amount, total_for_mix)}
        for day, amount in weekday_sums.sort_values(ascending=False).items()
    ]

    return {
        "total_revenue": round(total_revenue, 2),
        "avg_ticket": round(avg_ticket, 2),
        "transactions": transactions,
        "wow_change": round(wow_change, 4),
        "top_services": top_services,
        "weekday_mix": weekday_mix,
    }


def compute_inventory_kpis(
    usage_rows: List[Dict[str, Any]],
    stock_rows: List[Dict[str, Any]],
    lookback_days: int = 30,
) -> Dict[str, Any]:
    """
    usage_rows keys: date, item_name, quantity_used
    stock_rows keys: item_name, current_stock
    """
    usage_df = pd.DataFrame(usage_rows) if usage_rows else pd.DataFrame(columns=["date", "item_name", "quantity_used"])
    stock_df = pd.DataFrame(stock_rows) if stock_rows else pd.DataFrame(columns=["item_name", "current_stock"])

    if not len(stock_df):
        return {"items": []}

    usage_df["date"] = pd.to_datetime(usage_df["date"], errors="coerce")
    usage_df["quantity_used"] = pd.to_numeric(usage_df["quantity_used"], errors="coerce").fillna(0)

    cutoff = usage_df["date"].max() - timedelta(days=lookback_days) if len(usage_df) else None
    if cutoff:
        usage_df = usage_df[usage_df["date"] >= cutoff]

    avg_use = usage_df.groupby("item_name")["quantity_used"].mean() if len(usage_df) else pd.Series(dtype=float)
    results = []
    for _, row in stock_df.iterrows():
        name = row.get("item_name")
        current_stock = float(row.get("current_stock", 0) or 0)
        daily_use = float(avg_use.get(name, 0.0))
        days_to_out = (current_stock / daily_use) if daily_use else float("inf")

        if daily_use == 0:
            status = "no-usage-data"
        elif days_to_out < 7:
            status = "urgent-reorder"
        elif days_to_out < 14:
            status = "reorder-soon"
        else:
            status = "ok"

        results.append(
            {
                "item_name": name,
                "current_stock": current_stock,
                "avg_daily_use": round(daily_use, 2),
                "days_to_out": round(days_to_out, 2) if days_to_out != float("inf") else None,
                "status": status,
            }
        )

    return {"items": results}

