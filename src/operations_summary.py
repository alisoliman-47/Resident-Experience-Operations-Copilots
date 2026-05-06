from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class WeeklySummary:
    summary_text: str
    recurring_issues: pd.DataFrame
    recommendations: list[str]


def _top_change_text(df: pd.DataFrame) -> str:
    if df["timestamp"].isna().all() or len(df) < 2:
        return "Not enough timestamp history to compute week-over-week change."

    working = df.copy()
    working["week"] = working["timestamp"].dt.to_period("W").astype(str)
    by_week = working.groupby("week").size().sort_index()
    if len(by_week) < 2:
        return "Only one week of data is available so trend change is limited."

    last_week = by_week.iloc[-1]
    prev_week = by_week.iloc[-2]
    if prev_week == 0:
        return "Previous week had zero items; week-over-week change unavailable."
    change_pct = ((last_week - prev_week) / prev_week) * 100
    direction = "increased" if change_pct >= 0 else "decreased"
    return f"Total issues {direction} by {abs(change_pct):.1f}% vs last week."


def build_weekly_summary(df: pd.DataFrame) -> WeeklySummary:
    working = df.copy()

    category_counts = working["category"].value_counts()
    urgency_counts = working["urgency"].value_counts()
    recurring = (
        working.groupby(["category", "text"])
        .size()
        .reset_index(name="mentions")
        .sort_values("mentions", ascending=False)
    )
    recurring = recurring[recurring["mentions"] >= 2].head(10)

    top_category = category_counts.index[0] if not category_counts.empty else "other"
    top_urgency = urgency_counts.index[0] if not urgency_counts.empty else "low"
    top_change = _top_change_text(working)

    summary_text = (
        f"This week, the most common issue category was `{top_category}`. "
        f"Most requests were marked `{top_urgency}` urgency. {top_change}"
    )

    recommendations: list[str] = []
    high_urgent_count = int(working["urgency"].isin(["high", "urgent"]).sum())
    if high_urgent_count > 0:
        recommendations.append(
            f"Prioritize {high_urgent_count} high/urgent requests in the next 24 hours."
        )
    if top_category != "other":
        recommendations.append(
            f"Run a root-cause review for `{top_category}` and assign an owner."
        )
    if not recurring.empty:
        recommendations.append(
            "Recurring complaint patterns detected; create a proactive resident communication update."
        )
    recommendations.append(
        "Track category volume and urgent backlog daily to confirm interventions are reducing complaints."
    )

    return WeeklySummary(
        summary_text=summary_text,
        recurring_issues=recurring,
        recommendations=recommendations,
    )
