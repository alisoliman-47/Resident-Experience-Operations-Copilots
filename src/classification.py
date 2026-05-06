from __future__ import annotations

from dataclasses import dataclass
import re

import pandas as pd
from textblob import TextBlob


CATEGORY_KEYWORDS = {
    "maintenance": ["leak", "hvac", "repair", "broken", "plumbing", "electrical", "maintenance"],
    "noise": ["noise", "loud", "music", "neighbors", "midnight"],
    "parking": ["parking", "garage", "gate", "car", "vehicle"],
    "safety": ["safety", "security", "unsafe", "threat", "fire", "smoke"],
    "amenities": ["gym", "pool", "amenity", "equipment", "clubhouse"],
    "cleanliness": ["trash", "dirty", "clean", "hallway", "odor", "smell"],
    "leasing": ["lease", "rent", "billing", "payment", "contract"],
    "events": ["event", "activity", "program", "community", "social"],
}

URGENCY_KEYWORDS = {
    "urgent": ["water leak", "fire", "smoke", "flood", "unsafe", "security issue", "no heat"],
    "high": ["broken", "stuck", "not working", "urgent", "asap", "immediately", "hvac failure"],
    "medium": ["noise", "parking", "delay", "frustrated", "complaint"],
}

URGENCY_SCORES = {"low": 1, "medium": 2, "high": 3, "urgent": 4}


@dataclass
class ClassificationResult:
    enriched_df: pd.DataFrame


def _match_category(text: str) -> str:
    lowered = text.lower()
    best_category = "other"
    best_score = 0
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in lowered)
        if score > best_score:
            best_score = score
            best_category = category
    return best_category


def _sentiment_label(text: str) -> str:
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.2:
        return "positive"
    if polarity < -0.2:
        return "negative"
    return "neutral"


def _urgency_level(text: str, category: str) -> str:
    lowered = text.lower()
    for level in ("urgent", "high", "medium"):
        if any(keyword in lowered for keyword in URGENCY_KEYWORDS[level]):
            return level
    if category in {"safety", "maintenance"} and re.search(r"\b(issue|problem|broken|leak)\b", lowered):
        return "high"
    return "low"


def enrich_with_classification(df: pd.DataFrame) -> ClassificationResult:
    working_df = df.copy()
    working_df["category"] = working_df["text"].fillna("").astype(str).apply(_match_category)
    working_df["sentiment"] = working_df["text"].fillna("").astype(str).apply(_sentiment_label)
    working_df["urgency"] = working_df.apply(
        lambda row: _urgency_level(str(row["text"]), str(row["category"])),
        axis=1,
    )
    working_df["urgency_score"] = working_df["urgency"].map(URGENCY_SCORES).fillna(1).astype(int)
    working_df = working_df.sort_values(by=["urgency_score", "timestamp"], ascending=[False, False])
    return ClassificationResult(enriched_df=working_df)
