from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.classification import CATEGORY_KEYWORDS


@dataclass
class RetrievalHit:
    source_id: str
    category: str
    urgency: str
    text: str
    score: float


@dataclass
class AnswerQuality:
    grounded: bool
    confidence: float
    warning: str


def _meaningful_tokens(text: str) -> list[str]:
    return [
        t
        for t in re.findall(r"[a-z0-9]+", text.lower())
        if t not in ENGLISH_STOP_WORDS and len(t) > 2
    ]


def _time_filtered_df(df: pd.DataFrame, question: str) -> pd.DataFrame:
    """If the question implies 'this month', narrow to the latest month present in data."""
    q = question.lower()
    if "month" not in q and "this month" not in q and "past month" not in q:
        return df
    ts = df["timestamp"]
    if ts.isna().all():
        return df
    latest = ts.max()
    if pd.isna(latest):
        return df
    mask = (ts.dt.year == latest.year) & (ts.dt.month == latest.month)
    sub = df.loc[mask]
    return sub if len(sub) > 0 else df


def _rows_to_hits(sub: pd.DataFrame, score: float = 1.0) -> list[RetrievalHit]:
    hits: list[RetrievalHit] = []
    for _, row in sub.iterrows():
        hits.append(
            RetrievalHit(
                source_id=str(row.get("source_id", "")),
                category=str(row.get("category", "other")),
                urgency=str(row.get("urgency", "low")),
                text=str(row.get("text", "")),
                score=score,
            )
        )
    return hits


def structured_answer(df: pd.DataFrame, question: str) -> tuple[str | None, pd.DataFrame | None]:
    """
    Answer common operational questions directly from the dataframe (not lexical similarity).
    Returns (answer_markdown, evidence_rows) or (None, None) if no pattern matched.
    """
    q = question.lower().strip()
    if df.empty or not q:
        return None, None

    working = _time_filtered_df(df, question)

    # --- Building / tower volume ---
    if (
        re.search(r"\b(building|tower|wings?)\b", q)
        or "by building" in q
        or "per building" in q
        or "which building" in q
    ):
        if "building_id" in working.columns:
            bid = working["building_id"].fillna("").astype(str).str.strip()
            bid = bid.replace("", pd.NA)
            sub = working.loc[bid.notna()]
            if sub.empty:
                sub = working
            counts = sub.groupby(sub["building_id"].fillna("unknown")).size().sort_values(ascending=False)
            if len(counts):
                lines = [f"- **{idx}**: {int(cnt)} item(s)" for idx, cnt in counts.head(5).items()]
                top_b = counts.index[0]
                ev = sub.loc[sub["building_id"].astype(str) == str(top_b)].head(5)
                if ev.empty:
                    ev = sub.head(5)
                return "Feedback volume by building:\n" + "\n".join(lines), ev

    # --- Urgent / high priority ---
    if re.search(r"\b(urgent|high[\s-]*priority|critical|emergency|asap)\b", q):
        sub = working[working["urgency"].isin(["urgent", "high"])]
        if sub.empty:
            return (
                "There are **no** high or urgent items in the current (filtered) dataset.",
                working.head(3),
            )
        lines = []
        for _, row in sub.head(8).iterrows():
            t = str(row.get("text", ""))[:160]
            lines.append(f"- **{row.get('urgency')}** ({row.get('category')}): {t}{'…' if len(str(row.get('text',''))) > 160 else ''}")
        msg = f"**{len(sub)}** high/urgent item(s):\n" + "\n".join(lines)
        return msg, sub.head(5)

    # --- Sentiment ---
    if re.search(r"\b(negative|unhappy|dissatisfied|angry|frustrated)\b", q):
        sub = working[working["sentiment"] == "negative"]
        if sub.empty:
            return "No **negative** sentiment labels in the current data.", working.head(3)
        ex = "\n".join(
            f"- {str(r.get('text', ''))[:140]}{'…' if len(str(r.get('text', ''))) > 140 else ''}"
            for _, r in sub.head(5).iterrows()
        )
        return (
            f"**{len(sub)}** record(s) labeled negative sentiment. Examples:\n{ex}",
            sub.head(5),
        )
    if re.search(r"\b(positive|happy|pleased|satisfied|great)\b", q) and "unhappy" not in q:
        sub = working[working["sentiment"] == "positive"]
        if sub.empty:
            return "No **positive** sentiment labels in the current data.", working.head(3)
        return (
            f"**{len(sub)}** record(s) labeled positive sentiment.",
            sub.head(5),
        )

    # --- How many / total ---
    if re.search(r"\bhow many\b", q) and re.search(
        r"\b(records|issues|items|tickets|complaints?|feedback)\b", q
    ):
        return f"There are **{len(working)}** feedback records in the current view.", working.head(5)

    # --- Topic keywords -> category or text match (e.g. parking, noise, HVAC) ---
    matched_categories: list[str] = []
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            matched_categories.append(cat)
    if matched_categories:
        cat_mask = working["category"].isin(matched_categories)
        sub = working.loc[cat_mask]
        if len(sub) < 2:
            # Also match raw text if classifier missed
            kw_union = [kw for c in matched_categories for kw in CATEGORY_KEYWORDS.get(c, []) if kw in q]
            if kw_union:
                text_lower = working["text"].fillna("").str.lower()
                sub = working.loc[text_lower.str.contains("|".join(re.escape(k) for k in kw_union), na=False)]
        if not sub.empty:
            cats = ", ".join(f"`{c}`" for c in matched_categories)
            lines = [f"- ({r.get('urgency')} | {r.get('category')}) {str(r.get('text',''))[:150]}…" for _, r in sub.head(6).iterrows()]
            return (
                f"Records related to **{cats}** ({len(sub)} match(es)):\n" + "\n".join(lines),
                sub.head(5),
            )

    # --- Top categories / what are people complaining about ---
    if re.search(r"\b(most|top|biggest|main|common|frequent)\b", q) and re.search(
        r"\b(complain|complaint|issue|problem|concern|category|themes?)\b", q
    ):
        vc = working["category"].value_counts().head(5)
        lines = [f"- **{idx}**: {int(cnt)}" for idx, cnt in vc.items()]
        top_cat = vc.index[0]
        ev = working.loc[working["category"] == top_cat].head(5)
        return "Most common issue **categories**:\n" + "\n".join(lines), ev

    return None, None


class ResidentRAGEngine:
    def __init__(self, df: pd.DataFrame, use_sentence_transformers: bool = False) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.texts = self.df["text"].fillna("").astype(str).tolist()
        self.use_sentence_transformers = use_sentence_transformers
        self.embedding_mode = "tfidf"
        self._sentence_model: Any | None = None
        self._tfidf_vectorizer: TfidfVectorizer | None = None
        self._doc_vectors: Any = None
        self._build_index()

    def _build_index(self) -> None:
        if self.use_sentence_transformers:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self._sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
                self._doc_vectors = self._sentence_model.encode(self.texts, normalize_embeddings=True)
                self.embedding_mode = "sentence-transformers"
                return
            except Exception:
                self.embedding_mode = "tfidf"

        self._tfidf_vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            max_features=8192,
        )
        self._doc_vectors = self._tfidf_vectorizer.fit_transform(self.texts)

    def retrieve(self, query: str, top_k: int = 8) -> list[RetrievalHit]:
        if not query.strip() or not self.texts:
            return []

        q_tokens = _meaningful_tokens(query)
        # Expand query with bigrams from docs that share tokens (helps short resident text)
        expansion = " ".join(q_tokens)
        retrieval_query = f"{query} {expansion}"

        if self.embedding_mode == "sentence-transformers" and self._sentence_model is not None:
            query_vec = self._sentence_model.encode([retrieval_query], normalize_embeddings=True)
            sims = np.dot(self._doc_vectors, query_vec[0])
        else:
            if self._tfidf_vectorizer is None:
                return []
            query_vec = self._tfidf_vectorizer.transform([retrieval_query])
            sims = cosine_similarity(self._doc_vectors, query_vec).ravel()

        # Lexical overlap bonus: questions like "which building" still match "unit 1204" poorly without this
        if q_tokens:
            bonuses = np.array(
                [sum(1 for t in q_tokens if t in self.texts[i].lower()) for i in range(len(self.texts))],
                dtype=float,
            )
            bmax = bonuses.max()
            bonus_norm = bonuses / bmax if bmax > 0 else bonuses
            tmax = float(sims.max()) if len(sims) else 0.0
            tfidf_norm = sims / tmax if tmax > 0 else sims
            combined = 0.65 * tfidf_norm + 0.35 * bonus_norm
        else:
            combined = sims

        ranked_idx = np.argsort(-combined)[:top_k]
        hits: list[RetrievalHit] = []
        for idx in ranked_idx:
            row = self.df.iloc[int(idx)]
            hits.append(
                RetrievalHit(
                    source_id=str(row.get("source_id", "")),
                    category=str(row.get("category", "other")),
                    urgency=str(row.get("urgency", "low")),
                    text=str(row.get("text", "")),
                    score=float(combined[int(idx)]),
                )
            )
        return hits

    def evaluate_answer_quality(self, answer: str, hits: list[RetrievalHit]) -> AnswerQuality:
        if not hits:
            return AnswerQuality(grounded=False, confidence=0.0, warning="No retrieved evidence.")
        evidence_text = " ".join(h.text.lower() for h in hits)
        # Ignore markdown/formatting tokens for overlap
        answer_clean = re.sub(r"[*_#`]", " ", answer)
        answer_tokens = set(_meaningful_tokens(answer_clean))
        if not answer_tokens:
            answer_tokens = set(answer_clean.lower().split())
        if not answer_tokens:
            return AnswerQuality(grounded=False, confidence=0.0, warning="Empty answer.")
        overlap = sum(1 for token in answer_tokens if token in evidence_text)
        confidence = overlap / max(len(answer_tokens), 1)
        grounded = confidence >= 0.12
        warning = "" if grounded else "Low grounding score: answer may include unsupported claims."
        return AnswerQuality(grounded=grounded, confidence=round(confidence, 3), warning=warning)

    def _fallback_answer_from_hits(self, question: str, hits: list[RetrievalHit]) -> str:
        """When no LLM: show ranked evidence snippets so any question gets a useful response."""
        lines = []
        for i, h in enumerate(hits[:6], 1):
            excerpt = h.text[:220] + ("…" if len(h.text) > 220 else "")
            lines.append(f"{i}. **[{h.category} | {h.urgency}]** {excerpt}")
        cats = pd.Series([h.category for h in hits])
        top_cat = cats.mode().iloc[0] if len(cats) else "mixed"
        return (
            f"**Retrieval-ranked resident feedback** for: _{question}_\n\n"
            + "\n".join(lines)
            + f"\n\n**Dominant theme in these results:** `{top_cat}`.\n\n"
            "**Next steps:** Prioritize urgent/high items, verify on-site, and communicate timelines to residents."
        )

    def answer(
        self, question: str, top_k: int = 8, use_ollama: bool = False
    ) -> tuple[str, list[RetrievalHit], AnswerQuality]:
        working_df = _time_filtered_df(self.df, question)

        structured_text, evidence_df = structured_answer(working_df, question)
        if structured_text is not None and evidence_df is not None:
            hits = _rows_to_hits(evidence_df, score=1.0)
            quality = AnswerQuality(grounded=True, confidence=0.92, warning="")
            return structured_text, hits, quality

        # Subset changed: rebuild a lightweight retriever for TF-IDF (avoid double ST model load)
        if len(working_df) != len(self.df) or not working_df.index.equals(self.df.index):
            sub_engine = ResidentRAGEngine(working_df, use_sentence_transformers=False)
            hits = sub_engine.retrieve(question, top_k=top_k)
        else:
            hits = self.retrieve(question, top_k=top_k)

        if not hits:
            quality = AnswerQuality(grounded=False, confidence=0.0, warning="No evidence retrieved.")
            return "No relevant feedback found for this question.", [], quality

        context_lines = [f"- ({h.urgency.upper()} | {h.category}) {h.text}" for h in hits]
        context = "\n".join(context_lines)

        if use_ollama:
            try:
                import ollama

                prompt = (
                    "You are an operations copilot for a residential property manager.\n"
                    "Answer only from the provided context. If uncertain, say so.\n\n"
                    f"Question: {question}\n\n"
                    f"Context:\n{context}\n\n"
                    "Return a concise answer with 2-4 bullet action recommendations."
                )
                response = ollama.chat(
                    model="llama3.1",
                    messages=[{"role": "user", "content": prompt}],
                )
                answer = response["message"]["content"]
                quality = self.evaluate_answer_quality(answer, hits)
                if not quality.grounded:
                    answer += "\n\nNote: This answer has low evidence grounding confidence."
                return answer, hits, quality
            except Exception:
                pass

        answer = self._fallback_answer_from_hits(question, hits)
        quality = self.evaluate_answer_quality(answer, hits)
        return answer, hits, quality
