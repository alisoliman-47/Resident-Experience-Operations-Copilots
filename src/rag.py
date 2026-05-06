from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
        # Default to TF-IDF for reliability. Opt-in to sentence-transformers when desired.
        if self.use_sentence_transformers:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self._sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
                self._doc_vectors = self._sentence_model.encode(self.texts, normalize_embeddings=True)
                self.embedding_mode = "sentence-transformers"
                return
            except Exception:
                self.embedding_mode = "tfidf"

        self._tfidf_vectorizer = TfidfVectorizer(stop_words="english")
        self._doc_vectors = self._tfidf_vectorizer.fit_transform(self.texts)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        if not query.strip():
            return []

        if self.embedding_mode == "sentence-transformers" and self._sentence_model is not None:
            query_vec = self._sentence_model.encode([query], normalize_embeddings=True)
            sims = np.dot(self._doc_vectors, query_vec[0])
        else:
            if self._tfidf_vectorizer is None:
                return []
            query_vec = self._tfidf_vectorizer.transform([query])
            sims = cosine_similarity(self._doc_vectors, query_vec).ravel()

        ranked_idx = np.argsort(-sims)[:top_k]
        hits: list[RetrievalHit] = []
        for idx in ranked_idx:
            row = self.df.iloc[int(idx)]
            hits.append(
                RetrievalHit(
                    source_id=str(row.get("source_id", "")),
                    category=str(row.get("category", "other")),
                    urgency=str(row.get("urgency", "low")),
                    text=str(row.get("text", "")),
                    score=float(sims[int(idx)]),
                )
            )
        return hits

    def evaluate_answer_quality(self, answer: str, hits: list[RetrievalHit]) -> AnswerQuality:
        if not hits:
            return AnswerQuality(grounded=False, confidence=0.0, warning="No retrieved evidence.")
        evidence_text = " ".join(h.text.lower() for h in hits)
        answer_tokens = set(answer.lower().split())
        if not answer_tokens:
            return AnswerQuality(grounded=False, confidence=0.0, warning="Empty answer.")
        overlap = sum(1 for token in answer_tokens if token in evidence_text)
        confidence = overlap / len(answer_tokens)
        grounded = confidence >= 0.15
        warning = "" if grounded else "Low grounding score: answer may include unsupported claims."
        return AnswerQuality(grounded=grounded, confidence=round(confidence, 3), warning=warning)

    def answer(
        self, question: str, top_k: int = 5, use_ollama: bool = False
    ) -> tuple[str, list[RetrievalHit], AnswerQuality]:
        hits = self.retrieve(question, top_k=top_k)
        if not hits:
            quality = AnswerQuality(grounded=False, confidence=0.0, warning="No evidence retrieved.")
            return "No relevant feedback found for this question.", [], quality

        context_lines = [
            f"- ({h.urgency.upper()} | {h.category}) {h.text}" for h in hits
        ]
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

        # Deterministic fallback answer without LLM.
        categories = pd.Series([h.category for h in hits]).value_counts()
        urgencies = pd.Series([h.urgency for h in hits]).value_counts()
        top_category = categories.index[0] if not categories.empty else "other"
        top_urgency = urgencies.index[0] if not urgencies.empty else "low"

        answer = (
            f"Based on retrieved resident feedback, the dominant issue theme is `{top_category}` "
            f"with mostly `{top_urgency}` urgency.\n\n"
            "Recommended actions:\n"
            "- Prioritize high/urgent tickets in today's operations queue.\n"
            "- Investigate recurring root causes in the dominant category.\n"
            "- Publish resident communication with timeline and next steps.\n"
            "- Track whether complaint volume decreases over the next week."
        )
        quality = self.evaluate_answer_quality(answer, hits)
        return answer, hits, quality
