"""Core NLP logic: prompt loading, review analysis, validation, aggregation."""
import json
import logging
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import yaml

log = logging.getLogger(__name__)

SENTIMENTS = {"positive", "neutral", "negative"}
URGENCIES = {"low", "medium", "high"}


def load_prompts(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_json(text: str) -> dict:
    """Extract a JSON object from an LLM reply (handles ```json fences / extra text)."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            return json.loads(match.group())
        raise ValueError("No JSON object found in LLM output")


def validate(data: dict) -> dict:
    """Check and normalise the LLM output so downstream code can trust it."""
    sentiment = str(data.get("sentiment", "")).lower()
    if sentiment not in SENTIMENTS:
        raise ValueError(f"Invalid sentiment: {sentiment!r}")
    urgency = str(data.get("urgency", "low")).lower()
    if urgency not in URGENCIES:
        urgency = "low"
    rating = max(1, min(5, int(data.get("rating_estimate", 3))))
    aspects = [
        {"aspect": str(a.get("aspect", "")).strip().lower(),
         "sentiment": str(a.get("sentiment", "neutral")).lower()}
        for a in data.get("aspects", []) if isinstance(a, dict) and a.get("aspect")
    ]
    return {
        "sentiment": sentiment,
        "rating_estimate": rating,
        "emotion": str(data.get("emotion", "")).lower(),
        "aspects": aspects,
        "key_issue": str(data.get("key_issue", "None")),
        "urgency": urgency,
        "suggested_reply": str(data.get("suggested_reply", "")),
    }


class ReviewAnalyzer:
    def __init__(self, client, prompts: dict, parse_retries: int = 2, max_workers: int = 4):
        self.client = client
        self.prompts = prompts
        self.parse_retries = parse_retries
        self.max_workers = max_workers

    def analyze_review(self, review: str) -> dict:
        """Analyze one review. Re-asks the LLM if the JSON is malformed."""
        user = self.prompts["analyze_review"].replace("{review}", review.strip())
        last_err = None
        for _ in range(self.parse_retries + 1):
            raw = self.client.chat(self.prompts["system"], user, task="review")
            try:
                return validate(parse_json(raw))
            except (ValueError, TypeError, KeyError) as exc:
                last_err = exc
                log.warning("Bad LLM output (%s); asking for a repair", exc)
                user = user + "\n\n" + self.prompts["repair"]
        raise ValueError(f"Could not get valid analysis: {last_err}")

    def analyze_batch(self, reviews: list[str]) -> list[dict]:
        """Analyze many reviews in parallel; failures are recorded, not fatal."""
        def work(item):
            idx, text = item
            try:
                return {"id": idx, "review": text, **self.analyze_review(text)}
            except Exception as exc:
                log.error("Review %s failed: %s", idx, exc)
                return {"id": idx, "review": text, "error": str(exc)}

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(work, enumerate(reviews, start=1)))

    @staticmethod
    def aggregate(results: list[dict]) -> dict:
        ok = [r for r in results if "error" not in r]
        neg_aspects = Counter(a["aspect"] for r in ok for a in r["aspects"] if a["sentiment"] == "negative")
        pos_aspects = Counter(a["aspect"] for r in ok for a in r["aspects"] if a["sentiment"] == "positive")
        return {
            "count": len(ok),
            "failed": len(results) - len(ok),
            "sentiment_counts": dict(Counter(r["sentiment"] for r in ok)),
            "avg_rating": round(sum(r["rating_estimate"] for r in ok) / len(ok), 2) if ok else 0,
            "top_negative_aspects": neg_aspects.most_common(5),
            "top_positive_aspects": pos_aspects.most_common(5),
            "high_urgency": [{"id": r["id"], "issue": r["key_issue"]} for r in ok if r["urgency"] == "high"],
        }

    def executive_summary(self, stats: dict) -> str:
        prompt = (self.prompts["executive_summary"]
                  .replace("{count}", str(stats["count"]))
                  .replace("{digest}", json.dumps(stats, indent=2)))
        return self.client.chat(self.prompts["system_summary"], prompt, task="summary").strip()
