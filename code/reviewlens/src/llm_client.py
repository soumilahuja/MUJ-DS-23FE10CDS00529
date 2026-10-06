"""Thin, provider-agnostic wrapper around LLM APIs (Anthropic / OpenAI / offline mock)."""
import json
import logging
import time
import threading

log = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when the LLM API fails after all retries."""


class LLMClient:
    def __init__(self, provider, model, temperature=0.2, max_tokens=800, max_retries=3):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self._client = None
        self.min_interval = 13 if provider == "gemini" else 0  # free tier: 5 requests/minute
        self._lock = threading.Lock()
        self._last_call = 0.0

        if provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        elif provider == "gemini":
            import os
            from openai import OpenAI
            self._client = OpenAI(
                api_key=os.environ["GEMINI_API_KEY"],
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                max_retries=0,
            )
        elif provider == "mock":
            pass
        else:
            raise ValueError(f"Unknown provider '{provider}'. Use anthropic, openai or mock.")
        
    def _wait_for_slot(self):
        """Space out API calls to respect free-tier rate limits."""
        with self._lock:
            gap = time.time() - self._last_call
            if gap < self.min_interval:
                time.sleep(self.min_interval - gap)
            self._last_call = time.time()
    def chat(self, system: str, user: str, task: str = "") -> str:
        """Send one prompt, return text. Retries with exponential backoff."""
        for attempt in range(1, self.max_retries + 1):
            try:
                self._wait_for_slot()
                return self._call(system, user, task)
            except Exception as exc:  # network, rate-limit, auth, etc.
                if attempt == self.max_retries:
                    raise LLMError(f"LLM call failed after {attempt} attempts: {exc}") from exc
                wait = 2 ** attempt
                log.warning("LLM call failed (%s). Retry %d in %ds", exc, attempt, wait)
                time.sleep(wait)

    def _call(self, system, user, task):
        if self.provider == "anthropic":
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return "".join(b.text for b in resp.content if b.type == "text")

        if self.provider in ("openai", "gemini"):
            resp = self._client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
            )
            return resp.choices[0].message.content

        return _mock_response(user, task)


def _mock_response(user: str, task: str) -> str:
    """Deterministic fake LLM used ONLY for offline tests (provider: mock)."""
    if task == "summary":
        return ("**Overall mood:** Mixed (mock).\n**Top strengths:** - mock\n"
                "**Top problems:** - mock\n**Recommended actions:** 1. mock")
    text = user.split("REVIEW:")[-1].lower()
    neg = ["hot", "burning", "rude", "crash", "scratch", "late", "damaged", "disappoint", "switching", "awful"]
    pos = ["love", "great", "superb", "recommend", "decent", "comfortable"]
    n, p = sum(w in text for w in neg), sum(w in text for w in pos)
    sentiment = "negative" if n > p else "positive" if p > n else "neutral"
    return json.dumps({
        "sentiment": sentiment,
        "rating_estimate": {"negative": 2, "neutral": 3, "positive": 5}[sentiment],
        "emotion": "mock",
        "aspects": [{"aspect": "overall", "sentiment": sentiment}],
        "key_issue": "None" if sentiment != "negative" else "Mock issue",
        "urgency": "high" if "burning" in text else "low",
        "suggested_reply": "Thank you for your feedback (mock).",
    })
