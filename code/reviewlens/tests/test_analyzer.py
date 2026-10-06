import pytest

from src.analyzer import ReviewAnalyzer, load_prompts, parse_json, validate
from src.llm_client import LLMClient


def make_analyzer():
    return ReviewAnalyzer(LLMClient("mock", "mock"), load_prompts("prompts/prompts.yaml"))


def test_parse_json_handles_code_fences():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_json_handles_extra_text():
    assert parse_json('Sure! {"a": 2} hope that helps') == {"a": 2}


def test_validate_rejects_bad_sentiment():
    with pytest.raises(ValueError):
        validate({"sentiment": "angry"})


def test_validate_clamps_rating():
    assert validate({"sentiment": "positive", "rating_estimate": 9})["rating_estimate"] == 5


def test_end_to_end_mock():
    a = make_analyzer()
    results = a.analyze_batch(["Love it, great!", "Charger was burning hot, awful"])
    stats = a.aggregate(results)
    assert stats["count"] == 2
    assert stats["sentiment_counts"] == {"positive": 1, "negative": 1}
    assert stats["high_urgency"][0]["id"] == 2
