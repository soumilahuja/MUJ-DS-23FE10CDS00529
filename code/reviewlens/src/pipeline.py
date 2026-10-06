"""Shared setup used by both the CLI and the Streamlit app."""
import yaml
from dotenv import load_dotenv

from src.analyzer import ReviewAnalyzer, load_prompts
from src.llm_client import LLMClient


def load_config(path="config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_analyzer(cfg: dict) -> ReviewAnalyzer:
    load_dotenv()
    provider = cfg["provider"]
    client = LLMClient(
        provider=provider,
        model=cfg["models"].get(provider, "mock"),
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
        max_retries=cfg["max_retries"],
    )
    return ReviewAnalyzer(client, load_prompts(cfg["prompt_file"]),
                          parse_retries=cfg["parse_retries"], max_workers=cfg["max_workers"])
