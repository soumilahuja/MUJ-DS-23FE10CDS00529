"""CLI entry point:  python main.py --input data/sample_reviews.csv"""
import argparse
import logging

import pandas as pd

from src.pipeline import build_analyzer, load_config
from src.report import save_outputs


def main():
    ap = argparse.ArgumentParser(description="ReviewLens - LLM-powered review analyzer")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--input", help="CSV file (overrides config)")
    ap.add_argument("--provider", help="anthropic | openai | mock (overrides config)")
    ap.add_argument("--limit", type=int, help="only analyze the first N reviews")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = load_config(args.config)
    if args.provider:
        cfg["provider"] = args.provider
    df = pd.read_csv(args.input or cfg["input_file"])
    reviews = df[cfg["text_column"]].dropna().astype(str).tolist()
    if args.limit:
        reviews = reviews[: args.limit]

    analyzer = build_analyzer(cfg)
    print(f"Analyzing {len(reviews)} reviews with {cfg['provider']}...")
    results = analyzer.analyze_batch(reviews)
    stats = analyzer.aggregate(results)
    summary = analyzer.executive_summary(stats) if stats["count"] else "No reviews analyzed."
    path = save_outputs(results, stats, summary, cfg["output_dir"])

    print("\n" + summary)
    print(f"\nSentiment: {stats['sentiment_counts']} | Avg rating: {stats['avg_rating']}")
    print(f"Saved results.json, results.csv and {path}")


if __name__ == "__main__":
    main()
