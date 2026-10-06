"""Writes JSON, CSV and Markdown outputs."""
import json
import os

import pandas as pd


def save_outputs(results, stats, summary, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "summary": summary, "results": results}, f, indent=2, ensure_ascii=False)

    rows = [{**r, "aspects": "; ".join(f"{a['aspect']} ({a['sentiment']})" for a in r.get("aspects", []))}
            for r in results]
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "results.csv"), index=False)

    md = ["# ReviewLens Report", "", "## Executive Summary", "", summary, "", "## Statistics", "",
          f"- Reviews analyzed: **{stats['count']}** (failed: {stats['failed']})",
          f"- Average estimated rating: **{stats['avg_rating']} / 5**",
          f"- Sentiment: {stats['sentiment_counts']}", "", "## Top Negative Aspects", ""]
    md += [f"- {a} ({n})" for a, n in stats["top_negative_aspects"]] or ["- none"]
    md += ["", "## High-Urgency Reviews", ""]
    md += [f"- Review #{u['id']}: {u['issue']}" for u in stats["high_urgency"]] or ["- none"]
    path = os.path.join(output_dir, "report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    return path
