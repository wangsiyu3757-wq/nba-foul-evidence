#!/usr/bin/env python3
"""
clean_l2m_labels.py

Converts official NBA Last Two Minute (L2M) report JSON files (one game per
file, with top-level "game" / "stats" / "l2m" keys) into a JSONL file with one
record per call, keyed by (GameId, VideolLink) for alignment with video clips
and tracking outputs.

Dropped fields
- top-level "stats": game-level aggregates that cannot be attached to a clip
- imgChart: court-diagram percentages, unusable without court registration
- posID / posStart / posEnd: internal possession ordering, not a reliable
  frame-level reference (game clock does not map linearly to broadcast frames)

Kept fields
- GameId, GameDate, Home_team_abbr, Away_team_abbr
- VideolLink (the official field name is misspelled; kept as is)
- CallType, CallRatingName, Comment, CP, DP, Difficulty, PeriodName, PCTime,
  Qualifier, posTeamId, teamIdInFavor, errorInFavor

Usage:
    python data/clean_l2m_labels.py \
        --input_dir raw/l2m_json \
        --output_jsonl processed/l2m_clean.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

KEEP_L2M_FIELDS = [
    "VideolLink",  # 官方字段名拼写就是这样，保留原名
    "CallType",
    "CallRatingName",
    "Comment",
    "CP",
    "DP",
    "Difficulty",
    "PeriodName",
    "PCTime",
    "Qualifier",
    "posTeamId",
    "teamIdInFavor",
    "errorInFavor",
]

DROP_L2M_FIELDS = ["imgChart", "posID", "posStart", "posEnd"]


def process_one_file(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "game" not in data or "l2m" not in data:
        print(f"跳过 {path.name}：顶层缺少 'game' 或 'l2m' key，实际 key 有 {list(data.keys())}", file=sys.stderr)
        return []

    game_info = data["game"][0] if isinstance(data["game"], list) else data["game"]
    game_ctx = {
        "GameId": game_info.get("GameId"),
        "GameDate": game_info.get("GameDate"),
        "Home_team_abbr": game_info.get("Home_team_abbr"),
        "Away_team_abbr": game_info.get("Away_team_abbr"),
    }

    rows = []
    for row in data["l2m"]:
        cleaned = dict(game_ctx)
        for k in KEEP_L2M_FIELDS:
            cleaned[k] = row.get(k)
        rows.append(cleaned)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="原始 l2m_json 所在目录")
    ap.add_argument("--output_jsonl", required=True, help="输出的清洗后 JSONL 路径")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob("*.json"))
    if not files:
        print(f"在 {input_dir} 下没找到 .json 文件", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for path in files:
            rows = process_one_file(path)
            for row in rows:
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                total += 1
            print(f"{path.name}: {len(rows)} 条")

    print(f"\n共写出 {total} 条判罚记录 -> {out_path}")
    print(f"已丢弃字段: {DROP_L2M_FIELDS}（以及顶层 stats 段，整场比赛汇总，未纳入逐回合标注）")


if __name__ == "__main__":
    main()