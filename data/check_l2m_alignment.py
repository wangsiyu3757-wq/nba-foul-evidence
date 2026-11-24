#!/usr/bin/env python3
"""
check_l2m_alignment.py

Checks that cleaned L2M records can be matched to tracked clips via the key
(GameId, VideolLink) before building training samples, so formatting mismatches
(extension, case, whitespace) are caught early.

Usage:
    python data/check_l2m_alignment.py \
        --l2m_jsonl processed/l2m_clean.jsonl \
        --annotations_dir annotations/

Output: number of matched / unmatched records and the first 20 unmatched
examples.
"""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l2m_jsonl", required=True)
    ap.add_argument("--annotations_dir", required=True)
    args = ap.parse_args()

    annotations_dir = Path(args.annotations_dir)
    # 已有的raw annotation文件名(不含_filtered后缀、不含.json)，格式应该是
    # {GameId}_{VideoLink}_DF_BCAST_{bitrate}
    existing_stems = set(
        p.stem for p in annotations_dir.glob("*.json") if not p.stem.endswith("_filtered")
    )

    matched, unmatched = [], []
    with open(args.l2m_jsonl, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            game_id = str(row.get("GameId", ""))
            video_link = str(row.get("VideolLink", ""))
            if not game_id or not video_link:
                unmatched.append((row, "GameId或VideolLink为空"))
                continue

            # 文件名里GameId_VideoLink是前缀，后面还跟着_DF_BCAST_xxxkbps，
            # 所以用前缀匹配而不是精确匹配
            prefix = f"{game_id}_{video_link}_"
            hit = [stem for stem in existing_stems if stem.startswith(prefix)]
            if hit:
                matched.append((row, hit[0]))
            else:
                unmatched.append((row, "找不到对应annotation文件"))

    total = len(matched) + len(unmatched)
    print(f"共 {total} 条L2M记录，匹配上 {len(matched)} 条，匹配不上 {len(unmatched)} 条"
          f"（匹配率 {len(matched)/total*100:.1f}%）" if total else "L2M文件是空的")

    if unmatched:
        print("\n匹配不上的前20条样例:")
        for row, reason in unmatched[:20]:
            print(f"  GameId={row.get('GameId')}, VideolLink={row.get('VideolLink')!r}, 原因: {reason}")


if __name__ == "__main__":
    main()