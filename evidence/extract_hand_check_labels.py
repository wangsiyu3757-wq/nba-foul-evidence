#!/usr/bin/env python3
"""
extract_hand_check_labels.py

Weakly supervised extraction of candidate hand-check contact events from
on-court tracks (first foul category of the label scheme). Output is a
candidate pool for human review, not final labels.

Algorithm
  1. For every frame, pair up on-court track ids (from *_filtered.json).
  2. Compute the pixel distance from A's wrists to B's arm keypoints
     (wrists, elbows) and torso keypoints (shoulders, hips), and vice versa;
     take the minimum.
  3. Scale-adaptive threshold:
       threshold = CONTACT_RATIO * mean(longer bbox side of the two players)
     so camera zoom does not distort the threshold.
  4. Merge below-threshold frames into events, tolerating gaps of up to
     GAP_TOLERANCE frames (keypoint jitter).
  5. contact_body_part = arm or torso, by majority over the event's frames.
  6. ball_possession_at_contact: at the first frame, the player whose wrist is
     closest to the ball within BALL_POSSESSION_RATIO; left empty otherwise.

Usage (single clip):
    python evidence/extract_hand_check_labels.py \
        --json annotations/<clip>_filtered.json \
        --output qa_report/hand_check/<clip>_events.jsonl

Usage (batch):
    python evidence/extract_hand_check_labels.py \
        --annotations-dir annotations/ \
        --output-dir processed/hand_check_labels/ \
        --summary-csv processed/hand_check_summary.csv
"""

import argparse
import csv
import json
import math
from pathlib import Path

ARM_KEYPOINTS = ["left_wrist", "right_wrist", "left_elbow", "right_elbow"]
TORSO_KEYPOINTS = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
WRIST_KEYPOINTS = ["left_wrist", "right_wrist"]

CONTACT_RATIO = 0.14          # 接触判定距离阈值 = 这个比例 * 两人bbox较大边长均值
BALL_POSSESSION_RATIO = 0.35  # 持球判定距离阈值(比接触阈值宽松很多，球离手有一定距离也算持球)
MIN_KP_VISIBILITY = 0.3       # 关键点可见度低于这个值不参与计算(遮挡/边缘退化的点不可信)
GAP_TOLERANCE = 3             # 事件内允许中断几帧(抖动容错)
MIN_EVENT_FRAMES = 2          # 少于这么多帧的"接触"当噪声丢弃，不成事件


def bbox_size(bbox):
    """bbox: [x, y, w, h] -> 取较大边长，作为这个人在画面里的"尺度参考" """
    return max(bbox[2], bbox[3])


def kp_dict(person):
    """把persons[i]['keypoints']列表转成 name -> (x,y,visibility) 的dict，
    没有keypoints(None)或可见度太低的点不收录"""
    d = {}
    kps = person.get("keypoints")
    if not kps:
        return d
    for kp in kps:
        if kp.get("visibility", 0) >= MIN_KP_VISIBILITY:
            d[kp["name"]] = (kp["x"], kp["y"])
    return d


def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def min_contact_distance(kp_a, kp_b):
    """返回 (min_dist, body_part, contactor) —— contactor标记是A的手还是B的手触发的最小距离"""
    best = (float("inf"), None, None)

    def check(wrist_owner_kps, wrist_owner_label, target_kps, target_label_arm, target_label_torso):
        nonlocal best
        for wname in WRIST_KEYPOINTS:
            if wname not in wrist_owner_kps:
                continue
            wpt = wrist_owner_kps[wname]
            for tname in ARM_KEYPOINTS:
                if tname in target_kps:
                    d = dist(wpt, target_kps[tname])
                    if d < best[0]:
                        best = (d, target_label_arm, wrist_owner_label)
            for tname in TORSO_KEYPOINTS:
                if tname in target_kps:
                    d = dist(wpt, target_kps[tname])
                    if d < best[0]:
                        best = (d, target_label_torso, wrist_owner_label)

    check(kp_a, "A", kp_b, "arm-arm", "arm-torso")
    check(kp_b, "B", kp_a, "arm-arm", "arm-torso")
    return best  # (min_dist, body_part, contactor)


def ball_holder(kp_a, kp_b, balls, ball_ratio_threshold, ref_size):
    """balls: 这一帧的balls列表(bbox是x,y,w,h)。返回持球者是"A"/"B"/None"""
    if not balls:
        return None
    # 球通常只有一个，取置信度最高的
    ball = max(balls, key=lambda b: b.get("conf", 0))
    bx, by, bw, bh = ball["bbox"]
    ball_center = (bx + bw / 2, by + bh / 2)

    best_owner, best_d = None, float("inf")
    for owner, kps in (("A", kp_a), ("B", kp_b)):
        for wname in WRIST_KEYPOINTS:
            if wname in kps:
                d = dist(kps[wname], ball_center)
                if d < best_d:
                    best_d, best_owner = d, owner
    if best_owner is not None and best_d <= ball_ratio_threshold * ref_size:
        return best_owner
    return None


def extract_events(data, contact_ratio=CONTACT_RATIO, ball_ratio=BALL_POSSESSION_RATIO,
                    gap_tolerance=GAP_TOLERANCE, min_event_frames=MIN_EVENT_FRAMES):
    frames = data.get("frames", [])
    # pair_key(frozenset{tid_a, tid_b}) -> 正在进行中的事件状态
    open_events = {}
    closed_events = []

    for fr in frames:
        frame_idx = fr["frame"]
        persons = fr.get("persons", [])
        balls = fr.get("balls", [])

        # 这一帧里有效的(track_id, keypoints, bbox)
        valid = [(p["track_id"], kp_dict(p), p["bbox"]) for p in persons if kp_dict(p)]

        present_pairs_this_frame = set()

        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                tid_a, kp_a, bbox_a = valid[i]
                tid_b, kp_b, bbox_b = valid[j]
                pair_key = tuple(sorted([tid_a, tid_b]))

                ref_size = (bbox_size(bbox_a) + bbox_size(bbox_b)) / 2.0
                threshold = contact_ratio * ref_size

                min_d, body_part, contactor = min_contact_distance(kp_a, kp_b)
                if min_d > threshold:
                    continue  # 这一帧这对人没接触

                present_pairs_this_frame.add(pair_key)
                contactor_tid = tid_a if contactor == "A" else tid_b
                holder = ball_holder(kp_a, kp_b, balls, ball_ratio, ref_size)
                holder_tid = tid_a if holder == "A" else (tid_b if holder == "B" else None)

                if pair_key not in open_events:
                    open_events[pair_key] = {
                        "contact_track_ids": list(pair_key),
                        "contact_start_frame": frame_idx,
                        "contact_end_frame": frame_idx,
                        "last_seen_frame": frame_idx,
                        "body_part_votes": {},
                        "contactor_votes": {},
                        "ball_possession_votes": {},
                        "min_distance_px": min_d,
                    }
                ev = open_events[pair_key]
                ev["contact_end_frame"] = frame_idx
                ev["last_seen_frame"] = frame_idx
                ev["min_distance_px"] = min(ev["min_distance_px"], min_d)
                ev["body_part_votes"][body_part] = ev["body_part_votes"].get(body_part, 0) + 1
                ev["contactor_votes"][contactor_tid] = ev["contactor_votes"].get(contactor_tid, 0) + 1
                if holder_tid is not None:
                    ev["ball_possession_votes"][holder_tid] = ev["ball_possession_votes"].get(holder_tid, 0) + 1

        # 处理这一帧没再接触、但事件还开着的pair——超过gap_tolerance就收尾关闭
        for pair_key in list(open_events.keys()):
            if pair_key in present_pairs_this_frame:
                continue
            ev = open_events[pair_key]
            if frame_idx - ev["last_seen_frame"] > gap_tolerance:
                closed_events.append(ev)
                del open_events[pair_key]

    # clip跑完了，剩下还开着的事件也收尾
    closed_events.extend(open_events.values())

    # 整理成最终输出格式，丢弃太短的事件(噪声)
    results = []
    for ev in closed_events:
        n_frames = ev["contact_end_frame"] - ev["contact_start_frame"] + 1
        if n_frames < min_event_frames:
            continue
        body_part = max(ev["body_part_votes"].items(), key=lambda x: x[1])[0] if ev["body_part_votes"] else None
        contactor = max(ev["contactor_votes"].items(), key=lambda x: x[1])[0] if ev["contactor_votes"] else None
        ball_holder_tid = (max(ev["ball_possession_votes"].items(), key=lambda x: x[1])[0]
                            if ev["ball_possession_votes"] else None)
        results.append({
            "contact_track_ids": ev["contact_track_ids"],
            "contact_start_frame": ev["contact_start_frame"],
            "contact_end_frame": ev["contact_end_frame"],
            "n_frames": n_frames,
            "contact_body_part": body_part,
            "contactor_track_id": contactor,  # 哪个track的手引发的最近距离(候选"打手方")
            "ball_possession_at_contact": ball_holder_tid,  # 持球方track_id，找不到就是None
            "min_distance_px": round(ev["min_distance_px"], 1),
        })

    results.sort(key=lambda e: e["contact_start_frame"])
    return results


def process_one(json_path, output_path, **kwargs):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    events = extract_events(data, **kwargs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ev in events:
            ev_out = dict(ev)
            ev_out["clip"] = json_path.stem.replace("_filtered", "")
            f.write(json.dumps(ev_out, ensure_ascii=False) + "\n")
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="单个_filtered.json文件路径")
    ap.add_argument("--output", help="单文件模式下的输出jsonl路径")
    ap.add_argument("--annotations-dir", help="批量模式: 扫描这个目录下所有*_filtered.json")
    ap.add_argument("--output-dir", help="批量模式: 每个clip的事件分别输出到这个目录下")
    ap.add_argument("--summary-csv", help="批量模式: 额外汇总一份所有clip的事件到一个CSV，方便人工抽查")
    ap.add_argument("--contact-ratio", type=float, default=CONTACT_RATIO,
                     help=f"接触判定阈值比例，默认{CONTACT_RATIO}(相对两人bbox尺度)")
    ap.add_argument("--ball-ratio", type=float, default=BALL_POSSESSION_RATIO,
                     help=f"持球判定阈值比例，默认{BALL_POSSESSION_RATIO}")
    args = ap.parse_args()

    extract_kwargs = {"contact_ratio": args.contact_ratio, "ball_ratio": args.ball_ratio}

    if args.json:
        if not args.output:
            print("单文件模式需要指定 --output")
            return
        events = process_one(Path(args.json), Path(args.output), **extract_kwargs)
        print(f"{Path(args.json).name}: 检测到 {len(events)} 个候选接触事件 -> {args.output}")
        return

    if not args.annotations_dir or not args.output_dir:
        print("批量模式需要 --annotations-dir 和 --output-dir")
        return

    annotations_dir = Path(args.annotations_dir)
    output_dir = Path(args.output_dir)
    files = sorted(annotations_dir.glob("*_filtered.json"))
    print(f"共找到 {len(files)} 个_filtered.json，开始提取打手犯规候选事件")

    all_rows = []
    for i, json_path in enumerate(files, 1):
        stem = json_path.stem.replace("_filtered", "")
        out_path = output_dir / f"{stem}_hand_check.jsonl"
        events = process_one(json_path, out_path, **extract_kwargs)
        for ev in events:
            row = dict(ev)
            row["clip"] = stem
            all_rows.append(row)
        print(f"[{i}/{len(files)}] {stem}: {len(events)} 个候选事件")

    print(f"\n共提取 {len(all_rows)} 个候选接触事件，来自 {len(files)} 个clip")

    if args.summary_csv:
        summary_path = Path(args.summary_csv)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["clip", "contact_track_ids", "contact_start_frame", "contact_end_frame",
                      "n_frames", "contact_body_part", "contactor_track_id",
                      "ball_possession_at_contact", "min_distance_px"]
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_rows:
                row_out = dict(row)
                row_out["contact_track_ids"] = ",".join(map(str, row_out["contact_track_ids"]))
                writer.writerow(row_out)
        print(f"Summary CSV: {summary_path} (use contact_start_frame / contact_end_frame to locate events for review)")


if __name__ == "__main__":
    main()