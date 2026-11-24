#!/usr/bin/env python3
"""
export_event_snapshots.py

Turns candidate contact events into annotated snapshots for fast human review,
instead of converting game clock to frame numbers and scrubbing video by hand.

For each clip:
  1. Read the events produced by extract_hand_check_labels.py, sort by
     min_distance_px and keep the top --top-k (default 8).
  2. Within [contact_start_frame, contact_end_frame], recompute the per-frame
     contact distance and pick the closest-contact frame as the representative.
  3. Extract that frame, draw both players' boxes and track ids, and print the
     time from the start of the video.
  4. Save as JPG; the file name encodes rank, frame, time and distance, e.g.
       <clip>_ev01_f330_t11.0s_d5.7px.jpg

Usage (single clip):
    python evidence/export_event_snapshots.py \
        --video videos/<clip>.mp4 \
        --annotations annotations/<clip>_filtered.json \
        --events qa_report/hand_check/<clip>_events.jsonl \
        --output-dir qa_report/hand_check_snapshots/<clip> \
        --top-k 8

Usage (batch):
    python evidence/export_event_snapshots.py \
        --video-dir videos/ \
        --annotations-dir annotations/ \
        --events-dir processed/hand_check_labels/ \
        --output-dir qa_report/hand_check_snapshots/ \
        --top-k 8

Batch mode matches {events_dir}/{stem}_hand_check.jsonl,
{annotations_dir}/{stem}_filtered.json and {video_dir}/{stem}.(mp4|avi|mov|mkv).
Clips with a non-empty output folder are skipped (resumable); clips missing
any input are skipped with a warning.
"""

import argparse
import json
import math
from pathlib import Path

import cv2

VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv"]

ARM_KEYPOINTS = ["left_wrist", "right_wrist", "left_elbow", "right_elbow"]
TORSO_KEYPOINTS = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
WRIST_KEYPOINTS = ["left_wrist", "right_wrist"]
MIN_KP_VISIBILITY = 0.3

COLOR_A = (60, 60, 255)   # 红(BGR) — contact_track_ids里的第一个人
COLOR_B = (255, 180, 40)  # 蓝(BGR) — 第二个人


def kp_dict(person):
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


def pair_min_distance(kp_a, kp_b):
    """跟extract_hand_check_labels.py里min_contact_distance逻辑一致，
    只是这里只需要返回最小距离数值本身，不需要body_part/contactor标签"""
    best = float("inf")
    for wrist_kps, target_kps in ((kp_a, kp_b), (kp_b, kp_a)):
        for wname in WRIST_KEYPOINTS:
            if wname not in wrist_kps:
                continue
            wpt = wrist_kps[wname]
            for tname in ARM_KEYPOINTS + TORSO_KEYPOINTS:
                if tname in target_kps:
                    d = dist(wpt, target_kps[tname])
                    if d < best:
                        best = d
    return best


def build_frame_index(annotations_data):
    """frame_idx -> {track_id: person_dict}，方便按帧+track_id快速查bbox/keypoints"""
    idx = {}
    for fr in annotations_data.get("frames", []):
        idx[fr["frame"]] = {p["track_id"]: p for p in fr.get("persons", [])}
    return idx


def find_representative_frame(frame_index, tid_a, tid_b, start_frame, end_frame):
    """在事件的[start,end]范围内(含首尾各扩2帧，兼容事件被GAP_TOLERANCE桥接、
    真正的最近接触可能刚好在边界外一点点的情况)，找两人都出现且距离最小的那一帧"""
    best_frame, best_d = None, float("inf")
    for f in range(max(0, start_frame - 2), end_frame + 3):
        persons = frame_index.get(f)
        if not persons or tid_a not in persons or tid_b not in persons:
            continue
        kp_a, kp_b = kp_dict(persons[tid_a]), kp_dict(persons[tid_b])
        if not kp_a or not kp_b:
            continue
        d = pair_min_distance(kp_a, kp_b)
        if d < best_d:
            best_d, best_frame = d, f
    if best_frame is None:
        # 兜底: 两人关键点在事件范围内始终没同时出现(极少见)，退回事件中间帧
        best_frame = (start_frame + end_frame) // 2
    return best_frame, best_d


def draw_box(img, person, color, label):
    x, y, w, h = person["bbox"]
    x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2, cv2.LINE_AA)


def process_one_clip(video_path, annotations_path, events_path, output_dir, top_k):
    """单个clip的完整处理逻辑，单文件模式和批量模式共用。返回导出的行数(int)。"""
    with open(annotations_path, encoding="utf-8") as f:
        ann_data = json.load(f)
    fps = ann_data.get("fps", 30.0)
    frame_index = build_frame_index(ann_data)

    events = [json.loads(line) for line in open(events_path, encoding="utf-8")]
    if not events:
        print(f"  {Path(video_path).stem}: 事件jsonl是空的，没有可导出的候选事件")
        return 0
    events.sort(key=lambda e: e["min_distance_px"])
    events = events[:top_k]

    clip_stem = Path(video_path).stem
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  {clip_stem}: 打不开视频 {video_path}，跳过")
        return 0

    print(f"{clip_stem}: 共{len(events)}个候选事件(已按接触距离排序，最可信的排最前)，开始截图...")
    n_exported = 0
    for rank, ev in enumerate(events, 1):
        tid_a, tid_b = ev["contact_track_ids"]
        rep_frame, rep_d = find_representative_frame(
            frame_index, tid_a, tid_b, ev["contact_start_frame"], ev["contact_end_frame"]
        )

        cap.set(cv2.CAP_PROP_POS_FRAMES, rep_frame)
        ret, frame_img = cap.read()
        if not ret:
            print(f"  [ev{rank:02d}] frame={rep_frame} 读取失败，跳过")
            continue

        persons = frame_index.get(rep_frame, {})
        if tid_a in persons:
            draw_box(frame_img, persons[tid_a], COLOR_A, f"id={tid_a}")
        if tid_b in persons:
            draw_box(frame_img, persons[tid_b], COLOR_B, f"id={tid_b}")

        t_sec = rep_frame / fps
        overlay = (f"frame={rep_frame}  t={t_sec:.1f}s  "
                   f"event={ev['contact_start_frame']}-{ev['contact_end_frame']}  "
                   f"part={ev['contact_body_part']}  dist={rep_d:.1f}px")
        cv2.rectangle(frame_img, (0, 0), (frame_img.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(frame_img, overlay, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)

        out_name = f"{clip_stem}_ev{rank:02d}_f{rep_frame}_t{t_sec:.1f}s_d{rep_d:.1f}px.jpg"
        out_path = out_dir / out_name
        cv2.imwrite(str(out_path), frame_img)
        n_exported += 1
        print(f"  [ev{rank:02d}] frame={rep_frame} (t={t_sec:.1f}s) dist={rep_d:.1f}px -> {out_name}")

    cap.release()
    return n_exported


def find_video(video_dir, stem):
    for ext in VIDEO_EXTS:
        p = Path(video_dir) / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def run_batch(video_dir, annotations_dir, events_dir, output_dir, top_k):
    events_dir = Path(events_dir)
    annotations_dir = Path(annotations_dir)
    output_dir = Path(output_dir)

    event_files = sorted(events_dir.glob("*_hand_check.jsonl"))
    print(f"共找到 {len(event_files)} 个事件文件，单个clip最多导出{top_k}张截图\n")

    n_done, n_skipped, n_missing = 0, 0, 0
    total_exported = 0
    for i, events_path in enumerate(event_files, 1):
        stem = events_path.name[: -len("_hand_check.jsonl")]
        clip_out_dir = output_dir / stem

        if clip_out_dir.exists() and any(clip_out_dir.iterdir()):
            n_skipped += 1
            continue

        video_path = find_video(video_dir, stem)
        ann_path = annotations_dir / f"{stem}_filtered.json"
        if video_path is None or not ann_path.exists():
            missing = []
            if video_path is None:
                missing.append("视频")
            if not ann_path.exists():
                missing.append("annotation(_filtered.json)")
            print(f"[{i}/{len(event_files)}] {stem}: 缺{('/'.join(missing))}，跳过")
            n_missing += 1
            continue

        print(f"[{i}/{len(event_files)}]", end=" ")
        n_exported = process_one_clip(video_path, ann_path, events_path, clip_out_dir, top_k)
        total_exported += n_exported
        n_done += 1

    print(f"\n批量导出完成: 处理{n_done}个clip(共{total_exported}张截图)，"
          f"跳过{n_skipped}个(已处理过)，缺文件跳过{n_missing}个")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="单文件模式: 视频路径")
    ap.add_argument("--annotations", help="单文件模式: 对应的_filtered.json")
    ap.add_argument("--events", help="单文件模式: extract_hand_check_labels.py产出的事件jsonl")
    ap.add_argument("--video-dir", help="批量模式: 视频目录")
    ap.add_argument("--annotations-dir", help="批量模式: annotation目录(找{stem}_filtered.json)")
    ap.add_argument("--events-dir", help="批量模式: extract_hand_check_labels.py批量模式的输出目录")
    ap.add_argument("--output-dir", required=True,
                     help="单文件模式下直接是截图输出目录；批量模式下是父目录，每个clip单独建子目录")
    ap.add_argument("--top-k", type=int, default=8, help="按min_distance_px排序，每个clip只导出前几个，默认8")
    args = ap.parse_args()

    if args.video:
        if not args.annotations or not args.events:
            print("单文件模式需要同时指定 --annotations 和 --events")
            return
        n = process_one_clip(args.video, args.annotations, args.events, args.output_dir, args.top_k)
        print(f"\n共导出 {n} 张截图 -> {args.output_dir}")
        print("在文件管理器里按文件名排序、开缩略图预览模式，一眼扫过去就行；"
              "哪张图看着确实是打手动作，对应哪个事件是真实候选，不用再逐条算时间了。")
        return

    if not args.video_dir or not args.annotations_dir or not args.events_dir:
        print("批量模式需要 --video-dir、--annotations-dir、--events-dir 三个都指定")
        return
    run_batch(args.video_dir, args.annotations_dir, args.events_dir, args.output_dir, args.top_k)


if __name__ == "__main__":
    main()