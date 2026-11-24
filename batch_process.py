"""
batch_process.py

Resumable batch runner for the tracking stage. For each video it runs
  1. tracking/run_inference_decoupled.py   (detection + tracking + ReID + pose)
  2. tracking/filter_on_court_players.py   (keep on-court people only)

Design notes
- Decoupled tracking: BoT-SORT silently dropped confirmed detections in crowded
  scenes (one diagnostic case: 9 detected players/referees -> 2 tracked, a 78%
  loss caused by track-confirmation gating). The decoupled tracker writes every
  confident detection and links identities with a simple IoU matcher plus
  DINOv3 ReID; problem clips went from 1-2 to 40-55 tracks per frame.
- Raw outputs therefore include spectators; removing them is the job of the
  court filter in step 2.
- Each step is skipped independently if its output already exists.
- Court calibration is looked up as sideline_points/{GameId}.json, with an
  optional per-clip override sideline_points/{clip_stem}.json.
- Every subprocess has a timeout (--timeout, default 1800 s); a stuck clip is
  killed, logged as "timeout", and the batch moves on.
- Results are appended to a CSV log with a timestamp column.

Usage:
    python batch_process.py \
        --video-dir videos/ \
        --annotations-dir annotations/ \
        --sideline-points-dir sideline_points/ \
        --log batch_log.csv \
        --timeout 1800

Tracker options (--match-iou, --max-lost, --scene-threshold, ...) are passed
through to run_inference_decoupled.py; defaults match the validated setting.
"""

import argparse
import csv
import datetime
import subprocess
import sys
import time
from pathlib import Path

# 子脚本位置相对于本文件解析，与当前工作目录无关
RELEASE_ROOT = Path(__file__).resolve().parent
INFER_SCRIPT = RELEASE_ROOT / "tracking" / "run_inference_decoupled.py"
FILTER_SCRIPT = RELEASE_ROOT / "tracking" / "filter_on_court_players.py"

VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv"]


def run_with_timeout(cmd, timeout_sec):
    """跑subprocess，超时自动杀掉子进程(含子进程树)。
    返回 (returncode, stderr_tail, timed_out: bool)。"""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_sec,
        )
        return result.returncode, result.stderr[-500:], False
    except subprocess.TimeoutExpired as e:
        stderr_tail = (e.stderr or b"").decode("utf-8", errors="ignore")[-500:] if isinstance(e.stderr, bytes) else ""
        return None, stderr_tail, True


def parse_game_id(stem: str) -> str:
    return stem.split("_")[0]


def find_calibration(stem: str, game_id: str, sideline_points_dir: Path):
    """优先找 clip 级别精确标定，找不到再退回 GameId 级别标定。返回 Path 或 None。"""
    exact = sideline_points_dir / f"{stem}.json"
    if exact.exists():
        return exact
    by_game = sideline_points_dir / f"{game_id}.json"
    if by_game.exists():
        return by_game
    return None


def already_processed(video_path, annotations_dir):
    """检查这个视频是不是已经跑过完整pipeline了(推理json+过滤后json都存在)"""
    stem = video_path.stem
    infer_json = Path(annotations_dir) / f"{stem}.json"
    filtered_json = Path(annotations_dir) / f"{stem}_filtered.json"
    return infer_json.exists() and filtered_json.exists()


def process_one(video_path, annotations_dir, sideline_points_dir, timeout_sec, infer_extra_args):
    stem = video_path.stem
    game_id = parse_game_id(stem)
    infer_json = Path(annotations_dir) / f"{stem}.json"
    filtered_json = Path(annotations_dir) / f"{stem}_filtered.json"

    steps = []

    # 步骤1: 推理(run_inference_decoupled.py) —— infer_json已存在就跳过
    if infer_json.exists():
        steps.append(("run_inference_decoupled", "skipped(already exists)", 0.0))
    else:
        t0 = time.time()
        cmd = [
            sys.executable, str(INFER_SCRIPT),
            "--video", str(video_path),
            "--output-dir", str(annotations_dir),
        ] + infer_extra_args
        returncode, stderr_tail, timed_out = run_with_timeout(cmd, timeout_sec)
        elapsed = time.time() - t0
        if timed_out:
            steps.append(("run_inference_decoupled", "timeout", elapsed))
            return steps, f"run_inference_decoupled超时(>{timeout_sec}秒)，已自动杀掉子进程，跳到下一个"
        steps.append(("run_inference_decoupled", returncode, elapsed))
        if returncode != 0:
            return steps, f"run_inference_decoupled失败: {stderr_tail}"

    # 步骤2: 球场过滤 —— filtered_json已存在就跳过
    if filtered_json.exists():
        steps.append(("filter_on_court", "skipped(already exists)", 0.0))
        return steps, None

    calib_path = find_calibration(stem, game_id, sideline_points_dir)
    if calib_path is None:
        return steps, f"missing sideline calibration (GameId={game_id}); add sideline_points/{game_id}.json first"

    t0 = time.time()
    returncode, stderr_tail, timed_out = run_with_timeout(
        [sys.executable, str(FILTER_SCRIPT),
         "--json", str(infer_json),
         "--sideline-points", str(calib_path),
         "--output", str(filtered_json)],
        timeout_sec,
    )
    elapsed = time.time() - t0
    if timed_out:
        steps.append(("filter_on_court", "timeout", elapsed))
        return steps, f"filter_on_court超时(>{timeout_sec}秒)，已自动杀掉子进程，跳到下一个"
    steps.append(("filter_on_court", returncode, elapsed))
    if returncode != 0:
        return steps, f"filter_on_court失败: {stderr_tail}"

    return steps, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--annotations-dir", required=True)
    parser.add_argument("--sideline-points-dir", required=True,
                         help="按GameId命名的边线标定文件目录，见filter_on_court_players.py用法说明")
    parser.add_argument("--log", default="batch_log.csv")
    parser.add_argument("--timeout", type=int, default=1800,
                         help="单步(推理/过滤)超时秒数，超时自动杀掉子进程跳到下一个，默认1800秒(30分钟)")
    # 直接透传给run_inference_decoupled.py，不填就用它自己的默认值
    parser.add_argument("--match-iou", type=float, default=None,
                         help="透传给run_inference_decoupled.py的--match-iou，不填用其默认值(0.3)")
    parser.add_argument("--max-lost", type=int, default=None,
                         help="透传给run_inference_decoupled.py的--max-lost，不填用其默认值(5)")
    parser.add_argument("--scene-threshold", type=float, default=None,
                         help="透传给run_inference_decoupled.py的--scene-threshold，不填用其默认值(27)")
    args = parser.parse_args()

    infer_extra_args = []
    if args.match_iou is not None:
        infer_extra_args += ["--match-iou", str(args.match_iou)]
    if args.max_lost is not None:
        infer_extra_args += ["--max-lost", str(args.max_lost)]
    if args.scene_threshold is not None:
        infer_extra_args += ["--scene-threshold", str(args.scene_threshold)]

    video_dir = Path(args.video_dir)
    videos = sorted(
        p for ext in VIDEO_EXTS for p in video_dir.glob(f"*{ext}")
    )
    print(f"共找到 {len(videos)} 个视频，单步超时设置为 {args.timeout} 秒")
    if infer_extra_args:
        print(f"透传给run_inference_decoupled.py的额外参数: {infer_extra_args}")

    log_path = Path(args.log)
    log_exists = log_path.exists()
    log_file = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if not log_exists:
        log_writer.writerow(["timestamp", "video", "status", "error", "total_seconds"])

    n_skipped, n_ok, n_failed, n_missing_calib, n_timeout = 0, 0, 0, 0, 0

    for video_path in videos:
        if already_processed(video_path, args.annotations_dir):
            n_skipped += 1
            continue

        print(f"\n处理: {video_path.name}")
        t_start = time.time()
        steps, error = process_one(
            video_path, args.annotations_dir, Path(args.sideline_points_dir),
            args.timeout, infer_extra_args,
        )
        elapsed = time.time() - t_start
        ts = datetime.datetime.now().isoformat(timespec="seconds")

        if error:
            if "缺边线标定文件" in error:
                status = "missing_calibration"
            elif "超时" in error:
                status = "timeout"
            else:
                status = "failed"
            label = {"missing_calibration": "跳过(缺标定)", "timeout": "超时(已跳过)", "failed": "失败"}[status]
            print(f"  {label}: {error}")
            log_writer.writerow([ts, video_path.name, status, error, elapsed])
            if status == "missing_calibration":
                n_missing_calib += 1
            elif status == "timeout":
                n_timeout += 1
            else:
                n_failed += 1
        else:
            print(f"  完成，耗时 {elapsed:.1f}秒 ({', '.join(f'{s[0]}={s[1]}' for s in steps)})")
            log_writer.writerow([ts, video_path.name, "ok", "", elapsed])
            n_ok += 1

        log_file.flush()

    log_file.close()
    print(f"\n批量处理完成: 成功{n_ok}, 失败{n_failed}, 超时跳过{n_timeout}, "
          f"缺标定跳过{n_missing_calib}, 完全跳过(已处理过){n_skipped}")
    print(f"详细日志: {log_path}")


if __name__ == "__main__":
    main()