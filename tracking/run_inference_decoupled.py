"""
run_inference_decoupled.py

Per-frame detection, tracking, re-identification and pose for broadcast
basketball video, with detection decoupled from track confirmation.

Pipeline (per frame)
  1. Detection: YOLOv8x detects people and the ball (det_model.predict).
     Every confident detection is kept; there is no track-confirmation gate,
     which caused BoT-SORT to drop most players in crowded scenes.
  2. Tracking: SimpleIoUTracker links detections across frames by IoU
     (--match-iou). Tracks unmatched for --max-lost frames are handed to a
     DINOv3-based ReID gallery so identities can be recovered after occlusion.
     Trackers reset at scene cuts detected with PySceneDetect.
  3. Pose: YOLOv8x-pose runs on each detected person's crop (box padded by
     --pose-crop-padding), so keypoints belong to that detection by
     construction; no IoU matching between separate person and pose
     detections is needed. If a crop contains several people, the one whose
     box best overlaps the original detection is kept.
  4. Skeleton coverage: a frame's skeleton counts as usable based on the number
     of valid keypoints; a per-video <stem>_skeleton_coverage.json is written
     as a sanity check.

Usage:
    python tracking/run_inference_decoupled.py \
        --video videos/<clip>.mp4 \
        --output-dir annotations/

Options:
    --iou               NMS IoU for person/ball detection (default 0.3)
    --pose-iou          NMS IoU inside a single-person crop (default 0.3)
    --pose-crop-padding crop padding ratio around each box (default 0.15)
    --pose-crop-imgsz   inference size for crops (default 256)
    --match-iou         IoU threshold for frame-to-frame matching (default 0.3)
    --max-lost          frames before a lost track goes to ReID (default 5)
    --scene-threshold   PySceneDetect sensitivity (default 27)
    --det-weights / --pose-weights / --dinov3-repo / --dinov3-weights
                        model paths; --video-dir to process a whole folder
"""

import cv2
import json
import os
import sys
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
import argparse

try:
    from scenedetect import detect, ContentDetector
except ImportError:
    print("请先安装依赖: pip install scenedetect --break-system-packages", file=sys.stderr)
    raise

# ── CLI参数 ──────────────────────────────────────
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--video", default=None)
arg_parser.add_argument("--output-dir", default=None)
arg_parser.add_argument("--iou", type=float, default=0.3,
                         help="det_model.predict()的NMS iou阈值(人+球都用这个)")
arg_parser.add_argument("--pose-iou", type=float, default=0.3,
                         help="单人裁剪图内部pose_model的NMS阈值(v2: 不再是全画面姿态检测的NMS)")
arg_parser.add_argument("--pose-crop-padding", type=float, default=0.15,
                         help="裁剪时在det框基础上外扩的比例，默认0.15")
arg_parser.add_argument("--pose-crop-imgsz", type=int, default=256,
                         help="单人裁剪图喂给pose_model的推理分辨率，默认256")
arg_parser.add_argument("--match-iou", type=float, default=0.3, help="SimpleIoUTracker帧间匹配IoU阈值")
arg_parser.add_argument("--max-lost", type=int, default=5, help="track连续几帧没匹配上就转交ReID gallery")
arg_parser.add_argument("--scene-threshold", type=float, default=27.0, help="PySceneDetect敏感度")
arg_parser.add_argument("--min-shot-frames", type=int, default=5)
arg_parser.add_argument("--video-dir", default="./videos", help="不传--video时批量处理的视频目录")
arg_parser.add_argument("--det-weights", default="yolov8x.pt", help="YOLO检测模型权重")
arg_parser.add_argument("--pose-weights", default="yolov8x-pose.pt", help="YOLO姿态模型权重")
arg_parser.add_argument("--dinov3-repo", default="./dinov3", help="本地DINOv3仓库目录(torch.hub source=local)")
arg_parser.add_argument("--dinov3-weights",
                         default="./weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
                         help="DINOv3 ViT-L/16 预训练权重路径")
cli_args = arg_parser.parse_args()

# ── 配置 ──────────────────────────────────────────
VIDEO_DIR = cli_args.video_dir
OUTPUT_DIR = cli_args.output_dir or "./annotations"
FRAME_STRIDE = 1
CONF_THRESH = 0.25
BALL_CONF_THRESH = 0.15
PERSON_CLASS_ID = 0
BALL_CLASS_ID = 32

DET_IOU = cli_args.iou
POSE_IOU = cli_args.pose_iou
POSE_CROP_PADDING = cli_args.pose_crop_padding
POSE_CROP_IMGSZ = cli_args.pose_crop_imgsz
MATCH_IOU = cli_args.match_iou
MAX_LOST = cli_args.max_lost

DINOV3_REPO_DIR = cli_args.dinov3_repo
DINOV3_WEIGHTS_PATH = cli_args.dinov3_weights
DINOV3_MODEL_NAME = "dinov3_vitl16"
REID_SIM_THRESHOLD = 0.68
REID_GALLERY_TTL = 90
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Skeleton usability thresholds ---
KP_CONF_THRESH = 0.3
KP_MIN_VISIBLE = 5

# --- 裁剪图内多人消歧的IoU阈值 (v2新增) ---
CROP_MATCH_IOU_MIN = 0.3   # 裁剪图内检测框换算回整帧坐标后，和原始det框的IoU
                            # 至少要到这个值，才认为"这就是目标本人"，否则视为没匹配到

os.makedirs(OUTPUT_DIR, exist_ok=True)

det_model = YOLO(cli_args.det_weights)
pose_model = YOLO(cli_args.pose_weights)

COCO_KP_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


def compute_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def has_usable_skeleton(keypoints):
    """v2新增: 判断某一帧某个人的骨骼是否"可用于训练"。
    不用keypoints is None判断——只要det_model检测到这个人、裁剪图里pose_model
    找到了匹配的目标，keypoints就是17点数组，区别在于每个点自己的置信度。
    覆盖率统计和后面事件排除都靠这个函数，不要用None/not None判断。
    """
    if not keypoints:
        return False
    visible = sum(1 for kp in keypoints if kp["visibility"] >= KP_CONF_THRESH)
    return visible >= KP_MIN_VISIBLE


def estimate_pose_for_detection(frame, det_bbox_xyxy, frame_w, frame_h):
    """v2新增: 对det_model给出的单个检测框，裁剪出局部图像单独跑pose_model，
    取代"整张画面跑一次pose_model + IoU事后匹配"的做法。

    返回: 17点关键点list，或None(裁剪图里pose_model没找到可信匹配)
    """
    x1, y1, x2, y2 = det_bbox_xyxy
    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = bw * POSE_CROP_PADDING, bh * POSE_CROP_PADDING
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(frame_w, x2 + pad_x)
    cy2 = min(frame_h, y2 + pad_y)
    cx1i, cy1i, cx2i, cy2i = int(cx1), int(cy1), int(cx2), int(cy2)
    if cx2i <= cx1i or cy2i <= cy1i:
        return None

    crop = frame[cy1i:cy2i, cx1i:cx2i]
    pose_result = pose_model(
        crop, conf=CONF_THRESH, iou=POSE_IOU, imgsz=POSE_CROP_IMGSZ, verbose=False
    )[0]

    if pose_result.boxes is None or len(pose_result.boxes) == 0 or pose_result.keypoints is None:
        return None

    # padding有可能把邻居也框进裁剪图，裁剪内可能不止一个检测。
    # 用"换算回整帧坐标后和原始det框的IoU最高"来判断哪个是目标本人，
    # 不用置信度最高的那个——遮挡时目标本人置信度完全可能比旁边看得
    # 清楚的邻居低，用置信度选反而更容易张冠李戴。
    best_iou, best_idx = -1.0, None
    for idx, box in enumerate(pose_result.boxes):
        bx1, by1, bx2, by2 = map(float, box.xyxy[0])
        frame_box = [bx1 + cx1, by1 + cy1, bx2 + cx1, by2 + cy1]
        iou = compute_iou(frame_box, det_bbox_xyxy)
        if iou > best_iou:
            best_iou, best_idx = iou, idx

    if best_idx is None or best_iou < CROP_MATCH_IOU_MIN:
        return None

    kps = pose_result.keypoints.data[best_idx]
    keypoints = [
        {"name": COCO_KP_NAMES[j], "x": float(kp[0]) + cx1, "y": float(kp[1]) + cy1,
         "visibility": float(kp[2])}
        for j, kp in enumerate(kps)
    ]
    return keypoints


# ── 自己写的简单IoU追踪器，替代BoT-SORT的确认/关联逻辑 (未改动) ──────────
class SimpleIoUTracker:
    def __init__(self, match_iou=MATCH_IOU, max_lost=MAX_LOST):
        self.match_iou = match_iou
        self.max_lost = max_lost
        self.active = {}
        self.next_id = 1

    def reset(self):
        self.active = {}

    def update(self, detections):
        """detections: [{"bbox":[x1,y1,x2,y2], "conf": float, ...其他字段}, ...]
        返回: [{"raw_track_id": int, "bbox":[...], "conf": float, ...其他字段透传}, ...]

        v2改动: 原来这里显式只挑"bbox"/"conf"两个字段组装结果，把detections
        里其他字段(比如现在新加的"keypoints")全部丢掉了。现在改成**det透传，
        这样keypoints能跟着一路带到最后，不用再单独做一次对齐。
        """
        candidates = []
        for tid, info in self.active.items():
            for di, det in enumerate(detections):
                iou = compute_iou(info["bbox"], det["bbox"])
                if iou >= self.match_iou:
                    candidates.append((iou, tid, di))
        candidates.sort(key=lambda x: -x[0])

        matched_tid = set()
        matched_det = set()
        assignment = {}
        for iou, tid, di in candidates:
            if tid in matched_tid or di in matched_det:
                continue
            matched_tid.add(tid)
            matched_det.add(di)
            assignment[di] = tid

        results = []
        for di, det in enumerate(detections):
            if di in assignment:
                tid = assignment[di]
            else:
                tid = self.next_id
                self.next_id += 1
            self.active[tid] = {"bbox": det["bbox"], "lost_frames": 0}
            results.append({**det, "raw_track_id": tid})  # v2: **det透传所有字段(含keypoints)

        for tid in list(self.active.keys()):
            if tid not in matched_tid:
                self.active[tid]["lost_frames"] += 1
                if self.active[tid]["lost_frames"] > self.max_lost:
                    del self.active[tid]

        return results


# ── DINOv3 Re-ID module ─────────
class DinoV3ReID:
    def __init__(self, repo_dir=DINOV3_REPO_DIR, weights_path=DINOV3_WEIGHTS_PATH,
                 model_name=DINOV3_MODEL_NAME, device=DEVICE):
        self.device = device
        self.model = torch.hub.load(
            repo_dir, model_name, source="local", weights=weights_path
        ).to(device).eval()

        self.img_size = 224
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        self.gallery = {}
        self.active_feats = {}
        self.remap = {}

    def reset_state(self):
        self.gallery = {}
        self.active_feats = {}
        self.remap = {}

    @torch.no_grad()
    def extract_feat(self, frame_bgr, bbox_xyxy):
        x1, y1, x2, y2 = [int(max(0, v)) for v in bbox_xyxy]
        h, w = frame_bgr.shape[:2]
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame_bgr[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        resized = cv2.resize(crop_rgb, (self.img_size, self.img_size))
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        tensor = (tensor - self.mean) / self.std
        tensor = tensor.unsqueeze(0).to(self.device)

        feat = self.model(tensor)
        feat = F.normalize(feat, dim=-1)
        return feat.squeeze(0).cpu()

    def _match_against_gallery(self, feat, current_frame):
        best_id, best_sim = None, -1.0
        for lost_id, entry in list(self.gallery.items()):
            if current_frame - entry["lost_at_frame"] > REID_GALLERY_TTL:
                del self.gallery[lost_id]
                continue
            sim = torch.dot(feat, entry["feat"]).item()
            if sim > best_sim:
                best_sim, best_id = sim, lost_id
        if best_id is not None and best_sim >= REID_SIM_THRESHOLD:
            return best_id, best_sim
        return None, best_sim

    def update(self, frame_bgr, frame_idx, tracked_boxes):
        current_ids = set()
        results = []
        for item in tracked_boxes:
            raw_id = item["raw_track_id"]
            bbox = item["bbox"]
            feat = self.extract_feat(frame_bgr, bbox)
            if feat is None:
                final_id = raw_id
            elif raw_id in self.remap:
                final_id = self.remap[raw_id]
            else:
                matched_id, sim = self._match_against_gallery(feat, frame_idx)
                if matched_id is not None:
                    final_id = matched_id
                    self.remap[raw_id] = final_id
                    del self.gallery[matched_id]
                    print(f"  [ReID] frame={frame_idx}: raw_id={raw_id} "
                          f"重新关联到 track_id={final_id} (相似度={sim:.3f})")
                else:
                    final_id = raw_id
                    self.remap[raw_id] = final_id

            if final_id in current_ids:
                if raw_id != final_id:
                    final_id = raw_id
                    self.remap[raw_id] = final_id
                else:
                    impostor_raw_id = next(
                        (rid for rid, fid in self.remap.items()
                         if fid == final_id and rid != raw_id),
                        None,
                    )
                    if impostor_raw_id is not None:
                        self.remap[impostor_raw_id] = impostor_raw_id
                        current_ids.discard(final_id)
                        for r in results:
                            if r["track_id"] == final_id and r["raw_track_id"] == impostor_raw_id:
                                r["track_id"] = impostor_raw_id

            current_ids.add(final_id)
            if feat is not None:
                self.active_feats.setdefault(final_id, deque(maxlen=5)).append(feat)
            results.append({**item, "track_id": final_id})

        lost_ids = set(self.active_feats.keys()) - current_ids
        for lost_id in lost_ids:
            feats = self.active_feats.pop(lost_id)
            avg_feat = F.normalize(torch.stack(list(feats)).mean(dim=0), dim=0)
            self.gallery[lost_id] = {"feat": avg_feat, "lost_at_frame": frame_idx}

        return results


reid = DinoV3ReID()
simple_tracker = SimpleIoUTracker()


def _frame_num(timecode):
    return timecode.frame_num if hasattr(timecode, "frame_num") else timecode.get_frames()


def find_scene_cut_frames(video_path, threshold, min_shot_frames):
    scene_list = detect(str(video_path), ContentDetector(threshold=threshold))
    cut_frames = []
    for scene in scene_list:
        start_frame = _frame_num(scene[0])
        if start_frame > 0:
            cut_frames.append(start_frame)
    filtered = []
    for f in sorted(cut_frames):
        if not filtered or f - filtered[-1] >= min_shot_frames:
            filtered.append(f)
    return set(filtered)


def process_video(video_path):
    cut_frames = find_scene_cut_frames(video_path, cli_args.scene_threshold, cli_args.min_shot_frames)
    if cut_frames:
        print(f"  检测到 {len(cut_frames)} 个镜头切点(帧号): {sorted(cut_frames)}")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    results_all = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_STRIDE == 0:
            frame_data = {"frame": frame_idx, "width": W, "height": H, "persons": [], "balls": []}

            if frame_idx in cut_frames:
                print(f"  [镜头切点] frame={frame_idx} (约{frame_idx/fps:.2f}s): "
                      f"重置SimpleIoUTracker + 清空ReID gallery")
                simple_tracker.reset()
                reid.reset_state()

            # ── 人物检测: 仍然100%由已验证可靠的det_model负责，不变 ──
            det_results = det_model.predict(
                frame,
                conf=CONF_THRESH,
                iou=DET_IOU,
                imgsz=1280,
                augment=False,
                classes=[PERSON_CLASS_ID],
                verbose=False,
            )[0]

            # ── v2改动1: 对每个检测框单独裁剪、单独跑pose_model，
            #     不再对整张画面跑一次pose_model+IoU事后匹配 ──
            raw_detections = []
            if det_results.boxes is not None:
                for box in det_results.boxes:
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    keypoints = estimate_pose_for_detection(frame, [x1, y1, x2, y2], W, H)
                    raw_detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "conf": float(box.conf),
                        "keypoints": keypoints,
                    })

            tracked_boxes = simple_tracker.update(raw_detections)

            ball_results = det_model.predict(
                frame,
                conf=BALL_CONF_THRESH,
                classes=[BALL_CLASS_ID],
                verbose=False,
            )[0]

            ball_boxes = []
            if ball_results.boxes is not None:
                sorted_boxes = sorted(ball_results.boxes, key=lambda b: -float(b.conf))
                for i, box in enumerate(sorted_boxes):
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    ball_boxes.append({
                        "track_id": i,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "conf": float(box.conf),
                    })

            frame_data["balls"] = ball_boxes

            reid_boxes = reid.update(frame, frame_idx, tracked_boxes)

            # ── v2改动1: keypoints已经跟着item一路透传过来了，
            #     不再需要match_pose_to_tracks这个IoU事后匹配步骤 ──
            persons = []
            for item in reid_boxes:
                x1, y1, x2, y2 = item["bbox"]
                persons.append({
                    "track_id": item["track_id"],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "conf": item["conf"],
                    "keypoints": item["keypoints"],
                })
            frame_data["persons"] = persons

            results_all.append(frame_data)
            print(f"  Frame {frame_idx}/{total} done "
                  f"(raw_det={len(raw_detections)}, persons={len(persons)}, balls={len(ball_boxes)})")

        frame_idx += 1

    cap.release()

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    out_path = Path(OUTPUT_DIR) / (Path(video_path).stem + ".json")
    tmp_path = out_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump({"video": str(video_path), "fps": fps, "frames": results_all}, f, indent=2)
    tmp_path.replace(out_path)
    print(f"保存: {out_path}")

    # ── Per-video skeleton coverage sanity check ──
    coverage = {}
    for frame_data in results_all:
        for p in frame_data["persons"]:
            tid = p["track_id"]
            entry = coverage.setdefault(tid, {"frames_present": 0, "frames_with_skeleton": 0})
            entry["frames_present"] += 1
            if has_usable_skeleton(p["keypoints"]):
                entry["frames_with_skeleton"] += 1

    for entry in coverage.values():
        entry["coverage_ratio"] = (
            entry["frames_with_skeleton"] / entry["frames_present"]
            if entry["frames_present"] else 0.0
        )

    coverage_path = Path(OUTPUT_DIR) / (Path(video_path).stem + "_skeleton_coverage.json")
    with open(coverage_path, "w") as f:
        json.dump(coverage, f, indent=2)
    print(f"骨骼覆盖率统计(整段视频粒度): {coverage_path}")


if cli_args.video:
    vp = Path(cli_args.video)
    print(f"\n处理: {vp.name}  (iou={DET_IOU}, pose_iou[裁剪内]={POSE_IOU}, "
          f"match_iou={MATCH_IOU}, max_lost={MAX_LOST}, 输出到={OUTPUT_DIR})")
    process_video(vp)
else:
    for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv"]:
        for vp in sorted(Path(VIDEO_DIR).glob(ext)):
            print(f"\n处理: {vp.name}  (iou={DET_IOU}, pose_iou[裁剪内]={POSE_IOU}, "
                  f"match_iou={MATCH_IOU}, max_lost={MAX_LOST}, 输出到={OUTPUT_DIR})")
            process_video(vp)