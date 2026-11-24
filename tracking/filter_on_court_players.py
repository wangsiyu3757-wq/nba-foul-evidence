"""
filter_on_court_players.py

Removes spectators by keeping only people whose feet are on the court, using a
lightweight sideline calibration instead of a full homography.

Geometric mode (default)
  The calibration marks two points on the far sideline and two on the near
  sideline; a person is on court if their foot point lies between the lines.
  Optional "end_lines" (0-2 baselines visible in the clip) bound the band
  left/right, removing photographers and front-row spectators near the baseline.

  Calibration format:
    {
      "far_line":  [[x1, y1], [x2, y2]],
      "near_line": [[x3, y3], [x4, y4]],
      "inside_reference": [x, y],
      "end_lines": [[[x5, y5], [x6, y6]], [[x7, y7], [x8, y8]]]   (optional)
    }

Colour-adaptive mode (--color-adaptive)
  Fixed pixel lines are only accurate for the calibration frame; zooms and pans
  move the court in the image. This mode samples the floor colour inside the
  calibrated region of --ref-frame (HSV hue and saturation only, for robustness
  to brightness), fits a Gaussian model, and classifies each person's foot patch
  by Mahalanobis distance. A vertical safety band around the far/near lines
  (--band-expand-px, default 120) rejects stands above and below the court;
  left/right is left to the colour model so referees at the ends are kept.

A per-frame cap (--max-persons-per-frame, default 15: ten players plus
referees, with margin) bounds the result.

Usage (filter):
    python tracking/filter_on_court_players.py \
        --json annotations/<clip>.json \
        --sideline-points sideline_points.json \
        --output annotations/<clip>_filtered.json \
        [--video videos/<clip>.mp4 --color-adaptive --ref-frame 100]

Usage (preview the calibrated on-court region):
    python tracking/filter_on_court_players.py \
        --sideline-points sideline_points.json \
        --video videos/<clip>.mp4 \
        --visualize qa_report/calib_preview/<clip>.png \
        --preview-frame 0 [--color-adaptive --ref-frame 100]

Tuning: --patch-half (foot patch radius, 7), --chi2-threshold (6.0),
--keep-fraction (0.35), --band-expand-px (120).
"""

import argparse
import json

import numpy as np


def line_side(p1, p2, query):
    """判断query点在直线p1->p2的哪一侧，符号代表左右，够用来判断内外"""
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    px, py = query
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def build_side_checker(sideline_points):
    far_p1, far_p2 = sideline_points["far_line"]
    near_p1, near_p2 = sideline_points["near_line"]
    ref = sideline_points["inside_reference"]
    end_lines = sideline_points.get("end_lines", [])  # 可选，0~2条底线，不填=老行为

    lines = [(tuple(far_p1), tuple(far_p2)), (tuple(near_p1), tuple(near_p2))]
    lines += [(tuple(pts[0]), tuple(pts[1])) for pts in end_lines]

    inside_signs = [1 if line_side(p1, p2, ref) > 0 else -1 for p1, p2 in lines]

    def is_inside(point):
        for (p1, p2), sign in zip(lines, inside_signs):
            side = line_side(p1, p2, point)
            ok = (side > 0) == (sign > 0)
            if not ok:
                return False
        return True

    return is_inside, lines


def get_ground_point(person):
    kps = person.get("keypoints")
    if kps:
        kp_by_name = {k["name"]: k for k in kps}
        ankles = []
        for name in ["left_ankle", "right_ankle"]:
            if name in kp_by_name and kp_by_name[name].get("visibility", 0) > 0.3:
                ankles.append((kp_by_name[name]["x"], kp_by_name[name]["y"]))
        if ankles:
            xs = [a[0] for a in ankles]
            ys = [a[1] for a in ankles]
            return sum(xs) / len(xs), sum(ys) / len(ys)

    x, y, w, h = person["bbox"]
    return x + w / 2, y + h


def build_side_checker_margin(sideline_points, margin_px):
    """跟build_side_checker类似，但要求点离每条边界线的垂直距离至少margin_px，
    不只是"在正确的一侧"。专门给颜色采样用——球场边界附近的像素(不管是因为
    视频压缩产生的色块伪影，还是镜头模糊/抗锯齿)颜色不纯，可能是球场色和场
    外色的混合过渡色，采进颜色模型里会污染样本(实测复现过：靠近边界的过渡
    像素被k-means单独聚成了一类，这一类的颜色恰好跟观众席很接近，会导致把
    观众席误判成球场)。往内缩一个安全边距再采样，从纯色区域取样，模型更干净。"""
    far_p1, far_p2 = sideline_points["far_line"]
    near_p1, near_p2 = sideline_points["near_line"]
    ref = sideline_points["inside_reference"]
    end_lines = sideline_points.get("end_lines", [])

    lines = [(tuple(far_p1), tuple(far_p2)), (tuple(near_p1), tuple(near_p2))]
    lines += [(tuple(pts[0]), tuple(pts[1])) for pts in end_lines]

    def seg_len(p1, p2):
        return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5

    inside_signs = [1 if line_side(p1, p2, ref) > 0 else -1 for p1, p2 in lines]
    lens = [seg_len(p1, p2) for p1, p2 in lines]

    def is_inside_margin(point):
        for (p1, p2), sign, seg in zip(lines, inside_signs, lens):
            if seg == 0:
                continue
            dist = line_side(p1, p2, point) / seg
            signed_dist = dist if sign > 0 else -dist
            if signed_dist < margin_px:
                return False
        return True

    return is_inside_margin


def build_far_near_band_checker(sideline_points, expand_px):
    """只用far_line/near_line这两条线(不看end_lines，故意的)，判断一个点是否
    落在这两条线夹出来的"上下"带状区域内，且额外往外放宽expand_px像素再判断
    (不是像build_side_checker_margin那样往内缩，是反过来往外扩，容忍度更高)。

    这是为了配合"只堵上下(观众看台)、不堵左右(裁判/镜头晃动)"这个设计意图：
    far_line/near_line大致是球场"宽度方向"的两条边线，看台在画面里通常在这
    条带状区域的上方/下方，跟这条带状区域有明显的纵向间隔，所以即便镜头
    推拉导致这两条线的真实像素位置在clip其他帧里有一定偏移，只要放宽的
    幅度(expand_px)盖过这个偏移量，看台大概率仍然落在放宽后的区域外面，
    球员/裁判(在球场内或紧贴边线)大概率仍然落在放宽后的区域里面。

    左右方向完全不设边界(不用end_lines)，这样镜头左右平移/裁判站在场地
    左右两端都不会被这道边界误伤，颜色模型自己去区分左右方向的球场/非球场。
    """
    far_p1, far_p2 = sideline_points["far_line"]
    near_p1, near_p2 = sideline_points["near_line"]
    ref = sideline_points["inside_reference"]
    lines = [(tuple(far_p1), tuple(far_p2)), (tuple(near_p1), tuple(near_p2))]

    def seg_len(p1, p2):
        return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5

    inside_signs = [1 if line_side(p1, p2, ref) > 0 else -1 for p1, p2 in lines]
    lens = [seg_len(p1, p2) for p1, p2 in lines]

    def is_inside_band(point):
        for (p1, p2), sign, seg in zip(lines, inside_signs, lens):
            if seg == 0:
                continue
            dist = line_side(p1, p2, point) / seg
            signed_dist = dist if sign > 0 else -dist
            if signed_dist < -expand_px:
                return False
        return True

    return is_inside_band


def band_signed_distance(sideline_points, point):
    """返回一个点离far_line/near_line(仅这两条)的"最紧"有符号距离(取两条线里
    更小的那个)——正值表示在标定区域深处(离两条线都比较远)，负值表示已经在
    区域外面(在放宽的band_expand_px容忍范围内)。用来给per-frame人数上限做
    排序: 同样通过了颜色判断的人，离球场核心区域越深，越可能是真的球员/裁判；
    刚好卡在放宽边缘的，更可能是"半个身子探进画面"的观众。"""
    far_p1, far_p2 = sideline_points["far_line"]
    near_p1, near_p2 = sideline_points["near_line"]
    ref = sideline_points["inside_reference"]

    best = float("inf")
    for p1, p2 in [(tuple(far_p1), tuple(far_p2)), (tuple(near_p1), tuple(near_p2))]:
        seg = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
        if seg == 0:
            continue
        ref_side = line_side(p1, p2, ref)
        sign = 1 if ref_side > 0 else -1
        dist = line_side(p1, p2, point) / seg
        signed_dist = dist if sign > 0 else -dist
        best = min(best, signed_dist)
    return best


def band_mask_for_frame(H, W, sideline_points, expand_px):
    """跟build_far_near_band_checker逻辑一样，但向量化输出H×W的bool数组，
    给--visualize用，不用逐像素调is_inside_band循环"""
    ys, xs = np.mgrid[0:H, 0:W]
    far_p1, far_p2 = sideline_points["far_line"]
    near_p1, near_p2 = sideline_points["near_line"]
    ref = sideline_points["inside_reference"]

    mask = np.ones((H, W), dtype=bool)
    for (x1, y1), (x2, y2) in [(far_p1, far_p2), (near_p1, near_p2)]:
        seg = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if seg == 0:
            continue
        side = (x2 - x1) * (ys - y1) - (y2 - y1) * (xs - x1)
        ref_side = (x2 - x1) * (ref[1] - y1) - (y2 - y1) * (ref[0] - x1)
        sign = 1 if ref_side > 0 else -1
        signed_dist = (side / seg) if sign > 0 else -(side / seg)
        mask &= signed_dist >= -expand_px
    return mask


def build_color_model(video_path, sideline_points, ref_frame_idx, n_samples=4000, seed=0,
                       n_clusters=4, margin_px=15):
    """从ref_frame_idx这一帧的标定区域内随机采样像素颜色(HSV，只取H/S两个
    通道)，构建球场颜色模型。

    真实球场不是单一颜色——木地板、场内漆色(经常跟队伍配色相关，比如深绿/
    深蓝)、白色边线，色相跨度可能很大。一开始用单个"均值+协方差"的高斯模型
    去拟合这些差异很大的颜色，为了同时覆盖木地板和漆色，协方差椭圆被迫拉得
    很大，大到连观众席的颜色都被误判成"够接近"——实测直接把整个画面(包括
    观众席)全部判成了球场内，是这个单高斯模型方法本身的缺陷，不是阈值没调好。

    改成: 先对采样到的颜色做k-means聚类(默认4类)，每一类分别拟合一个更"紧"
    的均值+协方差(每一类内部颜色更接近，协方差自然小很多)；判断一个像素是不是
    "球场色"时，看它离最近的那一类够不够近(取所有类里最小的马氏距离)，而不是
    用一个大杂烩椭圆。这样多种真实球场颜色都能各自被精确覆盖，同时不会像单
    高斯那样因为"要兼顾几种差异很大的颜色"而被迫放宽到连无关颜色都能进来。

    采样时额外用margin_px往标定区域内缩一圈再取样(见build_side_checker_margin
    的说明)，避免边界过渡色污染颜色模型。

    返回: (clusters, 实际采样数)，clusters是[(mean_i, cov_inv_i), ...]的列表"""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, ref_frame_idx)
    ret, img = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"读不到第{ref_frame_idx}帧，没法采样球场颜色")

    is_inside_margin = build_side_checker_margin(sideline_points, margin_px)
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float64)

    rng = np.random.default_rng(seed)
    hs_samples = []
    # 多采几批候选坐标，防止标定区域较小时一次随机采样命中率太低
    for _ in range(20):
        xs = rng.integers(0, W, size=n_samples * 2)
        ys = rng.integers(0, H, size=n_samples * 2)
        for x, y in zip(xs, ys):
            if is_inside_margin((int(x), int(y))):
                hs_samples.append(hsv[int(y), int(x), :2])
            if len(hs_samples) >= n_samples:
                break
        if len(hs_samples) >= n_samples:
            break

    if len(hs_samples) < 50:
        raise RuntimeError(
            f"从标定区域里只采样到{len(hs_samples)}个像素(太少)，"
            f"标定范围可能太小，或者第{ref_frame_idx}帧跟标定用的坐标对不上"
        )

    samples = np.array(hs_samples, dtype=np.float32)
    k = min(n_clusters, len(samples) // 20) if len(samples) >= 20 else 1
    k = max(k, 1)

    if k == 1:
        labels = np.zeros(len(samples), dtype=np.int32)
    else:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
        _, labels, _ = cv2.kmeans(samples, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
        labels = labels.flatten()

    clusters = []
    for c in range(k):
        pts = samples[labels == c].astype(np.float64)
        if len(pts) < 10:
            continue  # 这一类样本太少，拟合出来的协方差不可靠，跳过(其他类通常已经覆盖了主要颜色)
        mean = pts.mean(axis=0)
        cov = np.cov(pts.T) if len(pts) > 1 else np.eye(2)
        # 正则化下限设成方差16(标准差4)，不能设太小: H/S的取值范围是0-179/0-255，
        # 真实画面里同一块地板本身也会有光照/阴影/压缩噪声带来的自然色差，方差
        # 小到1e-3这种量级(之前的版本)会导致模型严格到连同一块地板里差几个色值
        # 的像素都被拒之门外——实测复现过这个问题：采样点因为太"干净"(方差趋近0)，
        # 模型变得比理论上应该有的严格得多，反而把真正的球场像素也判成"不是球场"。
        cov = np.atleast_2d(cov) + np.eye(2) * 16.0
        cov_inv = np.linalg.inv(cov)
        clusters.append((mean, cov_inv))

    if not clusters:
        raise RuntimeError("聚类后没有任何一类样本数够用，标定区域内颜色采样可能有问题")

    return clusters, len(hs_samples)


def color_mask_for_frame(frame_bgr, clusters, chi2_threshold):
    """整张图逐像素算到每一类颜色的马氏距离、取最小值，向量化实现
    (不是逐点循环，一帧图片、几个聚类中心的计算量完全够快)"""
    import cv2

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float64)
    hs = hsv[:, :, :2]
    min_d2 = None
    for mean, cov_inv in clusters:
        diff = hs - mean
        d2 = np.einsum("...i,ij,...j->...", diff, cov_inv, diff)
        min_d2 = d2 if min_d2 is None else np.minimum(min_d2, d2)
    return min_d2 < chi2_threshold


def patch_fraction_inside(mask, gx, gy, half):
    """某个点周围一个(2*half+1)见方的小窗口里，颜色判定为'球场'的像素占比"""
    H, W = mask.shape
    x1, x2 = max(0, int(gx - half)), min(W, int(gx + half + 1))
    y1, y2 = max(0, int(gy - half)), min(H, int(gy + half + 1))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(mask[y1:y2, x1:x2].mean())


def run_filter_color_adaptive(json_path, sideline_points_path, video_path, output_path,
                               ref_frame_idx=0, patch_half=7, chi2_threshold=6.0,
                               keep_fraction_threshold=0.35, n_clusters=4, margin_px=15,
                               band_expand_px=120, max_persons_per_frame=15):
    import cv2
    from pathlib import Path as _Path

    with open(sideline_points_path) as f:
        sideline_points = json.load(f)

    clusters, n_sampled = build_color_model(video_path, sideline_points, ref_frame_idx,
                                             n_clusters=n_clusters, margin_px=margin_px)
    print(f"从第{ref_frame_idx}帧的标定区域采样了{n_sampled}个像素，聚成{len(clusters)}类颜色，构建球场颜色模型")

    # 上下(far_line/near_line)方向额外加一道放宽过的几何约束，专门用来堵看台——
    # 不用end_lines(不封左右)，保留镜头左右平移/裁判站在场地两端的容忍度，见
    # build_far_near_band_checker的说明
    is_inside_band = build_far_near_band_checker(sideline_points, band_expand_px)

    with open(json_path) as f:
        data = json.load(f)
    frames_by_idx = {fr["frame"]: fr for fr in data["frames"]}
    max_frame = max(frames_by_idx.keys()) if frames_by_idx else -1

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"打不开视频: {video_path}")
        return

    # ── 两遍处理，而不是原来的单帧即时截断 ──────────────────────
    # 背景: 实测发现原来"每帧单独按frac+band_dist排序、超过上限直接砍掉后面的"
    # 这个做法，砍的不是真正的观众，而是"这一帧颜色匹配分刚好从1.0掉到0.9几"的
    # 真实球员——最典型的场景就是两人贴身对抗(打手/挡拆)时，脚部采样窗口沾到了
    # 对方球衣颜色，分数被拉低，而同一帧里往往还有十几个"分数刚好是满分1.0"的
    # 候选(可能是观众，也可能是别的球员)，真正在场上激烈对抗的人反而排到后面
    # 被挤掉——这恰好是最需要保留数据的时刻。
    #
    # 改法: 第一遍只算band+颜色是否通过(不做人数截断)，同时统计每个track_id在
    # 整段clip里通过band+颜色这道检验的总帧数(track_history)——一个从头到尾
    # 稳定出现几百帧的track，几乎不可能是"半个身子探进画面"的观众，这是比"单帧
    # 颜色匹配分"可靠得多的信号。第二遍再对真正超过上限的帧做截断，排序时把
    # track_history放在frac/band_dist前面作为第一关键字——这样一个历史悠久的
    # 真实球员，哪怕这一帧分数被对抗拖累到0.9几，也不会被"只在这一帧刷了个
    # 1.0分"的一次性候选(大概率是观众)挤掉；反过来，真正只出现一两帧的可疑
    # 候选，哪怕这一帧分数很高，也会被历史悠久的track压下去。
    n_total, n_dropped_by_band = 0, 0
    per_frame_candidates = {}  # frame_idx -> [(person, frac, band_dist), ...]
    track_history = {}  # track_id -> 整段clip里通过band+颜色检验的帧数

    frame_idx = 0
    while frame_idx <= max_frame:
        ret, img = cap.read()
        if not ret:
            break
        frame_data = frames_by_idx.get(frame_idx)
        if frame_data is not None:
            mask = color_mask_for_frame(img, clusters, chi2_threshold)
            candidates = []
            for p in frame_data.get("persons", []):
                n_total += 1
                tid = p["track_id"]
                gx, gy = get_ground_point(p)
                if not is_inside_band((gx, gy)):
                    n_dropped_by_band += 1
                    continue
                frac = patch_fraction_inside(mask, gx, gy, patch_half)
                if frac < keep_fraction_threshold:
                    continue
                band_dist = band_signed_distance(sideline_points, (gx, gy))
                candidates.append((p, frac, band_dist))
                track_history[tid] = track_history.get(tid, 0) + 1
            per_frame_candidates[frame_idx] = candidates
        frame_idx += 1
    cap.release()

    n_kept, n_dropped_by_cap = 0, 0
    track_keep_ratio = {}
    for frame_idx, candidates in per_frame_candidates.items():
        frame_data = frames_by_idx[frame_idx]
        for p, _, _ in candidates:
            tid = p["track_id"]
            in_cnt, tot_cnt = track_keep_ratio.get(tid, (0, 0))
            track_keep_ratio[tid] = (in_cnt, tot_cnt + 1)  # 先记一次"出现过"，下面kept的再+1

        if max_persons_per_frame and len(candidates) > max_persons_per_frame:
            candidates = sorted(
                candidates,
                key=lambda c: (track_history.get(c[0]["track_id"], 0), c[1], c[2]),
                reverse=True,
            )
            n_dropped_by_cap += len(candidates) - max_persons_per_frame
            candidates = candidates[:max_persons_per_frame]

        kept = []
        for p, frac, band_dist in candidates:
            tid = p["track_id"]
            in_cnt, tot_cnt = track_keep_ratio[tid]
            track_keep_ratio[tid] = (in_cnt + 1, tot_cnt)
            n_kept += 1
            kept.append(p)
        frame_data["persons"] = kept

    # 注意: track_keep_ratio这里的分母是"band+颜色都通过的候选帧数"，不含被band/
    # 颜色直接刷掉的帧——衡量的是"这个track挤进候选名单之后，有多少次真的被
    # 上限保留下来"，比例低说明这个track虽然偶尔能过band+颜色，但历史出现次数
    # 太少、每次都在人数截断里输给更有历史的track，大概率是边缘漏网的观众

    _Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[颜色自适应模式] 共{n_total}条人物记录，保留{n_kept}条 ({(n_kept/n_total*100) if n_total else 0:.1f}%)")
    print(f"  其中{n_dropped_by_band}条是被上下带状区域(far_line/near_line，已放宽{band_expand_px}px)"
          f"直接排除的，没有进入颜色判断——这部分基本就是看台观众")
    if max_persons_per_frame:
        print(f"  其中{n_dropped_by_cap}条是band+颜色都通过、但单帧人数超过{max_persons_per_frame}人上限、"
              f"按'整段clip历史出现帧数(第一关键字)+颜色匹配度+离band边界深度'排序后被挤掉的——"
              f"历史出现帧数多的track(大概率是真实球员/裁判)会被优先保留，只在这一帧刷了高分但"
              f"历史很短的候选(大概率是边缘漏网的观众)会被挤掉")
    print(f"已保存: {output_path}")

    low_ratio = [
        (tid, c[0] / c[1]) for tid, c in track_keep_ratio.items()
        if c[1] >= 5 and c[0] / c[1] < 0.2
    ]
    if low_ratio:
        print(f"\n[提示] 以下track整体保留比例很低(<20%)，大概率是场外人:")
        for tid, ratio in sorted(low_ratio, key=lambda x: x[1]):
            print(f"  track_id={tid}: 保留比例={ratio*100:.0f}%")


def run_filter(json_path, sideline_points_path, output_path):
    with open(sideline_points_path) as f:
        sideline_points = json.load(f)
    is_inside, lines = build_side_checker(sideline_points)
    n_lines = len(lines)

    with open(json_path) as f:
        data = json.load(f)

    n_total, n_kept = 0, 0
    track_keep_ratio = {}

    for frame_data in data["frames"]:
        kept = []
        for p in frame_data.get("persons", []):
            n_total += 1
            gx, gy = get_ground_point(p)
            inside = is_inside((gx, gy))

            tid = p["track_id"]
            in_cnt, tot_cnt = track_keep_ratio.get(tid, (0, 0))
            track_keep_ratio[tid] = (in_cnt + (1 if inside else 0), tot_cnt + 1)

            if inside:
                n_kept += 1
                kept.append(p)
        frame_data["persons"] = kept

    from pathlib import Path as _Path
    _Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"用了{n_lines}条边界线(2条边线" + (f"+{n_lines-2}条底线" if n_lines > 2 else "，没有底线约束，跟老版本行为一致") + ")")
    print(f"共{n_total}条人物记录，保留{n_kept}条 ({n_kept/n_total*100:.1f}%)")
    print(f"已保存: {output_path}")

    low_ratio = [
        (tid, c[0] / c[1]) for tid, c in track_keep_ratio.items()
        if c[1] >= 5 and c[0] / c[1] < 0.2
    ]
    if low_ratio:
        print(f"\n[提示] 以下track整体保留比例很低(<20%)，大概率是场外人:")
        for tid, ratio in sorted(low_ratio, key=lambda x: x[1]):
            print(f"  track_id={tid}: 保留比例={ratio*100:.0f}%")


def run_visualize(sideline_points_path, video_path, output_path, preview_frame,
                   color_adaptive=False, ref_frame_idx=0, patch_half=7, chi2_threshold=6.0,
                   n_clusters=4, margin_px=15, band_expand_px=120):
    import cv2
    from pathlib import Path as _Path

    with open(sideline_points_path) as f:
        sideline_points = json.load(f)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"打不开视频: {video_path}")
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, preview_frame)
    ret, img = cap.read()
    cap.release()
    if not ret:
        print(f"读不到第{preview_frame}帧")
        return

    H, W = img.shape[:2]
    ref = sideline_points["inside_reference"]

    if color_adaptive:
        clusters, n_sampled = build_color_model(video_path, sideline_points, ref_frame_idx,
                                                 n_clusters=n_clusters, margin_px=margin_px)
        print(f"从第{ref_frame_idx}帧的标定区域采样了{n_sampled}个像素，聚成{len(clusters)}类颜色，"
              f"预览的是第{preview_frame}帧上这个模型判定的'场内'区域")
        color_only_mask = color_mask_for_frame(img, clusters, chi2_threshold)
        band_mask = band_mask_for_frame(H, W, sideline_points, band_expand_px)
        mask = color_only_mask & band_mask
        # 只画far_line/near_line(放宽band_expand_px像素的那道上下边界参考线)，
        # 不画end_lines——颜色自适应模式左右方向完全不设边界，画出来也没意义
        far_p1, far_p2 = sideline_points["far_line"]
        near_p1, near_p2 = sideline_points["near_line"]
        all_lines = [(far_p1, far_p2), (near_p1, near_p2)]
        end_lines = []
    else:
        # 半透明高亮"判定为场内"的区域: 逐像素判断成本不高(一张图而已)，
        # 用numpy做，不用逐点循环
        ys, xs = np.mgrid[0:H, 0:W]
        far_p1, far_p2 = sideline_points["far_line"]
        near_p1, near_p2 = sideline_points["near_line"]
        end_lines = sideline_points.get("end_lines", [])
        all_lines = [(far_p1, far_p2), (near_p1, near_p2)] + [(pts[0], pts[1]) for pts in end_lines]

        ok = np.ones((H, W), dtype=bool)
        for (x1, y1), (x2, y2) in all_lines:
            side = (x2 - x1) * (ys - y1) - (y2 - y1) * (xs - x1)
            ref_side = (x2 - x1) * (ref[1] - y1) - (y2 - y1) * (ref[0] - x1)
            ok &= (side > 0) == (ref_side > 0)
        mask = ok

    overlay = img.copy()
    overlay[mask] = (overlay[mask] * 0.5 + np.array([0, 200, 0]) * 0.5).astype(np.uint8)
    img = overlay

    colors = [(0, 0, 255), (255, 0, 0)] + [(0, 200, 255), (255, 0, 255)][: max(0, len(all_lines) - 2)]
    labels = ["far_line", "near_line"] + [f"end_line{i+1}" for i in range(len(end_lines))]
    for (p1, p2), color, label in zip(all_lines, colors, labels):
        p1i = (int(p1[0]), int(p1[1]))
        p2i = (int(p2[0]), int(p2[1]))
        cv2.line(img, p1i, p2i, color, 3)
        cv2.putText(img, label, p1i, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    cv2.circle(img, (int(ref[0]), int(ref[1])), 8, (0, 255, 255), -1)
    cv2.putText(img, "inside_reference", (int(ref[0]) + 12, int(ref[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    # cv2.imwrite在输出目录不存在时会静默失败(返回False，不抛异常)——之前这里
    # 没检查返回值，之前那次运行大概率就是卡在这儿: qa_report/calib_preview/
    # 这个目录当时不存在，图根本没写出来，但脚本还是打印了"已保存"，用户自然
    # 找不到文件。这次先保证目录存在，再检查真的写成功了才打印"已保存"。
    _Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ok_write = cv2.imwrite(str(output_path), img)
    if not ok_write:
        print(f"[错误] 图片没有写成功: {output_path}（检查一下路径是否合法、磁盘空间是否够）")
        return
    print(f"标定预览图已保存: {output_path}")
    if color_adaptive:
        print(f"绿色半透明区域 = 颜色模型判定的'球场内' 且 落在far_line/near_line放宽"
              f"{band_expand_px}px之后的上下带状区域内(红/蓝两条线)——"
              f"左右方向没有任何约束，故意不封。")
        print("如果看台(通常在红/蓝两条线的上方或下方)还是被高亮了，说明"
              "band-expand-px放得太宽，需要调小；如果球员/裁判在场边被漏了，"
              "适当调大band-expand-px，或者调chi2-threshold/keep-fraction。")
    else:
        print("绿色半透明区域 = 当前标定判定为'球场内'的范围；"
              "如果绿色区域把底线外的观众席/摄影记者也圈进去了，"
              "说明需要加end_lines(底线)约束，或者调整现有的far_line/near_line。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", help="过滤模式: 输入的追踪json")
    parser.add_argument("--sideline-points", required=True, help="标定文件路径")
    parser.add_argument("--output", help="过滤模式: 输出的过滤后json")
    parser.add_argument("--video", help="可视化模式必填；过滤模式下只有加了--color-adaptive才需要")
    parser.add_argument("--visualize", help="可视化模式: 预览图输出路径，指定这个参数即进入可视化模式")
    parser.add_argument("--preview-frame", type=int, default=0, help="可视化模式: 用第几帧做预览，默认0")

    parser.add_argument("--color-adaptive", action="store_true",
                         help="用球场颜色自适应判断代替固定边线坐标，扛得住clip内镜头推拉/平移")
    parser.add_argument("--ref-frame", type=int, default=0,
                         help="颜色自适应模式: 用第几帧的标定区域采样颜色，建议选能代表实际比赛画面(不是片头远景)的帧，默认0")
    parser.add_argument("--patch-half", type=int, default=7, help="颜色自适应模式: 脚部采样窗口半径(像素)，默认7")
    parser.add_argument("--chi2-threshold", type=float, default=6.0, help="颜色自适应模式: 马氏距离阈值，默认6.0")
    parser.add_argument("--keep-fraction", type=float, default=0.35,
                         help="颜色自适应模式: 局部窗口颜色匹配像素占比超过这个值才判'在场内'，默认0.35")
    parser.add_argument("--n-clusters", type=int, default=4,
                         help="颜色自适应模式: 球场颜色聚成几类(木地板/漆色/边线等颜色差异大，"
                              "分开拟合比强行用一个模型准)，默认4")
    parser.add_argument("--margin-px", type=int, default=15,
                         help="颜色自适应模式: 采样颜色时往标定区域内缩多少像素(避开边界过渡色污染样本)，默认15")
    parser.add_argument("--band-expand-px", type=int, default=120,
                         help="颜色自适应模式: 只用far_line/near_line(不含end_lines)围出的上下带状区域，"
                              "往外放宽多少像素再判'在带状区域内'，用来单独排除看台(通常在这条带状区域"
                              "明显上方/下方)，左右方向完全不设边界。默认120，看台还漏就调小，"
                              "球员/裁判在场边被误删就调大")
    parser.add_argument("--max-persons-per-frame", type=int, default=15,
                         help="颜色自适应过滤模式: 单帧最多保留几个人(10名球员+最多几个裁判，"
                              "留点余量)，超过这个数按'整段clip历史出现帧数+颜色匹配度+离band边界"
                              "深度'排序，只留最像真在场内的前N个——历史出现帧数多的track优先保留，"
                              "避免真实球员因为贴身对抗时某一帧颜色匹配分被拖累就被一次性候选"
                              "(大概率是边缘探进画面的观众)挤掉。"
                              "默认15，传0或负数关闭这个限制")
    args = parser.parse_args()

    if args.visualize:
        if not args.video:
            print("可视化模式需要指定 --video")
            return
        run_visualize(args.sideline_points, args.video, args.visualize, args.preview_frame,
                      color_adaptive=args.color_adaptive, ref_frame_idx=args.ref_frame,
                      patch_half=args.patch_half, chi2_threshold=args.chi2_threshold,
                      n_clusters=args.n_clusters, margin_px=args.margin_px,
                      band_expand_px=args.band_expand_px)
        return

    if not args.json or not args.output:
        print("过滤模式需要同时指定 --json 和 --output")
        return

    if args.color_adaptive:
        if not args.video:
            print("颜色自适应模式需要指定 --video")
            return
        run_filter_color_adaptive(args.json, args.sideline_points, args.video, args.output,
                                   ref_frame_idx=args.ref_frame, patch_half=args.patch_half,
                                   chi2_threshold=args.chi2_threshold,
                                   keep_fraction_threshold=args.keep_fraction,
                                   n_clusters=args.n_clusters, margin_px=args.margin_px,
                                   band_expand_px=args.band_expand_px,
                                   max_persons_per_frame=args.max_persons_per_frame)
        return

    run_filter(args.json, args.sideline_points, args.output)


if __name__ == "__main__":
    main()


    