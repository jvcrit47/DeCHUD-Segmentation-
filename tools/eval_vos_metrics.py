#!/usr/bin/env python3
"""
eval_vos_metrics.py

Compute VOS segmentation metrics from folders of PNG masks:

- J  : region similarity (mean IoU)
- F  : boundary accuracy (F-measure on boundaries)
- J&F: (J + F)/2

Works for YouTube-VOS and DAVIS style layouts:
  <ROOT>/Annotations/<video_id>/<frame>.png

Assumptions:
- Masks are PNG where pixel values are integer labels:
    0 = background, 1..N = object IDs
- Pred mask uses the same labeling convention.

Dependencies:
  pip install numpy pillow scipy tqdm
"""

import argparse, json, math
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from scipy.ndimage import binary_erosion, distance_transform_edt
except Exception as e:
    raise SystemExit(
        "Missing dependency: scipy\n"
        "Install with: pip install scipy\n"
        f"Original error: {e}"
    )

def read_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.int32)

def mask_labels_union(gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
    labs = np.union1d(np.unique(gt), np.unique(pr))
    return labs[labs != 0]

def iou_binary(gt: np.ndarray, pr: np.ndarray) -> float:
    inter = np.logical_and(gt, pr).sum(dtype=np.float64)
    union = np.logical_or(gt, pr).sum(dtype=np.float64)
    if union == 0:
        return 1.0
    return float(inter / union)

def seg2bmap(seg: np.ndarray) -> np.ndarray:
    seg = seg.astype(bool)
    er = binary_erosion(seg, structure=np.ones((3,3)), border_value=0)
    b = np.logical_xor(seg, er)
    return b.astype(np.uint8)

def boundary_f_measure(gt: np.ndarray, pr: np.ndarray, bound_th: float = 0.008) -> float:
    gt = gt.astype(bool)
    pr = pr.astype(bool)

    if gt.sum() == 0 and pr.sum() == 0:
        return 1.0
    if gt.sum() == 0 or pr.sum() == 0:
        return 0.0

    gt_b = seg2bmap(gt)
    pr_b = seg2bmap(pr)

    h, w = gt.shape
    diag = math.sqrt(h*h + w*w)
    tol = max(1, int(round(bound_th * diag)))

    gt_dt = distance_transform_edt(1 - gt_b)
    pr_dt = distance_transform_edt(1 - pr_b)

    pr_match = (pr_b.astype(bool) & (gt_dt <= tol))
    gt_match = (gt_b.astype(bool) & (pr_dt <= tol))

    n_pr = float(pr_b.sum())
    n_gt = float(gt_b.sum())

    precision = float(pr_match.sum()) / max(1.0, n_pr)
    recall    = float(gt_match.sum()) / max(1.0, n_gt)

    if precision + recall == 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))

def per_frame_metrics(gt_path: Path, pr_path: Path, bound_th: float) -> tuple[float, float]:
    gt = read_mask(gt_path)
    pr = read_mask(pr_path)

    labels = mask_labels_union(gt, pr)
    if labels.size == 0:
        return 1.0, 1.0

    Js, Fs = [], []
    for lab in labels:
        gt_bin = (gt == lab)
        pr_bin = (pr == lab)
        Js.append(iou_binary(gt_bin, pr_bin))
        Fs.append(boundary_f_measure(gt_bin, pr_bin, bound_th=bound_th))
    return float(np.mean(Js)), float(np.mean(Fs))

def list_videos(ann_dir: Path) -> list[str]:
    vids = [p.name for p in ann_dir.iterdir() if p.is_dir()]
    vids.sort()
    return vids

def list_frames(video_dir: Path) -> list[Path]:
    return sorted(video_dir.glob("*.png"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, type=Path, help="GT Annotations directory (contains video folders).")
    ap.add_argument("--pred", required=True, type=Path, help="Pred Annotations directory (contains video folders).")
    ap.add_argument("--bound_th", type=float, default=0.008, help="Boundary tolerance fraction of image diagonal.")
    ap.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    ap.add_argument("--strict", action="store_true", help="Error if any predicted frame is missing (default: skip).")
    ap.add_argument("--max_videos", type=int, default=0, help="Debug: limit number of videos (0=all).")
    args = ap.parse_args()

    gt_root, pr_root = args.gt, args.pred
    if not gt_root.exists():
        raise FileNotFoundError(f"GT Annotations dir not found: {gt_root}")
    if not pr_root.exists():
        raise FileNotFoundError(f"Pred Annotations dir not found: {pr_root}")

    vids = list_videos(gt_root)
    if args.max_videos and args.max_videos > 0:
        vids = vids[:args.max_videos]

    video_scores = {}
    all_J, all_F = [], []

    for vid in tqdm(vids, desc="Videos"):
        gt_vid = gt_root / vid
        pr_vid = pr_root / vid

        if not pr_vid.exists():
            if args.strict:
                raise FileNotFoundError(f"Missing pred video folder: {pr_vid}")
            continue

        gt_frames = list_frames(gt_vid)
        if not gt_frames:
            continue

        J_frames, F_frames = [], []
        for gt_f in gt_frames:
            pr_f = pr_vid / gt_f.name
            if not pr_f.exists():
                if args.strict:
                    raise FileNotFoundError(f"Missing pred frame: {pr_f}")
                continue
            J, F = per_frame_metrics(gt_f, pr_f, bound_th=args.bound_th)
            J_frames.append(J); F_frames.append(F)

        if not J_frames:
            continue

        Jv = float(np.mean(J_frames))
        Fv = float(np.mean(F_frames))
        video_scores[vid] = {"J": Jv, "F": Fv, "J&F": float((Jv + Fv) / 2.0)}
        all_J.append(Jv); all_F.append(Fv)

    if not all_J:
        raise SystemExit("No videos/frames evaluated. Check paths and naming.")

    mean_J = float(np.mean(all_J))
    mean_F = float(np.mean(all_F))
    mean_JF = float((mean_J + mean_F) / 2.0)

    print("\n===== VOS METRICS (Local) =====")
    print(f"GT:   {gt_root}")
    print(f"PRED: {pr_root}")
    print(f"Mean J (IoU): {mean_J:.4f}  ({mean_J*100:.2f}%)")
    print(f"Mean F:       {mean_F:.4f}  ({mean_F*100:.2f}%)")
    print(f"Mean J&F:     {mean_JF:.4f}  ({mean_JF*100:.2f}%)")

    items = sorted(video_scores.items(), key=lambda kv: kv[1]["J&F"])
    print("\nWorst 10 videos by J&F:")
    for vid, sc in items[:10]:
        print(f"  {vid}: J&F {sc['J&F']:.4f} (J {sc['J']:.4f}, F {sc['F']:.4f})")

    print("\nBest 10 videos by J&F:")
    for vid, sc in items[-10:][::-1]:
        print(f"  {vid}: J&F {sc['J&F']:.4f} (J {sc['J']:.4f}, F {sc['F']:.4f})")

    out = {
        "gt": str(gt_root),
        "pred": str(pr_root),
        "bound_th": args.bound_th,
        "mean": {"J": mean_J, "F": mean_F, "J&F": mean_JF},
        "per_video": video_scores,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2))
        print(f"\nWrote: {args.out}")

if __name__ == "__main__":
    main()
