#!/usr/bin/env python3
"""Standalone runner for the repository object detection pipeline.

Uses the project model, loss, and dataset modules without server or database imports.
Dependencies: torch, torchvision, albumentations, opencv-python-headless,
             numpy, matplotlib, pillow (use the same versions as training/server).

Example:
    python train.py --input-dir /kaggle/input/ua-detrac --output-dir /kaggle/working/od_output

Output: <output-dir>/<job-id>/ containing model.pt, metadata.json,
training_report.json, training_graph.jpg and validation images, plus <job-id>.zip.
Validation uses held-out videos/sequences, separated from training data.

Training/evaluation and artifact packaging are standalone. The canonical model,
loss, and dataset are imported from model_developement/object_detection.
"""
from __future__ import annotations

import argparse
import io
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
from PIL import Image

DEMO_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = DEMO_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from model_developement.object_detection.dataset import ObjectDetectionDataset, detection_collate_fn
from model_developement.object_detection.loss import ObjectDetectionLoss
from model_developement.object_detection.model import ObjectDetectionModel
from core.evaluation import detection_map, grouped_train_val_split

logger = logging.getLogger("od_standalone")



# ---- Standalone utility functions ----
def append_time_left_to_ui_status(
    ui_status: str,
    estimated_remaining_seconds: Optional[float],
) -> str:
    """
    Always append a time-left fragment so every status line mentions time:
    - unknown estimate -> (time left: unknown)
    - non-positive -> (ended)
    - positive -> (~Xs left) or (~Xm Ys left)
    """
    if estimated_remaining_seconds is None:
        return f"{ui_status} (time left: unknown)"
    try:
        rem = float(estimated_remaining_seconds)
    except (TypeError, ValueError):
        return f"{ui_status} (time left: unknown)"
    if rem <= 0:
        return f"{ui_status} (ended)"
    if rem >= 60:
        mins = int(rem // 60)
        secs = int(round(rem % 60))
        if secs == 60:
            mins += 1
            secs = 0
        return f"{ui_status} (~{mins}m {secs}s left)"
    return f"{ui_status} (~{max(0, int(round(rem)))}s left)"

def numpy_to_jpg(np_array):
    if np_array.max() > 255:
        np_array = (np_array / np_array.max()) * 255
    np_array = np_array.astype(np.uint8)

    image = Image.fromarray(np_array)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.read()


# Standalone training/evaluation pipeline; model components come from the project modules.
import os
import shutil
import tempfile
import time
import zipfile
import json
import traceback
from typing import Callable, Dict, Any

import torch
import torch.optim as optim
import torchvision.ops as ops
from torch.utils.data import DataLoader, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None


def get_loss_graph(epoch_losses: list) -> np.ndarray:
    fig = plt.figure(figsize=(8, 5))
    plt.plot(epoch_losses, label='Total Loss', color='blue', linewidth=2)
    plt.title('Training Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    
    fig.canvas.draw()
    data = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return data


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Normalized YOLO (cx, cy, w, h) -> (x1, y1, x2, y2)."""
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    return ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")


def render_eval_image(image_tensor: torch.Tensor, entry: dict) -> np.ndarray:
    """Draw GT and prediction boxes for one per-image evaluation entry.
    GT: green solid ("GT <label>"), missed GT: orange solid ("✗ missed <label>").
    Predictions: dashed — green with "✓ <label> <score>" when matched to a GT,
    red with "✗ <label> <score>" for false positives. Boxes are normalized xyxy.
    """
    img_np = image_tensor.cpu().numpy()
    if img_np.shape[0] == 3:
        img_np = np.transpose(img_np, (1, 2, 0))

    img_np = np.clip(img_np, 0, 1)
    H, W = img_np.shape[:2]

    fig, ax = plt.subplots(1, figsize=(10, 6))
    ax.imshow(img_np)

    def _draw(box, color, text, linestyle):
        x1, y1, x2, y2 = [v * s for v, s in zip(box, (W, H, W, H))]
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor='none', linestyle=linestyle,
        )
        ax.add_patch(rect)
        ax.text(
            x1, max(y1 - 3, 0), text, fontsize=9, color='white',
            bbox=dict(facecolor=color, alpha=0.8, pad=1, edgecolor='none'),
            verticalalignment='bottom',
        )

    for gt in entry.get("ground_truth", []):
        if gt["detected"]:
            _draw(gt["box"], 'green', f"GT {gt['label']}", '-')
        else:
            _draw(gt["box"], 'darkorange', f"\u2717 missed {gt['label']}", '-')

    for pred in entry.get("predictions", []):
        mark = '\u2713' if pred["correct"] else '\u2717'
        color = 'limegreen' if pred["correct"] else 'red'
        _draw(pred["box"], color, f"{mark} {pred['label']} {pred['score']:.2f}", '--')

    ax.set_title(
        f"{entry.get('image', '')} — "
        f"P {entry['precision']:.2f}  R {entry['recall']:.2f}  F1 {entry['f1']:.2f}  "
        f"({entry['inference_time_ms']:.0f} ms)",
        fontsize=10,
    )
    plt.axis('off')

    fig.canvas.draw()
    data = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return data


def decode_predictions(predictions, conf_thresh=0.2, nms_thresh=0.45, num_classes=20, h_model=352, w_model=640, class_ids=None):
    """
    Multi-scale decoding for YOLO-like outputs. 
    Returns bounding boxes in normalized [x1, y1, x2, y2] format.
    """
    if not isinstance(predictions, list):
        raise ValueError("Predictions must be a list of 3 tensors")
        
    strides = [8, 16, 32]
    outputs = []

    for k, stride in enumerate(strides):
        pred = predictions[k]
        if len(pred.shape) == 3:
            pred = pred.unsqueeze(0) 
            
        B, C, H, W = pred.shape
        
        yv, xv = torch.meshgrid([torch.arange(H), torch.arange(W)], indexing='ij')
        grid = torch.stack((xv, yv), 2).view(1, H * W, 2).type_as(pred)

        pred = pred.view(B, C, H * W).permute(0, 2, 1)  # [B, H*W, 5+C]
        xy = (pred[..., :2] + grid) * stride
        wh = torch.exp(pred[..., 2:4]) * stride
        pred = torch.cat([xy, wh, pred[..., 4:]], dim=-1)

        outputs.append(pred)

    outputs = torch.cat(outputs, 1) 
    
    output = outputs[0]
    
    bboxes = output[:, 0:4] # x_c, y_c, w, h
    obj_conf = torch.sigmoid(output[:, 4]) # Objectness score
    cls_conf = torch.sigmoid(output[:, 5:]) # Class scores
    
    # Optionally restrict inference to classes represented in the test set.
    if class_ids is None:
        max_cls_conf, labels = torch.max(cls_conf, dim=1)
    else:
        class_ids = torch.as_tensor(class_ids, dtype=torch.long, device=cls_conf.device)
        if class_ids.numel() == 0 or torch.any(class_ids < 0) or torch.any(class_ids >= num_classes):
            raise ValueError("class_ids must contain valid class indices")
        max_cls_conf, class_positions = torch.max(cls_conf[:, class_ids], dim=1)
        labels = class_ids[class_positions]
    final_scores = obj_conf * max_cls_conf
    
    # Filter by Confidence
    mask = final_scores > conf_thresh
    filtered_bboxes = bboxes[mask]
    filtered_scores = final_scores[mask]
    filtered_labels = labels[mask]
    
    if len(filtered_bboxes) == 0:
        return torch.empty((0, 4)), torch.empty(0), torch.empty(0, dtype=torch.long)

    # NMS per-class
    boxes_xyxy_abs = ops.box_convert(filtered_bboxes, in_fmt='cxcywh', out_fmt='xyxy')
    keep = ops.batched_nms(boxes_xyxy_abs, filtered_scores, filtered_labels, nms_thresh)
    filtered_bboxes = boxes_xyxy_abs[keep]
    filtered_scores = filtered_scores[keep]
    filtered_labels = filtered_labels[keep]

    if len(filtered_bboxes) == 0:
        return torch.empty((0, 4)), torch.empty(0), torch.empty(0, dtype=torch.long)

    # Normalize to [0, 1] xyxy format
    filtered_bboxes_norm = filtered_bboxes.clone()
    filtered_bboxes_norm[:, 0] = filtered_bboxes_norm[:, 0] / w_model
    filtered_bboxes_norm[:, 1] = filtered_bboxes_norm[:, 1] / h_model
    filtered_bboxes_norm[:, 2] = filtered_bboxes_norm[:, 2] / w_model
    filtered_bboxes_norm[:, 3] = filtered_bboxes_norm[:, 3] / h_model
    
    return filtered_bboxes_norm, filtered_scores, filtered_labels


def decode_one2one_predictions(predictions, conf_thresh=0.2, num_classes=20,
                               h_model=352, w_model=640, class_ids=None, max_detections=300):
    """Decode the trained one-to-one head using confidence filtering only."""
    strides = [8, 16, 32]
    levels = []
    for pred, stride in zip(predictions, strides):
        if pred.ndim == 3:
            pred = pred.unsqueeze(0)
        batch, channels, height, width = pred.shape
        y, x = torch.meshgrid(torch.arange(height, device=pred.device),
                              torch.arange(width, device=pred.device), indexing="ij")
        grid = torch.stack((x, y), -1).reshape(1, height * width, 2).to(pred.dtype)
        pred = pred.flatten(2).transpose(1, 2)
        xy = (pred[..., :2] + grid) * stride
        wh = pred[..., 2:4].clamp(max=math.log(max(h_model, w_model) / stride)).exp() * stride
        levels.append(torch.cat((xy, wh, pred[..., 4:]), dim=-1))
    output = torch.cat(levels, dim=1)[0]
    boxes = ops.box_convert(output[:, :4], "cxcywh", "xyxy")
    obj = output[:, 4].sigmoid()
    cls = output[:, 5:].sigmoid()
    if class_ids is None:
        cls_score, labels = cls.max(dim=1)
    else:
        class_ids = torch.as_tensor(class_ids, dtype=torch.long, device=cls.device)
        if class_ids.numel() == 0 or torch.any(class_ids < 0) or torch.any(class_ids >= num_classes):
            raise ValueError("class_ids must contain valid class indices")
        cls_score, positions = cls[:, class_ids].max(dim=1)
        labels = class_ids[positions]
    scores = obj * cls_score
    valid = (scores > conf_thresh) & torch.isfinite(boxes).all(1) & torch.isfinite(scores)
    boxes, scores, labels = boxes[valid], scores[valid], labels[valid]
    if boxes.numel() == 0:
        return boxes.reshape(0, 4), scores, labels
    if max_detections is not None and len(scores) > max_detections:
        selected = scores.topk(max_detections).indices
        boxes, scores, labels = boxes[selected], scores[selected], labels[selected]
    scale = boxes.new_tensor([w_model, h_model, w_model, h_model])
    return (boxes / scale).clamp(0, 1), scores, labels


def decode_target(target: torch.Tensor):
    if len(target) == 0:
        return (
            torch.empty((0, 4)),
            torch.empty((0,), dtype=torch.long)
        )

    labels = target[:, 0].long()
    boxes = _cxcywh_to_xyxy(target[:, 1:5])
    return boxes, labels


def precision_recall_f1(tp: int, fp: int, fn: int):
    """Vacuous cases are scored as perfect: no predictions -> precision 1,
    no ground truth -> recall 1. The old ``TP / max(TP + FP, 1)`` scored an
    empty image with no predictions as precision 0, which is wrong."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def evaluate_prediction(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, iou_threshold=0.5) -> dict:
    """Greedy score-ordered, class-aware, one-to-one matching at ``iou_threshold``.

    Returns TP/FP/FN, precision/recall/f1 (always present, unlike the old
    early-return branches which omitted them), and ``matches`` — a list of
    (pred_index, gt_index, iou) for the matched pairs.
    """
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)

    matches = []
    TP = 0

    if n_pred > 0 and n_gt > 0:
        if not isinstance(pred_boxes, torch.Tensor):
            pred_boxes = torch.tensor(pred_boxes, dtype=torch.float32)
        if not isinstance(gt_boxes, torch.Tensor):
            gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32)
        if not isinstance(gt_labels, torch.Tensor):
            gt_labels = torch.tensor(gt_labels, dtype=torch.long)

        order = torch.argsort(pred_scores, descending=True)
        ious = ops.box_iou(pred_boxes, gt_boxes)

        matched = set()

        for idx in order:
            idx = idx.item()
            best_iou = 0.0
            best_gt = -1

            for gt in range(n_gt):
                if gt in matched:
                    continue
                if int(pred_labels[idx]) != int(gt_labels[gt]):
                    continue
                if float(ious[idx, gt]) > best_iou:
                    best_iou = float(ious[idx, gt])
                    best_gt = gt

            if best_gt >= 0 and best_iou >= iou_threshold:
                TP += 1
                matched.add(best_gt)
                matches.append((idx, best_gt, best_iou))

    FP = n_pred - TP
    FN = n_gt - TP
    precision, recall, f1 = precision_recall_f1(TP, FP, FN)

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matches": matches,
    }


def save_od_output_zip(
    job_id: str, 
    weights_path: str, 
    metadata: dict, 
    training_report: dict,
    training_graph: np.ndarray = None,
    eval_images: Dict[str, np.ndarray] = None,
) -> str:
    model_zip_folder = os.getenv("MODEL_ZIP_FOLDER", "./runs/outputs")
    os.makedirs(model_zip_folder, exist_ok=True)
    zip_path = os.path.join(model_zip_folder, f"{job_id}.zip")

    with zipfile.ZipFile(zip_path, "w") as zipf:
        zipf.write(weights_path, "model.pt")
        zipf.writestr("metadata.json", json.dumps(metadata, indent=4))
        zipf.writestr("training_report.json", json.dumps(training_report, indent=4))

        if training_graph is not None:
            zipf.writestr("training_graph.jpg", numpy_to_jpg(training_graph))

        for name, image in (eval_images or {}).items():
            zipf.writestr(f"eval_images/{name}", numpy_to_jpg(image))

    return zip_path


def evaluate_detector(model, dataset, image_indices, confidence, num_classes,
                      image_h, image_w, device, use_amp, batch_size,
                      class_map, include_details=False):
    """Evaluate the NMS-free head on held-out images and compute AP metrics."""
    subset = Subset(dataset, image_indices)
    loader = DataLoader(
        subset, batch_size=min(max(int(batch_size), 1), 2), shuffle=False,
        num_workers=0, pin_memory=device.type == "cuda",
        collate_fn=detection_collate_fn,
    )
    all_predictions, all_ground_truth = [], []
    detail_rows = []
    tp = fp = fn = 0
    inference_ms = 0.0
    evaluated = 0
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            started = time.perf_counter()
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                outputs = model(images)
            batch_ms = (time.perf_counter() - started) * 1000.0
            inference_ms += batch_ms
            for batch_index, target in enumerate(targets):
                dataset_index = image_indices[evaluated]
                levels = [level[batch_index].float().cpu() for level in outputs]
                boxes, scores, labels = decode_one2one_predictions(
                    levels, conf_thresh=0.001, num_classes=num_classes,
                    h_model=image_h, w_model=image_w, max_detections=300,
                )
                gt_boxes, gt_labels = decode_target(target)
                for box, score, label in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
                    all_predictions.append({
                        "image_index": evaluated, "class_id": int(label),
                        "score": float(score), "box": box,
                    })
                for box, label in zip(gt_boxes.tolist(), gt_labels.tolist()):
                    all_ground_truth.append({
                        "image_index": evaluated, "class_id": int(label), "box": box,
                    })

                keep = scores >= confidence
                kept_boxes, kept_scores, kept_labels = boxes[keep], scores[keep], labels[keep]
                image_metrics = evaluate_prediction(
                    kept_boxes, kept_scores, kept_labels, gt_boxes, gt_labels,
                )
                tp += image_metrics["TP"]
                fp += image_metrics["FP"]
                fn += image_metrics["FN"]
                if include_details:
                    matched_preds = {pred for pred, _, _ in image_metrics["matches"]}
                    matched_gts = {gt for _, gt, _ in image_metrics["matches"]}
                    display_count = min(len(kept_scores), 100)
                    detail_rows.append({
                        "image_index": dataset_index,
                        "image": dataset.image_files[dataset_index],
                        "TP": image_metrics["TP"], "FP": image_metrics["FP"], "FN": image_metrics["FN"],
                        "precision": image_metrics["precision"],
                        "recall": image_metrics["recall"], "f1": image_metrics["f1"],
                        "inference_time_ms": batch_ms / max(len(images), 1),
                        "predictions": [
                            {"box": [round(float(v), 4) for v in kept_boxes[i].tolist()],
                             "label": class_map.get(int(kept_labels[i]), str(int(kept_labels[i]))),
                             "score": round(float(kept_scores[i]), 4), "correct": i in matched_preds}
                            for i in range(display_count)
                        ],
                        "ground_truth": [
                            {"box": [round(float(v), 4) for v in gt_boxes[g].tolist()],
                             "label": class_map.get(int(gt_labels[g]), str(int(gt_labels[g]))),
                             "detected": g in matched_gts}
                            for g in range(len(gt_boxes))
                        ],
                    })
                evaluated += 1

    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    ap = detection_map(all_predictions, all_ground_truth, range(num_classes))
    evaluation = {
        "images": evaluated,
        "TP": tp, "FP": fp, "FN": fn,
        "precision_at_confidence": precision,
        "recall_at_confidence": recall,
        "f1_at_confidence": f1,
        "confidence_threshold": confidence,
        "map50": ap["map50"], "map75": ap["map75"], "map50_95": ap["map50_95"],
        "ap_protocol": "101-point interpolated AP, IoU 0.50:0.05:0.95, max 300 detections/image",
        "per_class": {
            class_map.get(int(class_id), str(class_id)): metrics
            for class_id, metrics in ((int(key), value) for key, value in ap["per_class"].items())
        },
        "avg_inference_time_ms": inference_ms / max(evaluated, 1),
    }
    return evaluation, detail_rows


def save_preview_images(model, dataset, image_indices, preview_root, images_seen,
                        confidence, model_h, model_w, device, use_amp):
    """Render the same held-out images at a training-image milestone."""
    preview_dir = Path(preview_root) / f"images_seen_{images_seen:09d}"
    preview_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.inference_mode():
        for preview_index in image_indices:
            preview_image, preview_target = dataset[preview_index]
            preview_input = preview_image.unsqueeze(0).to(
                device, non_blocking=device.type == "cuda"
            )
            started = time.perf_counter()
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                outputs = model(preview_input)
            inference_ms = (time.perf_counter() - started) * 1000.0
            levels = [level[0].float().cpu() for level in outputs]
            boxes, scores, labels = decode_one2one_predictions(
                levels, conf_thresh=confidence, num_classes=dataset.num_classes,
                h_model=model_h, w_model=model_w,
            )
            if len(scores) > 100:
                selected = scores.topk(100).indices
                boxes, scores, labels = boxes[selected], scores[selected], labels[selected]
            gt_boxes, gt_labels = decode_target(preview_target)
            metrics = evaluate_prediction(boxes, scores, labels, gt_boxes, gt_labels)
            matched_preds = {pred for pred, _, _ in metrics["matches"]}
            matched_gts = {gt for _, gt, _ in metrics["matches"]}
            preview_entry = {
                "image": dataset.image_files[preview_index],
                "precision": metrics["precision"], "recall": metrics["recall"],
                "f1": metrics["f1"], "inference_time_ms": inference_ms,
                "predictions": [
                    {"box": [float(v) for v in boxes[i]],
                     "label": dataset.class_map.get(int(labels[i]), str(int(labels[i]))),
                     "score": float(scores[i]), "correct": i in matched_preds}
                    for i in range(len(boxes))
                ],
                "ground_truth": [
                    {"box": [float(v) for v in gt_boxes[g]],
                     "label": dataset.class_map.get(int(gt_labels[g]), str(int(gt_labels[g]))),
                     "detected": g in matched_gts}
                    for g in range(len(gt_boxes))
                ],
            }
            preview_array = render_eval_image(preview_image, preview_entry)
            preview_name = (f"image_{preview_index:06d}_"
                            f"{Path(dataset.image_files[preview_index]).stem}.jpg")
            Image.fromarray(preview_array).save(preview_dir / preview_name, quality=90)
    model.train()
    snapshots = sorted(
        path for path in Path(preview_root).glob("images_seen_*")
        if path.is_dir() and path.name.removeprefix("images_seen_").isdigit()
    )
    for stale_snapshot in snapshots[:-3]:
        shutil.rmtree(stale_snapshot)
    return preview_dir


def write_json_atomic(path: Path, payload: dict) -> None:
    """Atomically write JSON so a killed process cannot leave partial metadata."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=4), encoding="utf-8")
    os.replace(temporary, path)


def progress_iterator(iterable, enabled, description, total=None):
    if not enabled or _tqdm is None:
        return iterable
    return _tqdm(iterable, total=total, desc=description, dynamic_ncols=True, leave=False)


def find_yolo_dataset_root(input_dir: Path) -> Path:
    """Resolve a Kaggle-mounted dataset root without copying or extracting it."""
    input_dir = Path(input_dir).resolve()
    candidates = [input_dir]
    try:
        children = [path for path in input_dir.iterdir()
                    if path.is_dir() and path.name not in {"images", "labels"}]
    except OSError as exc:
        raise ValueError(f"Cannot read dataset directory: {input_dir}: {exc}") from exc
    candidates.extend(children)
    for child in children:
        try:
            candidates.extend(path for path in child.iterdir()
                             if path.is_dir() and path.name not in {"images", "labels"})
        except OSError:
            continue
    for candidate in candidates:
        if ((candidate / "images").is_dir()
                and (candidate / "labels").is_dir()
                and (candidate / "metadata.json").is_file()):
            return candidate
    raise ValueError(
        f"Could not find a YOLO dataset under {input_dir}; expected images/, "
        "labels/ and metadata.json in that folder or up to two levels below it"
    )


def run_od_pipeline(
    job_id: str, 
    job_definition: dict, 
    job_dao: object, 
    report_ui_func: Callable
) -> dict:
    report_ui_func(job_dao, job_id, ui_status="Starting Object Detection job.", progress_percent=0, stage="start")
    
    extract_tmp_dir = None
    tmp_weights_dir = None 

    try:
        input_dir = job_definition.get("input_dir")
        zip_path = job_definition.get("zip_path")
        if input_dir:
            report_ui_func(job_dao, job_id, ui_status="Reading mounted dataset...",
                           progress_percent=5, stage="load_dataset")
            training_dir = find_yolo_dataset_root(Path(input_dir))
            logger.info("Using dataset directory directly: %s", training_dir)
        else:
            report_ui_func(job_dao, job_id, ui_status="Extracting images...",
                           progress_percent=5, stage="extract_zip")
            extract_tmp_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_tmp_dir)
            training_dir = find_yolo_dataset_root(Path(extract_tmp_dir))

        config_str = job_definition.get("configuration")
        config_dict = {}
        if config_str:
            try:
                config_dict = json.loads(config_str)
            except Exception:
                pass

        requested_device = config_dict.get("device", "auto")
        if requested_device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(requested_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        cpu_threads = int(config_dict.get("cpu_threads") or min(8, os.cpu_count() or 1))
        torch.set_num_threads(cpu_threads)
        print(f"Training device: {device} (CPU threads: {cpu_threads})")

        progress_mode = config_dict.get("progress", "auto")
        progress_enabled = progress_mode == "on" or (
            progress_mode == "auto" and sys.stdout.isatty()
            and not os.environ.get("STY") and not os.environ.get("TMUX")
        )
        if progress_enabled and _tqdm is None:
            logger.warning("Install tqdm to enable progress bars; continuing without them")
            progress_enabled = False
        train_metrics = bool(config_dict.get("train_metrics", False))
        num_workers = int(config_dict.get("workers", min(4, os.cpu_count() or 1)))
        eval_limit = int(config_dict.get("eval_limit", 2000))
        epoch_val_limit = int(config_dict.get("epoch_val_limit", 256))
        log_every_images = int(config_dict.get("log_every_images", 500))
        preview_every_images = int(config_dict.get("preview_every_images", 2000))
        preview_count = int(config_dict.get("preview_images", 4))
        preview_root = job_definition.get("preview_dir")
        use_amp = bool(config_dict.get("amp", True)) and device.type == "cuda"
        if (num_workers < 0 or eval_limit < 0 or epoch_val_limit <= 0
                or log_every_images <= 0 or preview_every_images < 0 or preview_count <= 0):
            raise ValueError("workers/eval_limit/preview interval must be non-negative; validation, log and preview-image values must be positive")

        full_dataset = ObjectDetectionDataset(data_dir=training_dir, image_size=(360, 640))
        train_image_names, val_image_names, split_manifest = grouped_train_val_split(
            full_dataset,
            val_fraction=float(config_dict.get("val_fraction", 0.2)),
            seed=int(config_dict.get("split_seed", 42)),
        )
        train_dataset = ObjectDetectionDataset(
            data_dir=training_dir, image_size=full_dataset.image_size,
            image_files=train_image_names,
        )
        val_dataset = ObjectDetectionDataset(
            data_dir=training_dir, image_size=full_dataset.image_size,
            image_files=val_image_names,
        )
        print(f"Dataset images: {len(full_dataset)}")
        print(f"Training images: {len(train_dataset)}; validation images: {len(val_dataset)}")
        print(f"Classes: {train_dataset.class_map}")
        logger.info(
            "Split: %d train groups, %d validation groups; actual validation fraction %.3f; class objects %s",
            len(split_manifest["train_groups"]), len(split_manifest["validation_groups"]),
            split_manifest["actual_validation_fraction"], split_manifest["validation_class_objects"],
        )

        model_h, model_w = train_dataset.image_size

        if len(train_dataset) == 0:
            raise ValueError("Training dataset contains no images.")
        if len(val_dataset) == 0:
            raise ValueError("Validation split contains no images.")

        num_epochs = config_dict.get("epochs", 40)
        batch_size = config_dict.get("batch_size", 4)
        lr = config_dict.get("learning_rate", 5e-4)
        
        p_flip = config_dict.get("flip_p", 0)
        p_brightness = config_dict.get("brightness_p", 0)
        brightness_limit = config_dict.get("brightness_limit", 0.2)
        p_rotate = config_dict.get("rotate_p", 0)
        rotate_angle = config_dict.get("rotate_angle", 10)

        train_transform = A.Compose(
            [
                A.Resize(height=model_h, width=model_w),
                A.HorizontalFlip(p=p_flip),
                A.RandomBrightnessContrast(
                    brightness_limit=brightness_limit,
                    contrast_limit=0.2,
                    p=p_brightness,
                ),
                A.Rotate(limit=rotate_angle, p=p_rotate),
                A.Normalize(
                    mean=(0.0, 0.0, 0.0),
                    std=(1.0, 1.0, 1.0),
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="yolo",
                label_fields=["class_labels"],
                min_visibility=0.2,
            ),
        )
        
        train_dataset.transform = train_transform

        loader_options = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "collate_fn": detection_collate_fn,
            "pin_memory": device.type == "cuda",
        }
        if num_workers > 0:
            loader_options.update(persistent_workers=True, prefetch_factor=2)
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
        
        model = ObjectDetectionModel(
            num_classes=train_dataset.num_classes,
            depthwise=True
        ).to(device)

        pretrained_weights = job_definition.get("pretrained_weights") or config_dict.get("pretrained_weights")

        if not pretrained_weights and hasattr(train_dataset, "metadata"):
            pretrained_weights = train_dataset.metadata.get("pretrained_weights")

        if pretrained_weights and not os.path.exists(pretrained_weights):
            logger.warning(
                "Pretrained weights not found: %s. Training from scratch.",
                pretrained_weights,
            )
            pretrained_weights = None

        if pretrained_weights:
            print(f"Loading pretrained weights from: {pretrained_weights}")
            checkpoint = torch.load(pretrained_weights, map_location=device)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            if not isinstance(checkpoint, dict):
                raise ValueError("Pretrained checkpoint must contain a state_dict")
            legacy_checkpoint = not any(key.startswith("one2one_head.") for key in checkpoint)
            if legacy_checkpoint:
                checkpoint = dict(checkpoint)
                for key, value in list(checkpoint.items()):
                    if key.startswith("head."):
                        checkpoint["one2one_head." + key[len("head."):]] = value
            incompatible = model.load_state_dict(checkpoint, strict=False)
            expected_missing = {
                key for key in model.state_dict()
                if legacy_checkpoint and key.startswith("one2one_head.")
            }
            if incompatible.unexpected_keys or set(incompatible.missing_keys) != expected_missing:
                raise RuntimeError(
                    "Incompatible pretrained checkpoint; "
                    f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
                )
            print("Pretrained weights loaded successfully.")
        else:
            print("No pretrained weights provided. Training from scratch.")
        
        obj_focal_gamma = float(config_dict.get("obj_focal_gamma", 2.0))
        if not math.isfinite(obj_focal_gamma) or obj_focal_gamma < 0:
            raise ValueError("obj_focal_gamma must be finite and non-negative")
        criterion = ObjectDetectionLoss(
            num_classes=train_dataset.num_classes,
            obj_focal_gamma=obj_focal_gamma,
        ).to(device)
        one2many_loss_weight = float(config_dict.get("one2many_loss_weight", 0.5))
        if not math.isfinite(one2many_loss_weight) or one2many_loss_weight < 0:
            raise ValueError("one2many_loss_weight must be finite and non-negative")
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scaler = torch.amp.GradScaler(enabled=use_amp)

        report_ui_func(job_dao, job_id, ui_status="Starting training...", progress_percent=15, stage="training")
        
        training_report = {
            "job_id": job_id,
            "hyperparameters": {
                "epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "device": str(device),
                "workers": num_workers,
                "amp": use_amp,
                "train_metrics": train_metrics,
                "eval_limit": eval_limit,
                "epoch_val_limit": epoch_val_limit,
                "val_fraction": split_manifest["actual_validation_fraction"],
                "split_seed": split_manifest["seed"],
                "preview_every_images": preview_every_images,
                "preview_images": preview_count,
                "obj_focal_gamma": obj_focal_gamma,
                "one2many_loss_weight": one2many_loss_weight,
            },
            "kpi_per_epoch": [],
            "per_image_predictions": [],
            "data_split": split_manifest,
        }

        preview_dataset = None
        preview_indices = []
        if preview_root:
            preview_dataset = ObjectDetectionDataset(
                data_dir=training_dir, image_size=(model_h, model_w),
                image_files=val_image_names,
            )
            preview_indices = np.linspace(
                0, len(preview_dataset) - 1,
                min(preview_count, len(preview_dataset)), dtype=int,
            ).tolist()

        epoch_val_indices = np.linspace(
            0, len(val_dataset) - 1,
            min(epoch_val_limit, len(val_dataset)), dtype=int,
        ).tolist()

        epoch_losses = []
        checkpoint_dir = job_definition.get("checkpoint_dir")
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_weights_path = (checkpoint_dir / "best_model.pt") if checkpoint_dir else None
        best_validation_map = -1.0
        best_epoch = None
        total_images_seen = 0
        next_image_log = log_every_images
        next_preview_images = preview_every_images
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss_sum = torch.zeros((), device=device)
            epoch_one2many_loss_sum = torch.zeros((), device=device)
            epoch_one2one_loss_sum = torch.zeros((), device=device)
            epoch_one2many_box_sum = torch.zeros((), device=device)
            epoch_one2many_obj_sum = torch.zeros((), device=device)
            epoch_one2many_cls_sum = torch.zeros((), device=device)
            epoch_one2one_box_sum = torch.zeros((), device=device)
            epoch_one2one_obj_sum = torch.zeros((), device=device)
            epoch_one2one_cls_sum = torch.zeros((), device=device)
            epoch_tp = epoch_fp = epoch_fn = 0
            batches = progress_iterator(
                train_loader, progress_enabled, f"Epoch {epoch + 1}/{num_epochs}",
                total=len(train_loader),
            )

            for step, (images, targets) in enumerate(batches, start=1):
                images = images.to(device, non_blocking=device.type == "cuda")
                targets = [t.to(device, non_blocking=device.type == "cuda") for t in targets]
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    predictions, one2one_predictions = model(images, return_one2one=True)
                # Keep assignment/loss arithmetic in float32 under mixed precision.
                loss_inputs = [prediction.float() for prediction in predictions] if use_amp else predictions
                one2one_loss_inputs = [prediction.float() for prediction in one2one_predictions] if use_amp else one2one_predictions
                one2many_loss, loss_dict = criterion(loss_inputs, targets)
                one2one_loss, one2one_loss_dict = criterion.one2one_loss(one2one_loss_inputs, targets)
                loss = one2many_loss_weight * one2many_loss + one2one_loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss_sum += loss.detach().sum()
                # The legacy one-to-many criterion returns shape [1], while
                # one-to-one and the epoch accumulators are scalar tensors.
                # Reduce both paths to scalars before in-place accumulation.
                epoch_one2many_loss_sum += one2many_loss.detach().mean()
                epoch_one2one_loss_sum += one2one_loss.detach().mean()
                epoch_one2many_box_sum += loss_dict["loss_giou"].mean()
                epoch_one2many_obj_sum += loss_dict["loss_obj"].mean()
                epoch_one2many_cls_sum += loss_dict["loss_cls"].mean()
                epoch_one2one_box_sum += one2one_loss_dict["loss_giou"].mean()
                epoch_one2one_obj_sum += one2one_loss_dict["loss_obj"].mean()
                epoch_one2one_cls_sum += one2one_loss_dict["loss_cls"].mean()
                total_images_seen += len(images)

                if (preview_root and preview_dataset is not None
                        and preview_every_images > 0
                        and total_images_seen >= next_preview_images):
                    preview_dir = save_preview_images(
                        model, preview_dataset, preview_indices, preview_root,
                        total_images_seen, confidence=0.2, model_h=model_h,
                        model_w=model_w, device=device, use_amp=use_amp,
                    )
                    logger.info("Preview saved after %d training images: %s",
                                total_images_seen, preview_dir)
                    crossed = (total_images_seen - next_preview_images) // preview_every_images + 1
                    next_preview_images += crossed * preview_every_images

                if not progress_enabled and total_images_seen >= next_image_log:
                    logger.info(
                        "Training progress: epoch %d/%d, %d images seen, average loss %.4f",
                        epoch + 1, num_epochs, total_images_seen,
                        epoch_loss_sum.item() / step,
                    )
                    while next_image_log <= total_images_seen:
                        next_image_log += log_every_images

                if train_metrics:
                    with torch.no_grad():
                        for b in range(len(images)):
                            preds_for_b = [level[b].detach().float().cpu() for level in one2one_predictions]
                            pred_boxes, pred_scores, pred_labels = decode_one2one_predictions(
                                preds_for_b, conf_thresh=0.2, num_classes=train_dataset.num_classes,
                                h_model=model_h, w_model=model_w,
                            )
                            gt_boxes, gt_labels = decode_target(targets[b].cpu())
                            metrics = evaluate_prediction(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels)
                            epoch_tp += metrics["TP"]
                            epoch_fp += metrics["FP"]
                            epoch_fn += metrics["FN"]

                if progress_enabled and step % 10 == 0:
                    batches.set_postfix(
                        loss=f"{epoch_loss_sum.item() / step:.4f}",
                        o2o=f"{one2one_loss.detach().item():.3f}",
                    )

            avg_train_loss = epoch_loss_sum.item() / max(len(train_loader), 1)
            avg_one2many_loss = epoch_one2many_loss_sum.item() / max(len(train_loader), 1)
            avg_one2one_loss = epoch_one2one_loss_sum.item() / max(len(train_loader), 1)
            num_train_batches = max(len(train_loader), 1)
            epoch_losses.append(avg_train_loss)
            if not progress_enabled:
                logger.info("Epoch %d/%d complete - average loss %.4f",
                            epoch + 1, num_epochs, avg_train_loss)
            if train_metrics:
                precision, recall, f1 = precision_recall_f1(epoch_tp, epoch_fp, epoch_fn)
            else:
                precision = recall = f1 = None

            model.eval()
            optimizer.zero_grad(set_to_none=True)
            epoch_validation, _ = evaluate_detector(
                model, val_dataset, epoch_val_indices, confidence=0.2,
                num_classes=train_dataset.num_classes, image_h=model_h,
                image_w=model_w, device=device, use_amp=use_amp,
                batch_size=batch_size, class_map=train_dataset.class_map,
            )
            logger.info(
                "Validation epoch %d/%d: mAP50 %.4f, mAP50-95 %.4f, P %.4f, R %.4f",
                epoch + 1, num_epochs, epoch_validation["map50"],
                epoch_validation["map50_95"],
                epoch_validation["precision_at_confidence"],
                epoch_validation["recall_at_confidence"],
            )
            if epoch_validation["map50_95"] > best_validation_map:
                best_validation_map = epoch_validation["map50_95"]
                best_epoch = epoch + 1
                if best_weights_path:
                    temporary_best = best_weights_path.with_name(best_weights_path.name + ".tmp")
                    torch.save(model.state_dict(), temporary_best)
                    os.replace(temporary_best, best_weights_path)
            if checkpoint_dir:
                temporary_last = checkpoint_dir / "last_model.pt.tmp"
                torch.save(model.state_dict(), temporary_last)
                os.replace(temporary_last, checkpoint_dir / "last_model.pt")
                training_report["latest_validation"] = epoch_validation
                training_report["best_epoch"] = best_epoch
                training_report["best_validation_map50_95"] = best_validation_map
                write_json_atomic(checkpoint_dir / "training_report.json", training_report)

            training_report["kpi_per_epoch"].append({
                "epoch": epoch + 1,
                "loss": avg_train_loss,
                "one2many_loss": avg_one2many_loss,
                "one2one_loss": avg_one2one_loss,
                "one2many_loss_giou": epoch_one2many_box_sum.item() / num_train_batches,
                "one2many_loss_obj": epoch_one2many_obj_sum.item() / num_train_batches,
                "one2many_loss_cls": epoch_one2many_cls_sum.item() / num_train_batches,
                "one2one_loss_giou": epoch_one2one_box_sum.item() / num_train_batches,
                "one2one_loss_obj": epoch_one2one_obj_sum.item() / num_train_batches,
                "one2one_loss_cls": epoch_one2one_cls_sum.item() / num_train_batches,
                "TP": epoch_tp if train_metrics else None,
                "FP": epoch_fp if train_metrics else None,
                "FN": epoch_fn if train_metrics else None,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "validation": epoch_validation,
            })

            progress_val = 15 + int(((epoch + 1) / num_epochs) * 65)
            report_ui_func(job_dao, job_id,
                           ui_status=f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_train_loss:.4f}",
                           progress_percent=progress_val, stage="training")

        # Export the checkpoint selected on held-out sequence mAP, not simply
        # the weights from the final epoch.
        if best_weights_path and best_weights_path.exists():
            model.load_state_dict(torch.load(best_weights_path, map_location=device))
            logger.info("Using best validation checkpoint from epoch %s", best_epoch)

        # Persist a usable checkpoint before evaluation, so an eval/render failure
        # does not discard a completed training run.
        checkpoint_dir = job_definition.get("checkpoint_dir")
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            weights_path = checkpoint_dir / "model.pt"
            temporary_weights = checkpoint_dir / "model.pt.tmp"
            torch.save(model.state_dict(), temporary_weights)
            os.replace(temporary_weights, weights_path)
        else:
            tmp_weights_dir = tempfile.mkdtemp()
            weights_path = Path(tmp_weights_dir) / "model.pt"
            torch.save(model.state_dict(), weights_path)

        model_metadata = dict(train_dataset.metadata)
        model_metadata.update({
            "backend": "custom",
            "weights": "model.pt",
            "pretrained_weights": pretrained_weights,
            "input_size": [model_h, model_w],
            "conf_thresh": 0.2,
            "iou_thresh": 0.45,
            "inference_mode": "one2one_nms_free",
            "best_epoch": best_epoch,
            "validation_map50_95": best_validation_map,
        })
        if checkpoint_dir:
            write_json_atomic(checkpoint_dir / "metadata.json", model_metadata)
            write_json_atomic(checkpoint_dir / "training_report.json", training_report)
            write_json_atomic(checkpoint_dir / "checkpoint_status.json", {
                "stage": "evaluation_pending", "job_id": job_id,
                "eval_limit": eval_limit,
            })
            logger.info("Training checkpoint saved before evaluation: %s", checkpoint_dir)

        # Release persistent train workers before starting evaluation workers.
        for parameter in model.parameters():
            parameter.grad = None
        del images, targets, predictions, one2one_predictions, loss, loss_dict
        del one2one_loss_dict, one2many_loss, one2one_loss, loss_inputs, one2one_loss_inputs
        del epoch_one2many_loss_sum, epoch_one2one_loss_sum
        del epoch_one2many_box_sum, epoch_one2many_obj_sum, epoch_one2many_cls_sum
        del epoch_one2one_box_sum, epoch_one2one_obj_sum, epoch_one2one_cls_sum
        del optimizer, criterion, scaler, batches, train_loader
        import gc
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        report_ui_func(job_dao, job_id, ui_status="Generating Per-Image Predictions...", progress_percent=80, stage="evaluating")
        model.eval()

        # Evaluate a deterministic spread of images; use --eval-limit 0 for all.
        # Final metrics and rendered examples come exclusively from held-out sequences.
        eval_dataset = val_dataset
        class_map = eval_dataset.class_map
        eval_count = len(eval_dataset) if eval_limit == 0 else min(len(eval_dataset), eval_limit)
        eval_indices = np.linspace(0, len(eval_dataset) - 1, eval_count, dtype=int).tolist()
        logger.info("Starting evaluation on %d/%d held-out images", eval_count, len(eval_dataset))
        evaluation, details = evaluate_detector(
            model, eval_dataset, eval_indices, confidence=0.2,
            num_classes=eval_dataset.num_classes, image_h=model_h,
            image_w=model_w, device=device, use_amp=use_amp,
            batch_size=batch_size, class_map=class_map, include_details=True,
        )
        training_report["per_image_predictions"] = details
        per_image = training_report["per_image_predictions"]
        training_report["evaluation"] = {
            **evaluation,
            "precision": evaluation["precision_at_confidence"],
            "recall": evaluation["recall_at_confidence"],
            "f1": evaluation["f1_at_confidence"],
            "iou_threshold": 0.5,
            "images": len(per_image),
            "total_images": len(eval_dataset),
            "sampled": len(per_image) < len(eval_dataset),
        }
        training_report["best_epoch"] = best_epoch
        training_report["best_validation_map50_95"] = best_validation_map

        # Select Top 5 / Bottom 5 for Rendering
        ranked = sorted(per_image, key=lambda e: (e["f1"], -(e["FP"] + e["FN"])), reverse=True)
        top_entries = ranked[:5]
        bottom_entries = list(reversed(ranked[max(5, len(ranked) - 5):]))

        report_ui_func(job_dao, job_id, ui_status="Rendering evaluation images...", progress_percent=83, stage="evaluating")

        eval_images = {}
        for prefix, entries in (("top", top_entries), ("bottom", bottom_entries)):
            report_key = f"{prefix}_images"
            training_report["evaluation"][report_key] = []
            for rank, entry in enumerate(entries, start=1):
                artifact = f"eval_{prefix}_{rank}.jpg"
                img_tensor, _ = eval_dataset[entry["image_index"]]
                eval_images[artifact] = render_eval_image(img_tensor, entry)
                summary = dict(entry)
                summary["artifact"] = artifact
                summary["rank"] = rank
                training_report["evaluation"][report_key].append(summary)

        report_ui_func(job_dao, job_id, ui_status="Saving weights and graphs...", progress_percent=85, stage="packaging")
        
        training_graph = get_loss_graph(epoch_losses)

        save_od_output_zip(
            job_id=job_id,
            weights_path=weights_path,
            metadata=model_metadata,
            training_report=training_report,
            training_graph=training_graph,
            eval_images=eval_images,
        )
        
        report_ui_func(job_dao, job_id, ui_status="Done.", progress_percent=100, stage="done", estimated_remaining_seconds=0)

        return {
            "result": "success",
            "ui_status": append_time_left_to_ui_status("Model download package is ready.", 0),
            "progress_percent": 100,
            "estimated_remaining_seconds": 0,
        }

    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error in run_od_pipeline: {e}")
        checkpoint_path = job_definition.get("checkpoint_dir")
        if checkpoint_path and (Path(checkpoint_path) / "model.pt").is_file():
            try:
                write_json_atomic(Path(checkpoint_path) / "checkpoint_status.json", {
                    "stage": "evaluation_or_packaging_failed",
                    "job_id": job_id,
                    "error": str(e),
                })
            except Exception:
                logger.exception("Could not update retained checkpoint status")
        report_ui_func(job_dao, job_id, ui_status="Error.", progress_percent=100, stage="error", extra={"error": str(e)})
        return {"result": "error", "error": str(e)}

    finally:
        if extract_tmp_dir and os.path.exists(extract_tmp_dir):
            shutil.rmtree(extract_tmp_dir)
        if tmp_weights_dir and os.path.exists(tmp_weights_dir):
            shutil.rmtree(tmp_weights_dir)



# ---- Command-line entry point ----
def report_console(_job_dao, job_id, **status):
    logger.info("[%s] %s%% %s: %s", job_id,
                status.get("progress_percent", 0), status.get("stage", ""),
                status.get("ui_status", ""))
    if status.get("extra"):
        logger.error("%s", status["extra"])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    input_source = parser.add_mutually_exclusive_group(required=True)
    input_source.add_argument("--input-dir", type=Path,
                              help="Mounted/extracted YOLO dataset folder (Kaggle input)")
    input_source.add_argument("--input-zip", type=Path,
                              help="YOLO dataset ZIP (extracted to temporary storage)")
    parser.add_argument("--output-dir", type=Path, default=Path("od_output"))
    parser.add_argument("--job-id", default="od_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    parser.add_argument("--configuration", type=Path, help="JSON file with original pipeline configuration")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--obj-focal-gamma", type=float, default=2.0,
                        help="Focal gamma for objectness hard negatives; 0 restores BCE")
    parser.add_argument("--one2many-loss-weight", type=float,
                        help="Weight for the one-to-many loss alongside the NMS-free one-to-one loss")
    parser.add_argument("--pretrained-weights", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="Training device (default: auto-select CUDA when available)")
    parser.add_argument("--progress", choices=("auto", "on", "off"), default="auto",
                        help="tqdm progress; auto enables in a TTY and disables in screen/tmux")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--log-every-images", type=int, default=500,
                        help="In non-interactive runs, log training progress every N images")
    parser.add_argument("--cpu-threads", type=int, default=0, help="CPU compute threads; 0 selects automatically")
    parser.add_argument("--eval-limit", type=int, default=2000,
                        help="Final validation images (0 evaluates all; default: 2000 spread across held-out sequences)")
    parser.add_argument("--preview-every-images", type=int, default=2000,
                        help="Save validation previews every N training images; 0 disables previews")
    parser.add_argument("--preview-images", type=int, default=4,
                        help="Number of fixed images to render at each preview epoch")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable/disable mixed precision (default: enabled on CUDA)")
    parser.add_argument("--train-metrics", action="store_true",
                        help="Compute expensive per-epoch detection metrics during training")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.input_dir is not None and not args.input_dir.is_dir():
        parser.error(f"Input dataset directory does not exist: {args.input_dir}")
    if args.input_zip is not None and not args.input_zip.is_file():
        parser.error(f"Input ZIP does not exist: {args.input_zip}")
    if not args.job_id or args.job_id in (".", "..") or any(c not in
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.job_id):
        parser.error("job-id must contain only ASCII letters, digits, underscores or hyphens")
    config = {}
    if args.configuration:
        with args.configuration.open(encoding="utf-8") as handle:
            config = json.load(handle)
        if not isinstance(config, dict):
            parser.error("configuration must be a JSON object")
    for key, value in (("epochs", args.epochs), ("batch_size", args.batch_size),
                       ("learning_rate", args.learning_rate)):
        if value is not None:
            config[key] = value
    config.update(device=args.device, progress=args.progress, workers=args.workers,
                  cpu_threads=args.cpu_threads, eval_limit=args.eval_limit,
                  log_every_images=args.log_every_images,
                  obj_focal_gamma=args.obj_focal_gamma)
    config.update(preview_every_images=args.preview_every_images,
                  preview_images=args.preview_images)
    if args.one2many_loss_weight is not None:
        config["one2many_loss_weight"] = args.one2many_loss_weight
    if args.amp is not None:
        config["amp"] = args.amp
    if args.train_metrics:
        config["train_metrics"] = True
    for key, default in (("epochs", 40), ("batch_size", 4)):
        value = config.get(key, default)
        if type(value) is not int or value <= 0:
            parser.error(f"{key} must be a positive integer")
    lr = config.get("learning_rate", 5e-4)
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not 0 < lr < float("inf"):
        parser.error("learning_rate must be a finite positive number")
    if not math.isfinite(args.obj_focal_gamma) or args.obj_focal_gamma < 0:
        parser.error("obj-focal-gamma must be finite and non-negative")
    if args.one2many_loss_weight is not None and (
            not math.isfinite(args.one2many_loss_weight) or args.one2many_loss_weight < 0):
        parser.error("one2many-loss-weight must be finite and non-negative")
    if (args.workers < 0 or args.eval_limit < 0 or args.cpu_threads < 0
            or args.log_every_images <= 0 or args.preview_every_images < 0
            or args.preview_images <= 0):
        parser.error("workers/eval-limit/cpu-threads must be non-negative; log and preview values must be positive")
    if args.pretrained_weights and not args.pretrained_weights.is_file():
        parser.error(f"Pretrained weights do not exist: {args.pretrained_weights}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_dir = output_dir / args.job_id
    result_zip = output_dir / f"{args.job_id}.zip"
    checkpoint_dir = output_dir / f"{args.job_id}_checkpoint"
    preview_dir = output_dir / f"{args.job_id}_preview"
    if result_dir.exists() or result_zip.exists() or checkpoint_dir.exists() or preview_dir.exists():
        parser.error(f"Output/checkpoint already exists for job {args.job_id}; choose a different job-id")

    job_definition = {"configuration": json.dumps(config),
                      "checkpoint_dir": str(checkpoint_dir),
                      "preview_dir": str(preview_dir)}
    if args.input_dir is not None:
        job_definition["input_dir"] = str(args.input_dir.resolve())
    else:
        job_definition["zip_path"] = str(args.input_zip.resolve())
    if args.pretrained_weights:
        job_definition["pretrained_weights"] = str(args.pretrained_weights.resolve())

    # Use an isolated packaging directory, then publish the original ZIP and
    # its extracted contents together after the pipeline finishes successfully.
    with tempfile.TemporaryDirectory(prefix=".od_", dir=output_dir) as staging:
        previous = os.environ.get("MODEL_ZIP_FOLDER")
        os.environ["MODEL_ZIP_FOLDER"] = staging
        try:
            result = run_od_pipeline(args.job_id, job_definition, None, report_console)
        finally:
            if previous is None:
                os.environ.pop("MODEL_ZIP_FOLDER", None)
            else:
                os.environ["MODEL_ZIP_FOLDER"] = previous
        if result.get("result") != "success":
            logger.error("Pipeline failed: %s", result.get("error", result))
            if (checkpoint_dir / "model.pt").is_file():
                logger.error("Trained checkpoint was retained at: %s", checkpoint_dir)
            return 1
        archive = Path(staging) / f"{args.job_id}.zip"
        extracted = Path(staging) / "contents"
        with zipfile.ZipFile(archive) as package:
            required = {"model.pt", "metadata.json", "training_report.json", "training_graph.jpg"}
            if not required.issubset(package.namelist()):
                raise RuntimeError("Pipeline output is missing required artifacts")
            package.extractall(extracted)
        extracted.rename(result_dir)
        archive.rename(result_zip)
    shutil.rmtree(checkpoint_dir)
    print(f"Output folder: {result_dir}")
    print(f"Output ZIP:    {result_zip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
