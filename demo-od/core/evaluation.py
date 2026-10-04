"""Leakage-resistant data splitting and COCO-style box metrics."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import re

import numpy as np


_SEQUENCE_FRAME = re.compile(r"^(?P<group>.+?)(?:_img|_frame)_?\d+$", re.IGNORECASE)


def grouped_train_val_split(dataset, val_fraction: float = 0.2, seed: int = 42,
                            trials: int = 4096) -> tuple[list[str], list[str], dict]:
    """Split by video/sequence when filenames expose one; otherwise split images.

    For UA-DETRAC, names such as ``MVI_20011_img00001.jpg`` are grouped by
    ``MVI_20011`` so neighboring frames never cross the train/validation split.
    Candidate sequence subsets are searched deterministically to balance image
    count and per-class object counts while keeping each class represented.
    """
    if not 0.05 <= val_fraction <= 0.5:
        raise ValueError("val_fraction must be between 0.05 and 0.5")

    groups: dict[str, list[str]] = defaultdict(list)
    class_counts: dict[str, Counter] = defaultdict(Counter)
    group_pattern_matches = 0
    for image_name in dataset.image_files:
        stem = image_name.rsplit(".", 1)[0]
        match = _SEQUENCE_FRAME.match(stem)
        group = match.group("group") if match else f"__image__/{image_name}"
        group_pattern_matches += match is not None
        groups[group].append(image_name)
        label_path = Path(dataset.label_dir) / (stem + ".txt")
        try:
            with label_path.open(encoding="utf-8") as handle:
                for line in handle:
                    parts = line.split()
                    if len(parts) == 5:
                        class_id = int(float(parts[0]))
                        if class_id in dataset.class_map:
                            class_counts[group][class_id] += 1
        except FileNotFoundError:
            continue

    group_names = sorted(groups)
    if len(group_names) < 2:
        raise ValueError("Need at least two independent groups/images for validation")
    n_val_groups = max(1, min(len(group_names) - 1, round(len(group_names) * val_fraction)))
    rng = np.random.default_rng(seed)
    classes = sorted(dataset.class_map)
    total_images = len(dataset.image_files)
    total_objects = Counter()
    for counts in class_counts.values():
        total_objects.update(counts)

    best_groups: set[str] | None = None
    best_score = float("inf")
    for _ in range(max(1, trials)):
        chosen = set(rng.choice(group_names, size=n_val_groups, replace=False).tolist())
        n_images = sum(len(groups[name]) for name in chosen)
        val_objects = Counter()
        for name in chosen:
            val_objects.update(class_counts[name])
        image_error = abs(n_images / total_images - val_fraction)
        class_error = 0.0
        missing_classes = 0
        for class_id in classes:
            total = total_objects[class_id]
            if not total:
                continue
            ratio = val_objects[class_id] / total
            class_error += abs(ratio - val_fraction)
            missing_classes += int(val_objects[class_id] == 0)
            missing_classes += int(val_objects[class_id] == total)
        score = 3.0 * image_error + class_error + 2.0 * missing_classes
        if score < best_score:
            best_score, best_groups = score, chosen

    assert best_groups is not None
    val_names = sorted(name for group in best_groups for name in groups[group])
    train_names = sorted(name for group in group_names if group not in best_groups
                         for name in groups[group])
    val_objects = Counter()
    train_objects = Counter()
    for group in best_groups:
        val_objects.update(class_counts[group])
    for group in set(group_names) - best_groups:
        train_objects.update(class_counts[group])
    manifest = {
        "method": "deterministic_group_stratified",
        "seed": seed,
        "requested_validation_fraction": val_fraction,
        "actual_validation_fraction": len(val_names) / total_images,
        "group_pattern": "filename sequence prefix (_imgNNN or _frameNNN); image-level fallback",
        "grouped_filename_count": group_pattern_matches,
        "train_groups": sorted(set(group_names) - best_groups),
        "validation_groups": sorted(best_groups),
        "train_images": len(train_names),
        "validation_images": len(val_names),
        "train_class_objects": {str(k): train_objects[k] for k in classes},
        "validation_class_objects": {str(k): val_objects[k] for k in classes},
    }
    return train_names, val_names, manifest


def _box_iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    left_top = np.maximum(box[:2], boxes[:, :2])
    right_bottom = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.clip(right_bottom - left_top, 0, None).prod(axis=1)
    area_a = np.clip(box[2:] - box[:2], 0, None).prod()
    area_b = np.clip(boxes[:, 2:] - boxes[:, :2], 0, None).prod(axis=1)
    return intersection / np.maximum(area_a + area_b - intersection, 1e-9)


def detection_map(predictions: list[dict], ground_truth: list[dict], class_ids,
                  iou_thresholds=None) -> dict:
    """Compute 101-point interpolated AP at IoU .50:.05:.95 (COCO-style).

    Inputs are per-detection/per-object dictionaries containing image_index,
    class_id, score (predictions only), and normalized xyxy box coordinates.
    Each image should contribute at most 300 predictions, ranked by confidence.
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 0.951, 0.05)
    recall_samples = np.linspace(0.0, 1.0, 101)
    per_class = {}
    threshold_aps = []

    for class_id in class_ids:
        gt_by_image: dict[int, list[np.ndarray]] = defaultdict(list)
        for item in ground_truth:
            if int(item["class_id"]) == int(class_id):
                gt_by_image[int(item["image_index"])].append(np.asarray(item["box"], dtype=np.float32))
        gt_count = sum(map(len, gt_by_image.values()))
        if gt_count == 0:
            continue
        class_predictions = [item for item in predictions
                             if int(item["class_id"]) == int(class_id)]
        class_predictions.sort(key=lambda item: float(item["score"]), reverse=True)
        class_aps = []

        for threshold in iou_thresholds:
            matched: dict[int, set[int]] = defaultdict(set)
            tp = np.zeros(len(class_predictions), dtype=np.float64)
            fp = np.zeros(len(class_predictions), dtype=np.float64)
            for index, prediction in enumerate(class_predictions):
                image_index = int(prediction["image_index"])
                gt_boxes = gt_by_image.get(image_index, [])
                if not gt_boxes:
                    fp[index] = 1.0
                    continue
                gt_array = np.stack(gt_boxes)
                ious = _box_iou_one_to_many(np.asarray(prediction["box"], dtype=np.float32), gt_array)
                for gt_index in np.argsort(ious)[::-1]:
                    if ious[gt_index] < threshold:
                        break
                    if int(gt_index) not in matched[image_index]:
                        matched[image_index].add(int(gt_index))
                        tp[index] = 1.0
                        break
                if tp[index] == 0:
                    fp[index] = 1.0

            if len(tp):
                cumulative_tp = np.cumsum(tp)
                cumulative_fp = np.cumsum(fp)
                recalls = cumulative_tp / gt_count
                precisions = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)
                precisions = np.maximum.accumulate(precisions[::-1])[::-1]
                ap = float(np.mean([
                    precisions[recalls >= recall].max() if np.any(recalls >= recall) else 0.0
                    for recall in recall_samples
                ]))
            else:
                ap = 0.0
            class_aps.append(ap)
            threshold_aps.append((float(threshold), int(class_id), ap))

        per_class[str(class_id)] = {
            "ap50": class_aps[0],
            "ap75": class_aps[5] if len(class_aps) > 5 else None,
            "ap50_95": float(np.mean(class_aps)),
            "ground_truth_objects": gt_count,
            "predictions": len(class_predictions),
        }

    if not per_class:
        return {"map50": 0.0, "map75": 0.0, "map50_95": 0.0, "per_class": {}}
    class_values = list(per_class.values())
    return {
        "map50": float(np.mean([item["ap50"] for item in class_values])),
        "map75": float(np.mean([item["ap75"] for item in class_values if item["ap75"] is not None])),
        "map50_95": float(np.mean([item["ap50_95"] for item in class_values])),
        "per_class": per_class,
    }
