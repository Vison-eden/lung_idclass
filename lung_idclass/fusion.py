"""Aligned seed averaging and the two final probability combinations."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

SEEDS = (3407, 2025, 2026)


def validate_probabilities(probabilities, classes):
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != classes or not len(probabilities):
        raise ValueError(f"Expected a nonempty N x {classes} probability array")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be finite and in [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-6):
        raise ValueError("Each probability row must sum to one; export sufficient decimal precision")
    return probabilities


def read_predictions(path, classes):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = [f"p{i}" for i in range(classes)]
        required = ["case_id"] + columns
        if not reader.fieldnames or not set(required).issubset(reader.fieldnames):
            raise ValueError(f"{path}: required columns are {required}")
        unexpected = {f"p{i}" for i in range(classes, 10)} & set(reader.fieldnames)
        if unexpected:
            raise ValueError(f"{path}: unexpected probability columns {sorted(unexpected)}")
        rows = list(reader)
    ids = [row["case_id"] for row in rows]
    if not ids or any(value is None or not value.strip() for value in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{path}: case_id values must be nonempty and unique")
    probabilities = validate_probabilities([[float(row[name]) for name in columns] for row in rows], classes)
    return ids, probabilities


def align_probabilities(reference_ids, ids, probabilities):
    if len(reference_ids) != len(ids) or set(reference_ids) != set(ids):
        raise ValueError("Components must contain exactly the same case_id set")
    lookup = {case_id: index for index, case_id in enumerate(ids)}
    return probabilities[[lookup[case_id] for case_id in reference_ids]]


def average_predictions(paths, classes, already_averaged=False):
    expected = 1 if already_averaged else len(SEEDS)
    if len(paths) != expected:
        raise ValueError(f"Expected {expected} prediction file(s); use the averaged flag for one seed-averaged file")
    if len({str(Path(path).resolve()) for path in paths}) != len(paths):
        raise ValueError("Supply each seed prediction file only once")
    reference_ids, first = read_predictions(paths[0], classes)
    aligned = [first]
    for path in paths[1:]:
        ids, values = read_predictions(path, classes)
        aligned.append(align_probabilities(reference_ids, ids, values))
    return reference_ids, np.mean(np.stack(aligned), axis=0)


def final_probabilities(task, left, right, weight=None):
    if task not in ("identification", "invasiveness"):
        raise ValueError("task must be identification or invasiveness")
    classes = 2 if task == "identification" else 3
    left = validate_probabilities(left, classes)
    right = validate_probabilities(right, classes)
    if left.shape != right.shape:
        raise ValueError("Component shapes differ")
    if weight is None:
        weight = 0.75 if task == "identification" else 0.5
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("weight must be in [0, 1]")
    return float(weight) * left + (1.0 - float(weight)) * right


def choose_identification_blend(labels, base, clinical):
    """Original validation grid: AUC, then macro-F1 and balanced accuracy."""
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, roc_curve

    labels = np.asarray(labels)
    if len(np.unique(labels)) != 2 or not np.isin(labels, [0, 1]).all():
        raise ValueError("Validation labels must contain both 0 and 1")
    if len(labels) != len(base) or len(labels) != len(clinical):
        raise ValueError("Validation labels and probabilities have different lengths")
    rows = []
    for weight in np.linspace(0.0, 1.0, 21):
        probability = final_probabilities("identification", base, clinical, weight)
        fpr, tpr, thresholds = roc_curve(labels, probability[:, 1])
        finite = np.isfinite(thresholds)
        threshold = float(thresholds[finite][np.argmax((tpr - fpr)[finite])]) if finite.any() else 0.5
        predicted = (probability[:, 1] >= threshold).astype(int)
        rows.append({"weight": float(weight), "threshold": threshold,
                     "auc": float(roc_auc_score(labels, probability[:, 1])),
                     "f1_macro": float(f1_score(labels, predicted, average="macro", zero_division=0)),
                     "balanced_accuracy": float(balanced_accuracy_score(labels, predicted))})
    rows.sort(key=lambda row: (row["auc"], row["f1_macro"], row["balanced_accuracy"]), reverse=True)
    return rows[0], rows


def read_validation_labels(path, case_ids):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"case_id", "label"}.issubset(reader.fieldnames):
            raise ValueError("Validation labels require case_id and label columns")
        rows = list(reader)
    ids = [row["case_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate case_id in validation labels")
    if "split" in (reader.fieldnames or []):
        if any(row["split"].strip().lower() not in {"val", "validation"} for row in rows):
            raise ValueError("Weight/threshold selection accepts validation rows only")
    labels = np.asarray([float(row["label"]) for row in rows]).reshape(-1, 1)
    return align_probabilities(case_ids, ids, labels).ravel()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=["identification", "invasiveness"])
    parser.add_argument("--left", nargs="+", required=True, help="CT-semantic for identification; CT-only for invasiveness")
    parser.add_argument("--right", nargs="+", required=True, help="Clinical for identification; BiCA for invasiveness")
    parser.add_argument("--left-averaged", action="store_true")
    parser.add_argument("--right-averaged", action="store_true")
    parser.add_argument("--weight", type=float, help="Left weight; default .75 for identification or .50 for invasiveness")
    parser.add_argument("--threshold", type=float, default=0.7400414428163694, help="Identification threshold; default is the stored study value")
    parser.add_argument("--select-validation", help="Validation-only case_id,label CSV to select the identification blend and threshold")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    classes = 2 if args.task == "identification" else 3
    ids, left = average_predictions(args.left, classes, args.left_averaged)
    right_ids, right = average_predictions(args.right, classes, args.right_averaged)
    right = align_probabilities(ids, right_ids, right)
    weight = args.weight if args.weight is not None else (0.75 if classes == 2 else 0.5)
    threshold = args.threshold
    selection = None
    if args.select_validation:
        if args.task != "identification":
            parser.error("The invasiveness combination has fixed equal weights")
        best, grid = choose_identification_blend(read_validation_labels(args.select_validation, ids), left, right)
        weight, threshold = best["weight"], best["threshold"]
        selection = {"selected": best, "grid": grid, "selection_partition": "validation"}
    if classes == 2 and (not np.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("Identification threshold must be in [0, 1]")
    probability = final_probabilities(args.task, left, right, weight)
    prediction = (probability[:, 1] >= threshold).astype(int) if classes == 2 else probability.argmax(axis=1)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case_id"] + [f"p{i}" for i in range(classes)] + ["prediction"])
        for case_id, values, predicted in zip(ids, probability, prediction):
            writer.writerow([case_id] + values.tolist() + [int(predicted)])
    metadata = {"task": args.task, "left_weight": weight, "right_weight": 1.0 - weight,
                "threshold": threshold if classes == 2 else None,
                "class_order": ["non_spectrum", "adenocarcinoma_spectrum"] if classes == 2 else ["AAH_AIS", "MIA", "IAC"]}
    if selection:
        metadata["validation_selection"] = selection
    destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(ids)} predictions to {destination}")


if __name__ == "__main__":
    main()
