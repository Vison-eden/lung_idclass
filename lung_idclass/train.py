"""Train one primary neural component for one study seed."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .common import TASKS, LABELS, read_manifest, build_model, write_predictions, load_config
from .image_data import MultiModalDataset, collate_batch
from .semantic import fit_semantic
from .models import set_partial_encoder_trainable
from .training_core import (
    seed_everything, calculate_class_weights, compute_metrics,
    run_ct_epoch, run_fusion_epoch, build_optimizer,
    load_image_encoder, load_semantic_encoder,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--model", choices=["ct", "semantic", "fusion"], required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, choices=[3407, 2025, 2026], default=3407)
    parser.add_argument("--ct-init")
    parser.add_argument("--semantic-init")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, help="Override for a short smoke run.")
    args = parser.parse_args()
    config = load_config(args.task)
    settings = config["models"][args.model].copy()
    if args.epochs is not None:
        if args.epochs < 1:
            parser.error("--epochs must be positive")
        settings["epochs"] = args.epochs
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "best.pt").exists():
        raise FileExistsError("Output already contains best.pt; choose a new output directory.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Use --device cpu for a smoke run.")
    seed_everything(args.seed)
    frame = read_manifest(args.manifest, args.task, args.model, training=True)
    train = frame.loc[frame.split == "train"].reset_index(drop=True)
    val = frame.loc[frame.split == "val"].reset_index(drop=True)
    schema, train_x, val_x = None, None, None
    if args.model != "ct":
        train_x, val_x, _, _, _ = fit_semantic(train, val, TASKS[args.task], output)
        schema = json.loads((output / "semantic_token_schema.json").read_text(encoding="utf-8"))
    dataset_mode = {"ct": "ct_only", "semantic": "semantic_only", "fusion": "fusion_gated"}[args.model]
    train_data = MultiModalDataset(train, train.label, train_x, dataset_mode, 5, 128, args.model != "semantic")
    val_data = MultiModalDataset(val, val.label, val_x, dataset_mode, 5, 128, False)
    loader_kwargs = dict(batch_size=settings["batch_size"], num_workers=args.workers,
                         pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
                         collate_fn=collate_batch)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_data, shuffle=False, **loader_kwargs)
    model = build_model(args.task, args.model, schema).to(device)
    if args.model == "fusion":
        if not args.ct_init or not args.semantic_init:
            parser.error("fusion requires --ct-init and --semantic-init from the same task and seed")
        for path, branch in [(args.ct_init, "ct"), (args.semantic_init, "semantic")]:
            initial = torch.load(path, map_location="cpu", weights_only=False)
            if initial.get("public_task") != args.task or initial.get("seed") != args.seed or initial.get("public_model") != branch:
                raise ValueError("Initial checkpoint task, seed or component mismatch.")
            if branch == "semantic":
                if initial.get("semantic_schema") != schema:
                    raise ValueError("Semantic feature/category schema differs from the fitted branch.")
            del initial
        load_image_encoder(model, args.ct_init)
        load_semantic_encoder(model, args.semantic_init)
        model.set_encoder_trainable(False)
    runtime = SimpleNamespace(
        mode="fusion_bica" if args.model == "fusion" else "semantic_token_only",
        lr=settings["learning_rate"], weight_decay=0.0001, encoder_lr_scale=0.2,
        aux_weight=0.2 if args.model == "fusion" else 0.0,
        gate_weight=settings.get("gate_weight", 0.0),
    )
    weights = calculate_class_weights(train.label.to_numpy(), len(LABELS[args.task])).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    def make_optim():
        optimizer = build_optimizer(model, runtime)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5,
            patience=8 if args.model == "ct" else 7, min_lr=1e-6)
        return optimizer, scheduler

    optimizer, scheduler = make_optim()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    def epoch(loader, training):
        with torch.set_grad_enabled(training):
            if args.model == "ct":
                loss, y, p, _ = run_ct_epoch(model, loader, criterion, optimizer, device, scaler, training)
                return loss, y, p
            result = run_fusion_epoch(model, loader, criterion, optimizer, scaler, device, runtime, training)
            return result["loss"], result["true_labels"], result["probabilities"]

    best_auc, waiting, history = -np.inf, 0, []
    for number in range(1, settings["epochs"] + 1):
        if args.model == "fusion" and args.task == "identification" and number == 9:
            set_partial_encoder_trainable(model)
            optimizer, scheduler = make_optim()
        train_loss, _, _ = epoch(train_loader, True)
        val_loss, y, probabilities = epoch(val_loader, False)
        metrics = compute_metrics(TASKS[args.task], y, probabilities)
        auc = metrics["auc" if args.task == "identification" else "macro_auc_ovr"]
        scheduler.step(auc)
        record = {"epoch": number, "train_loss": train_loss, "val_loss": val_loss, "val_auc": auc}
        history.append(record)
        print(json.dumps(record), flush=True)
        if auc > best_auc + 1e-6:
            best_auc, waiting = auc, 0
            checkpoint = dict(
                model_state=model.state_dict(), public_task=args.task, public_model=args.model,
                task=TASKS[args.task], seed=args.seed, best_epoch=number,
                best_selection_metric=auc, semantic_schema=schema,
                settings=settings, labels=LABELS[args.task], scale_mm=5, max_dim=128,
            )
            torch.save(checkpoint, output / "best.pt")
            write_predictions(val, probabilities, output / "validation.csv")
            (output / "validation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        else:
            waiting += 1
        if waiting >= settings["patience"]:
            break
    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    (output / "run_config.json").write_text(json.dumps(
        dict(task=args.task, model=args.model, seed=args.seed, settings=settings,
             device=str(device), workers=args.workers, labels=LABELS[args.task]), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
