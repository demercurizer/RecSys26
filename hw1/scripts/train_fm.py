import argparse
import json
import random
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.fm import FM


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data/processed/train.parquet"
ITEMS_PATH = ROOT / "data/processed/items_train.parquet"

FEATURE_COLUMNS = ["user_id", "item_id", "category", "store"]


def prepare_features(train, items):
    items = (
        items
        .unique("item_id", keep="first")
        .with_columns(
            pl.col("categories").list.last().fill_null("__UNK__").alias("category"),
            pl.col("store").fill_null("__UNK__"),
        )
    )

    df = (
        train
        .join(
            items.select("item_id", "category", "store"),
            on="item_id",
            how="left",
        )
        .with_columns(
            pl.col("category").fill_null("__UNK__"),
            pl.col("store").fill_null("__UNK__"),
        )
    )

    mappings = {}
    offsets = {}
    offset = 0

    for column in FEATURE_COLUMNS:
        values = df[column].unique().sort().to_list()
        mappings[column] = {value: i for i, value in enumerate(values)}
        offsets[column] = offset
        offset += len(values)

    x = np.empty((df.height, len(FEATURE_COLUMNS)), dtype=np.int64)

    for j, column in enumerate(FEATURE_COLUMNS):
        mapping = mappings[column]
        base = offsets[column]

        x[:, j] = np.fromiter(
            (base + mapping[value] for value in df[column]),
            dtype=np.int64,
            count=df.height,
        )

    y = df["rating"].to_numpy().astype(np.float32)

    return x, y, mappings, offsets, offset


def save_checkpoint(model, output, epoch):
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
        },
        output / f"checkpoint_epoch_{epoch:02d}.pt",
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="artifacts/fm/default")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train = pl.read_parquet(TRAIN_PATH)
    items = pl.read_parquet(ITEMS_PATH)

    x, y, mappings, offsets, n_features = prepare_features(train, items)

    dataset = TensorDataset(
        torch.from_numpy(x),
        torch.from_numpy(y),
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = FM(
        n_features=n_features,
        dim=args.dim,
    ).to(device)

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name == "bias":
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": decay_params,
                "weight_decay": args.weight_decay,
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
            },
        ],
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        fused=device.type == "cuda",
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.05,
    )

    criterion = torch.nn.MSELoss()

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
    )

    output = Path(args.output)

    if not output.is_absolute():
        output = ROOT / output

    output.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "n_features": n_features,
            "dim": args.dim,
            "feature_columns": FEATURE_COLUMNS,
            "mappings": mappings,
            "offsets": offsets,
        },
        output / "preprocessing.pt",
    )

    print(f"device={device}")
    print(f"interactions={len(dataset)}")
    print(f"n_features={n_features}")
    print(f"features={FEATURE_COLUMNS}")
    print(f"dim={args.dim}")
    print(f"batch_size={args.batch_size}")
    print(f"lr={args.lr}")
    print(f"weight_decay={args.weight_decay}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                pred = model(xb)
                loss = criterion(pred, yb)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * xb.size(0)

        train_mse = total_loss / len(dataset)

        scheduler.step()

        print(
            f"epoch={epoch}/{args.epochs} "
            f"mse={train_mse:.6f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}",
            flush=True,
        )

        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            save_checkpoint(model, output, epoch)

    config = vars(args).copy()
    config["device"] = str(device)
    config["train_path"] = str(TRAIN_PATH)
    config["items_path"] = str(ITEMS_PATH)
    config["features"] = FEATURE_COLUMNS

    with open(output / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"saved_to={output}")


if __name__ == "__main__":
    main()