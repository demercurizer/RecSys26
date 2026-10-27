import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from models.als import ALS
from models.ease import EASE
from models.signal_als import SignalALS


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data/processed/train.parquet"


def save_common(model, output):
    np.save(output / "users.npy", np.asarray(model.users, dtype=object), allow_pickle=True)
    np.save(output / "items.npy", np.asarray(model.items, dtype=object), allow_pickle=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["als", "signal_als", "ease"], required=True)
    parser.add_argument("--factors", type=int, default=32)
    parser.add_argument("--reg", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--tau-days", type=float, default=365.0)
    parser.add_argument("--rating-threshold", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    train = pl.read_parquet(TRAIN_PATH)

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)

    print(f"model={args.model}", flush=True)
    print(f"train_interactions={train.height}", flush=True)

    if args.model == "als":
        model = ALS(factors=args.factors, reg=args.reg, epochs=args.epochs, seed=args.seed)
        target = (train["rating"].to_numpy().astype(np.float32) - 3.0) / 2.0
        model.fit(train, target=target)

        np.save(output / "user_factors.npy", model.user_factors)
        np.save(output / "item_factors.npy", model.item_factors)

    elif args.model == "signal_als":
        model = SignalALS(factors=args.factors, reg=args.reg, epochs=args.epochs, beta=args.beta, tau_days=args.tau_days, seed=args.seed)
        model.fit(train)

        np.save(output / "user_factors.npy", model.user_factors)
        np.save(output / "item_factors.npy", model.item_factors)

    else:
        model = EASE(reg=args.reg, rating_threshold=args.rating_threshold)
        model.fit(train)

        np.save(output / "B.npy", model.B)

    save_common(model, output)

    config = vars(args).copy()
    config["train_path"] = str(TRAIN_PATH)

    with open(output / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"saved_to={output}", flush=True)


if __name__ == "__main__":
    main()