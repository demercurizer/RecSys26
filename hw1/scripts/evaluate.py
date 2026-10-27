import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from scipy.sparse import csr_matrix

from models.fm import FM
from models.item_knn import ItemKNN


ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data/processed/train.parquet"
VAL_PATH = ROOT / "data/processed/val.parquet"
ITEMS_PATH = ROOT / "data/processed/items_train.parquet"

ALS_PATH = ROOT / "artifacts/als/d32_r01"
SIGNAL_ALS_PATH = ROOT / "artifacts/signal_als/d32_r01_b05_t365"
EASE_PATH = ROOT / "artifacts/ease/r300"
FM_PATH = ROOT / "artifacts/fm/clean_d32"

OUTPUT_PATH = ROOT / "artifacts/evaluation"

K = 10


def prepare_data():
    train = pl.read_parquet(TRAIN_PATH)
    val = pl.read_parquet(VAL_PATH)

    train_users = train.select("user_id").unique()
    train_items = train.select("item_id").unique()

    val_warm = val.join(train_users, on="user_id", how="semi").join(train_items, on="item_id", how="semi")
    val_relevant = val_warm.filter(pl.col("rating") >= 4)

    targets_df = val_relevant.group_by("user_id").agg(pl.col("item_id").alias("relevant_items"))
    history_df = train.group_by("user_id").agg(pl.col("item_id").alias("seen_items"), pl.len().alias("history_len"))
    popularity_df = train.group_by("item_id").len().rename({"len": "popularity"})

    targets = {row["user_id"]: set(row["relevant_items"]) for row in targets_df.iter_rows(named=True)}
    seen = {row["user_id"]: set(row["seen_items"]) for row in history_df.iter_rows(named=True)}
    history_len = {row["user_id"]: row["history_len"] for row in history_df.iter_rows(named=True)}
    popularity = {row["item_id"]: row["popularity"] for row in popularity_df.iter_rows(named=True)}

    eval_users = targets_df["user_id"].to_list()
    candidate_items = train_items["item_id"].to_list()

    return train, eval_users, targets, seen, history_len, popularity, candidate_items


def metrics_at_k(recommended, relevant, popularity, total_interactions, k=10):
    recommended = recommended[:k]
    hits = [item in relevant for item in recommended]
    n_hits = sum(hits)

    recall = n_hits / len(relevant)
    precision = n_hits / k
    hit_rate = float(n_hits > 0)

    dcg = sum(hit / np.log2(rank + 2) for rank, hit in enumerate(hits))
    idcg = sum(1 / np.log2(rank + 2) for rank in range(min(len(relevant), k)))
    ndcg = dcg / idcg if idcg else 0.0

    relevant_ranks = [rank + 1 for rank, hit in enumerate(hits) if hit]
    mrr = 1 / relevant_ranks[0] if relevant_ranks else 0.0

    avg_popularity = np.mean([popularity[item] for item in recommended]) if recommended else 0.0
    novelty = np.mean([-np.log2(popularity[item] / total_interactions) for item in recommended]) if recommended else 0.0

    return recall, precision, ndcg, hit_rate, mrr, avg_popularity, novelty


def evaluate(name, recommendations, eval_users, targets, history_len, popularity, candidate_items, total_interactions):
    metric_rows = []
    recommendation_rows = []
    recommended_items = set()

    for user_id in eval_users:
        recs = recommendations[user_id][:K]
        relevant = targets[user_id]

        recall, precision, ndcg, hit_rate, mrr, avg_popularity, novelty = metrics_at_k(
            recs, relevant, popularity, total_interactions, K
        )

        recommended_items.update(recs)

        metric_rows.append({
            "model": name,
            "user_id": user_id,
            "history_len": history_len[user_id],
            "target_len": len(relevant),
            "recall@10": recall,
            "precision@10": precision,
            "ndcg@10": ndcg,
            "hit_rate@10": hit_rate,
            "mrr@10": mrr,
            "avg_popularity@10": avg_popularity,
            "novelty@10": novelty,
        })

        recommendation_rows.append({
            "model": name,
            "user_id": user_id,
            "recommended_items": recs,
        })

    metrics_df = pl.DataFrame(metric_rows)
    recommendations_df = pl.DataFrame(recommendation_rows)

    summary = {
        "model": name,
        "ndcg@10": metrics_df["ndcg@10"].mean(),
        "recall@10": metrics_df["recall@10"].mean(),
        "precision@10": metrics_df["precision@10"].mean(),
        "hit_rate@10": metrics_df["hit_rate@10"].mean(),
        "mrr@10": metrics_df["mrr@10"].mean(),
        "coverage@10": len(recommended_items) / len(candidate_items),
        "novelty@10": metrics_df["novelty@10"].mean(),
        "avg_popularity@10": metrics_df["avg_popularity@10"].mean(),
    }

    print(
        f"{name}: "
        f"NDCG@10={summary['ndcg@10']:.5f} "
        f"Recall@10={summary['recall@10']:.5f} "
        f"Coverage@10={summary['coverage@10']:.5f}",
        flush=True,
    )

    return summary, metrics_df, recommendations_df


def topk_from_scores(scores, items, seen_batch, item_to_idx, k=10):
    for row, user_seen in enumerate(seen_batch):
        indices = [item_to_idx[item] for item in user_seen if item in item_to_idx]
        if indices:
            scores[row, indices] = -np.inf

    top = np.argpartition(scores, -k, axis=1)[:, -k:]
    top_scores = np.take_along_axis(scores, top, axis=1)
    order = np.argsort(top_scores, axis=1)[:, ::-1]
    top = np.take_along_axis(top, order, axis=1)

    return [[items[i] for i in row] for row in top]


def recommend_most_popular(train, eval_users, seen):
    popular_items = (
        train
        .filter(pl.col("rating") >= 4)
        .group_by("item_id")
        .len()
        .sort("len", descending=True)["item_id"]
        .to_list()
    )

    return {user_id: [item for item in popular_items if item not in seen[user_id]][:K] for user_id in eval_users}


def recommend_item_knn(train, eval_users):
    model = ItemKNN(n_neighbors=100, rating_threshold=4).fit(train)
    return {user_id: model.recommend(user_id, K) for user_id in eval_users}


def recommend_als(path, eval_users, seen, batch_size=512):
    users = np.load(path / "users.npy", allow_pickle=True).tolist()
    items = np.load(path / "items.npy", allow_pickle=True).tolist()
    user_factors = np.load(path / "user_factors.npy")
    item_factors = np.load(path / "item_factors.npy")

    user_to_idx = {user_id: i for i, user_id in enumerate(users)}
    item_to_idx = {item_id: i for i, item_id in enumerate(items)}

    recommendations = {}

    for start in range(0, len(eval_users), batch_size):
        batch_users = eval_users[start:start + batch_size]
        user_indices = [user_to_idx[user_id] for user_id in batch_users]

        scores = user_factors[user_indices] @ item_factors.T
        batch_recs = topk_from_scores(scores, items, [seen[user_id] for user_id in batch_users], item_to_idx, K)

        for user_id, recs in zip(batch_users, batch_recs):
            recommendations[user_id] = recs

    return recommendations


def recommend_ease(train, path, eval_users, seen, batch_size=256):
    users = np.load(path / "users.npy", allow_pickle=True).tolist()
    items = np.load(path / "items.npy", allow_pickle=True).tolist()
    B = np.load(path / "B.npy")

    user_to_idx = {user_id: i for i, user_id in enumerate(users)}
    item_to_idx = {item_id: i for i, item_id in enumerate(items)}

    positive = train.filter(pl.col("rating") >= 4)

    rows = np.fromiter((user_to_idx[user_id] for user_id in positive["user_id"]), dtype=np.int32, count=positive.height)
    cols = np.fromiter((item_to_idx[item_id] for item_id in positive["item_id"]), dtype=np.int32, count=positive.height)

    X = csr_matrix(
        (np.ones(positive.height, dtype=np.float32), (rows, cols)),
        shape=(len(users), len(items)),
        dtype=np.float32,
    )

    recommendations = {}

    for start in range(0, len(eval_users), batch_size):
        batch_users = eval_users[start:start + batch_size]
        user_indices = [user_to_idx[user_id] for user_id in batch_users]

        scores = np.asarray(X[user_indices] @ B)
        batch_recs = topk_from_scores(scores, items, [seen[user_id] for user_id in batch_users], item_to_idx, K)

        for user_id, recs in zip(batch_users, batch_recs):
            recommendations[user_id] = recs

    return recommendations


def recommend_fm(checkpoint_path, eval_users, seen, batch_size=512):
    preprocessing = torch.load(FM_PATH / "preprocessing.pt", map_location="cpu", weights_only=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    mappings = preprocessing["mappings"]
    offsets = preprocessing["offsets"]
    n_features = preprocessing["n_features"]
    dim = preprocessing["dim"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FM(n_features=n_features, dim=dim).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    item_ids = sorted(mappings["item_id"], key=mappings["item_id"].get)
    item_to_idx = {item_id: i for i, item_id in enumerate(item_ids)}

    items = (
        pl.read_parquet(ITEMS_PATH)
        .unique("item_id", keep="first")
        .with_columns(
            pl.col("categories").list.last().fill_null("__UNK__").alias("category"),
            pl.col("store").fill_null("__UNK__"),
        )
    )

    metadata = (
        pl.DataFrame({"item_id": item_ids})
        .join(items.select("item_id", "category", "store"), on="item_id", how="left")
        .with_columns(
            pl.col("category").fill_null("__UNK__"),
            pl.col("store").fill_null("__UNK__"),
        )
    )

    item_feature_ids = torch.tensor(
        [
            [
                offsets["item_id"] + mappings["item_id"][row["item_id"]],
                offsets["category"] + mappings["category"][row["category"]],
                offsets["store"] + mappings["store"][row["store"]],
            ]
            for row in metadata.iter_rows(named=True)
        ],
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        item_linear = model.linear(item_feature_ids).squeeze(-1).sum(dim=1)
        item_vectors = model.embedding(item_feature_ids)
        item_context = item_vectors.sum(dim=1)
        item_pair = 0.5 * ((item_context ** 2 - (item_vectors ** 2).sum(dim=1)).sum(dim=1))
        item_constant = item_linear + item_pair
        bias = model.bias.item()

    recommendations = {}

    with torch.no_grad():
        for start in range(0, len(eval_users), batch_size):
            batch_users = eval_users[start:start + batch_size]

            user_ids = torch.tensor(
                [offsets["user_id"] + mappings["user_id"][user_id] for user_id in batch_users],
                dtype=torch.long,
                device=device,
            )

            user_linear = model.linear(user_ids).squeeze(-1)
            user_vectors = model.embedding(user_ids)

            scores = bias + user_linear[:, None] + item_constant[None, :] + user_vectors @ item_context.T

            for row, user_id in enumerate(batch_users):
                indices = [item_to_idx[item] for item in seen[user_id] if item in item_to_idx]
                if indices:
                    scores[row, torch.tensor(indices, device=device)] = -torch.inf

            top = torch.topk(scores, k=K, dim=1).indices.cpu().numpy()

            for user_id, indices in zip(batch_users, top):
                recommendations[user_id] = [item_ids[i] for i in indices]

    return recommendations


def run_model(name, function, *args):
    started = time.perf_counter()
    recommendations = function(*args)
    print(f"{name}: inference_finished time={time.perf_counter() - started:.1f}s", flush=True)
    return recommendations


def main():
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    train, eval_users, targets, seen, history_len, popularity, candidate_items = prepare_data()
    total_interactions = train.height

    print(f"eval_users={len(eval_users)} candidate_items={len(candidate_items)}", flush=True)

    summaries = []
    per_user = []
    all_recommendations = []

    recommendations = run_model("MostPopular", recommend_most_popular, train, eval_users, seen)
    summary, metrics, recs = evaluate("MostPopular", recommendations, eval_users, targets, history_len, popularity, candidate_items, total_interactions)
    summaries.append(summary)
    per_user.append(metrics)
    all_recommendations.append(recs)

    recommendations = run_model("ItemKNN", recommend_item_knn, train, eval_users)
    summary, metrics, recs = evaluate("ItemKNN", recommendations, eval_users, targets, history_len, popularity, candidate_items, total_interactions)
    summaries.append(summary)
    per_user.append(metrics)
    all_recommendations.append(recs)

    for name, path in [("ALS", ALS_PATH), ("SignalALS", SIGNAL_ALS_PATH)]:
        recommendations = run_model(name, recommend_als, path, eval_users, seen)
        summary, metrics, recs = evaluate(name, recommendations, eval_users, targets, history_len, popularity, candidate_items, total_interactions)
        summaries.append(summary)
        per_user.append(metrics)
        all_recommendations.append(recs)

    recommendations = run_model("EASE", recommend_ease, train, EASE_PATH, eval_users, seen)
    summary, metrics, recs = evaluate("EASE", recommendations, eval_users, targets, history_len, popularity, candidate_items, total_interactions)
    summaries.append(summary)
    per_user.append(metrics)
    all_recommendations.append(recs)

    for checkpoint in sorted(FM_PATH.glob("checkpoint_epoch_*.pt")):
        epoch = int(checkpoint.stem.split("_")[-1])
        name = f"FM_epoch_{epoch:02d}"

        recommendations = run_model(name, recommend_fm, checkpoint, eval_users, seen)
        summary, metrics, recs = evaluate(name, recommendations, eval_users, targets, history_len, popularity, candidate_items, total_interactions)

        summaries.append(summary)
        per_user.append(metrics)
        all_recommendations.append(recs)

    summary_df = pl.DataFrame(summaries).sort("ndcg@10", descending=True)
    per_user_df = pl.concat(per_user)
    recommendations_df = pl.concat(all_recommendations)

    summary_df.write_csv(OUTPUT_PATH / "validation_metrics.csv")
    per_user_df.write_parquet(OUTPUT_PATH / "validation_per_user.parquet")
    recommendations_df.write_parquet(OUTPUT_PATH / "validation_recommendations.parquet")

    print()
    print(summary_df)

    fm = summary_df.filter(pl.col("model").str.starts_with("FM_")).sort("ndcg@10", descending=True)

    if fm.height:
        print()
        print(f"best_fm={fm[0, 'model']} ndcg@10={fm[0, 'ndcg@10']:.5f}")


if __name__ == "__main__":
    main()