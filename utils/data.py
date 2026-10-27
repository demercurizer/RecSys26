from pathlib import Path
import gzip
import orjson

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from tqdm.auto import tqdm
from isal import igzip

REVIEWS_SCHEMA = pa.schema([
    ("user_id", pa.string()),
    ("item_id", pa.string()),
    ("rating", pa.float32()),
    ("timestamp", pa.int64()),
    ("verified_purchase", pa.bool_()),
])

def reviews_to_parquet(src, dst, chunk_size=500_000):
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    tmp_dst = dst.with_suffix(".tmp.parquet")

    if tmp_dst.exists():
        tmp_dst.unlink()

    users = []
    items = []
    ratings = []
    timestamps = []
    verified = []

    try:
        with igzip.open(src, "rb") as f, \
             pq.ParquetWriter(tmp_dst, REVIEWS_SCHEMA) as writer, \
             tqdm(desc="Reviews", unit=" rows") as pbar:

            for line in f:
                row = orjson.loads(line)

                users.append(row["user_id"])
                items.append(row["parent_asin"])
                ratings.append(row["rating"])
                timestamps.append(row["timestamp"])
                verified.append(row.get("verified_purchase", False))

                if len(users) >= chunk_size:
                    writer.write_table(
                        pa.table(
                            {
                                "user_id": users,
                                "item_id": items,
                                "rating": ratings,
                                "timestamp": timestamps,
                                "verified_purchase": verified,
                            },
                            schema=REVIEWS_SCHEMA,
                        )
                    )

                    pbar.update(len(users))

                    users.clear()
                    items.clear()
                    ratings.clear()
                    timestamps.clear()
                    verified.clear()

            if users:
                writer.write_table(
                    pa.table(
                        {
                            "user_id": users,
                            "item_id": items,
                            "rating": ratings,
                            "timestamp": timestamps,
                            "verified_purchase": verified,
                        },
                        schema=REVIEWS_SCHEMA,
                    )
                )

                pbar.update(len(users))

        tmp_dst.replace(dst)

    except Exception:
        if tmp_dst.exists():
            tmp_dst.unlink()
        raise

    return dst


METADATA_SCHEMA = pa.schema([
    ("item_id", pa.string()),
    ("title", pa.string()),
    ("main_category", pa.string()),
    ("categories", pa.list_(pa.string())),
    ("store", pa.string()),
    ("price", pa.float32()),
])

def metadata_to_parquet(src, dst, chunk_size=100_000):
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_dst = dst.with_suffix(".tmp.parquet")

    items, titles, main_categories, categories, stores, prices = [], [], [], [], [], []

    try:
        with igzip.open(src, "rb") as f, pq.ParquetWriter(tmp_dst, METADATA_SCHEMA) as writer:
            for line in f:
                row = orjson.loads(line)

                items.append(row["parent_asin"])
                titles.append(row.get("title"))
                main_categories.append(row.get("main_category"))
                categories.append(row.get("categories") or [])
                stores.append(row.get("store"))
                prices.append(row.get("price"))

                if len(items) >= chunk_size:
                    writer.write_table(pa.table({
                        "item_id": items,
                        "title": titles,
                        "main_category": main_categories,
                        "categories": categories,
                        "store": stores,
                        "price": prices,
                    }, schema=METADATA_SCHEMA))

                    items.clear(); titles.clear(); main_categories.clear(); categories.clear(); stores.clear(); prices.clear()

            if items:
                writer.write_table(pa.table({
                    "item_id": items,
                    "title": titles,
                    "main_category": main_categories,
                    "categories": categories,
                    "store": stores,
                    "price": prices,
                }, schema=METADATA_SCHEMA))

        tmp_dst.replace(dst)

    except Exception:
        if tmp_dst.exists():
            tmp_dst.unlink()
        raise

    return dst