import os
import glob
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

TEST_RATIO = 0.2
SEED       = 42


def split_and_save(data_dir: str,
                   output_dir: str,
                   test_ratio: float = TEST_RATIO,
                   seed: int = SEED):

    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files in {data_dir}")

    # Get time_ids from first file (shared across all stocks)
    tids = pd.read_csv(files[0], usecols=["time_id"])["time_id"].unique()
    rng  = np.random.default_rng(seed)
    rng.shuffle(tids)

    n_test    = int(len(tids) * test_ratio)
    test_ids  = set(tids[:n_test])
    train_ids = set(tids[n_test:])
    assert train_ids.isdisjoint(test_ids)

    print(f"{len(tids)} time_ids → {len(train_ids)} train, {len(test_ids)} test")

    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.parquet")
    test_path  = os.path.join(output_dir, "test.parquet")

    train_writer = None
    test_writer  = None

    for i, f in enumerate(files):
        stock_id = os.path.basename(f).replace(".csv", "")
        df = pd.read_csv(f)
        df["stock_id"] = stock_id

        train_chunk = df[df["time_id"].isin(train_ids)].reset_index(drop=True)
        test_chunk  = df[df["time_id"].isin(test_ids)].reset_index(drop=True)

        # Write train chunk
        table = pa.Table.from_pandas(train_chunk, preserve_index=False)
        if train_writer is None:
            train_writer = pq.ParquetWriter(train_path, table.schema)
        train_writer.write_table(table)

        # Write test chunk
        table = pa.Table.from_pandas(test_chunk, preserve_index=False)
        if test_writer is None:
            test_writer = pq.ParquetWriter(test_path, table.schema)
        test_writer.write_table(table)

        print(f"[{i+1}/{len(files)}] {stock_id} — train {len(train_chunk)}, test {len(test_chunk)}")

        del df, train_chunk, test_chunk, table

    if train_writer:
        train_writer.close()
    if test_writer:
        test_writer.close()

    print(f"\nDone → {train_path}")
    print(f"     → {test_path}")


if __name__ == "__main__":
    split_and_save(
        data_dir   = r"C:\Users\DELL\OneDrive - The University of Sydney (Students)\individual_book_train_denorm",
        output_dir = "processed",
        test_ratio = TEST_RATIO,
        seed       = SEED,
    )