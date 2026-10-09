"""Add `aesthetic_score` and `captions` from TRELLIS-500K to the CGMLLM_t500k_sketchfab parquet files.

    hf download JeffreyXiang/TRELLIS-500K ObjaverseXL_sketchfab.csv --repo-type dataset --local-dir datasets/TRELLIS-500K
    python scripts/add_sketchfab_captions.py \
        --parquet_dir datasets/CGMLLM_t500k_sketchfab \
        --csv_path datasets/TRELLIS-500K/ObjaverseXL_sketchfab.csv
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import pandas as pd
from tqdm import tqdm


def process_parquet(filename, parquet_dir, output_dir, csv_df):
    df = pd.read_parquet(os.path.join(parquet_dir, filename), columns=["uid", "image_list", "surface"])
    df = df.merge(csv_df, on="uid", how="left", validate="m:1", sort=False)
    df.to_parquet(os.path.join(output_dir, filename), index=False)
    return filename


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_dir", required=True)
    parser.add_argument("--csv_path", required=True, help="ObjaverseXL_sketchfab.csv from TRELLIS-500K.")
    parser.add_argument("--output_dir", default=None, help="Defaults to overwriting the files in --parquet_dir.")
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    output_dir = args.output_dir or args.parquet_dir
    os.makedirs(output_dir, exist_ok=True)

    # The `sha256` column of the Sketchfab csv holds the Sketchfab model id, which is our `uid`.
    csv_df = pd.read_csv(args.csv_path, usecols=["sha256", "aesthetic_score", "captions"])
    csv_df = csv_df.rename(columns={"sha256": "uid"}).drop_duplicates(subset=["uid"])

    files = sorted(f for f in os.listdir(args.parquet_dir) if f.endswith(".parquet"))
    worker = partial(process_parquet, parquet_dir=args.parquet_dir, output_dir=output_dir, csv_df=csv_df)
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        for _ in tqdm(executor.map(worker, files), total=len(files)):
            pass


if __name__ == "__main__":
    main()
