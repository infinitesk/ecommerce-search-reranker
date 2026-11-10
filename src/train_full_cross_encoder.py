"""
Full Dataset Fine-Tuning of Cross-Encoder on Amazon ESCI (419,653 Pairs)
Optimizes Binary Cross-Entropy with Logits against Graded ESCI Soft Targets:
  - Exact (E)       -> 1.0
  - Substitute (S)  -> 0.1
  - Complement (C)  -> 0.01
  - Irrelevant (I)  -> 0.0

Hardware optimization: Apple Silicon 8-thread vectorized execution (batch_size=64).
Saves checkpoint to models/esci_cross_encoder_full/
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "8"
import time
import argparse
from pathlib import Path
import numpy as np
import polars as pl
import torch
torch.set_num_threads(8)
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from cascade_engine import clean_text, ESCI_GAINS


class ESCIFullDataset(Dataset):
    def __init__(self, queries, titles, labels, tokenizer, max_length=128):
        self.queries = queries
        self.titles = titles
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "query": self.queries[idx],
            "title": self.titles[idx],
            "label": self.labels[idx]
        }

    def collate_fn(self, batch):
        queries = [item["query"] for item in batch]
        titles = [item["title"] for item in batch]
        labels = torch.stack([item["label"] for item in batch])

        encoded = self.tokenizer(
            queries,
            titles,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        encoded["labels"] = labels
        return encoded


def run_full_fine_tuning(
    batch_size: int = 64,
    lr: float = 2e-5,
    epochs: int = 1,
    output_dir: str = "models/esci_cross_encoder_full"
):
    print("=" * 75)
    print("FULL DATASET CROSS-ENCODER FINE-TUNING ON AMAZON ESCI")
    print("=" * 75)

    # 1. Load All Train Pairs
    t0 = time.time()
    print("Loading full ESCI US Task 1 training dataset...")
    train_df = pl.read_parquet(
        "shopping_queries_dataset/shopping_queries_dataset_examples.parquet",
        columns=["query_id", "query", "product_id", "product_locale", "esci_label", "small_version", "split"]
    ).filter(
        (pl.col("product_locale") == "us") &
        (pl.col("small_version") == 1) &
        (pl.col("split") == "train")
    )

    needed_pids = set(train_df["product_id"].unique().to_list())
    print(f"Loaded {train_df.height:,} judgments over {train_df['query_id'].n_unique():,} unique queries.")
    print(f"Loading metadata for {len(needed_pids):,} products...")

    products_df = pl.scan_parquet(
        "shopping_queries_dataset/shopping_queries_dataset_products.parquet"
    ).filter(
        (pl.col("product_locale") == "us") &
        (pl.col("product_id").is_in(list(needed_pids)))
    ).select(["product_id", "product_title"]).collect()

    print("Joining examples with product titles...")
    joined = train_df.join(products_df, on="product_id", how="inner").filter(
        pl.col("product_title").is_not_null()
    )

    queries = joined["query"].to_list()
    titles = [clean_text(t) for t in joined["product_title"].to_list()]
    raw_labels = joined["esci_label"].to_list()
    labels = [ESCI_GAINS.get(lbl, 0.0) for lbl in raw_labels]

    total_pairs = len(labels)
    print(f"Prepared {total_pairs:,} training pairs across all ESCI train queries in {time.time()-t0:.2f}s.")

    # Print label distribution
    label_counts = pl.Series("labels", raw_labels).value_counts()
    print("\nTraining Set ESCI Label Distribution:")
    for row in label_counts.to_dicts():
        lbl = row["labels"]
        cnt = row["count"]
        pct = (cnt / total_pairs) * 100
        gain = ESCI_GAINS.get(lbl, 0.0)
        print(f"  {lbl} (Soft Target: {gain:4.2f}): {cnt:7,} pairs ({pct:5.1f}%)")

    # 2. Initialize Model & Tokenizer
    base_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    print(f"\nInitializing base Cross-Encoder: {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=1)

    dataset = ESCIFullDataset(queries, titles, labels, tokenizer, max_length=128)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=0
    )

    total_steps = len(dataloader) * epochs
    warmup_steps = int(total_steps * 0.05)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    loss_fct = torch.nn.BCEWithLogitsLoss()

    # 3. Training Loop
    print("\n" + "=" * 75)
    print(f"STARTING FULL TRAINING: {total_pairs:,} pairs | {total_steps:,} steps | batch_size={batch_size}")
    print(f"Hardware: 8 CPU threads | Learning Rate: {lr} | Warmup Steps: {warmup_steps}")
    print("=" * 75)

    model.train()
    start_time = time.time()
    running_loss = 0.0
    log_interval = 200

    for epoch in range(1, epochs + 1):
        for step, batch in enumerate(dataloader, start=1):
            t_batch_start = time.perf_counter()
            target_labels = batch.pop("labels")

            outputs = model(**batch)
            logits = outputs.logits.squeeze(-1)

            loss = loss_fct(logits, target_labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item()

            if step % log_interval == 0 or step == total_steps:
                elapsed = time.time() - start_time
                avg_loss = running_loss / (log_interval if step % log_interval == 0 else (step % log_interval))
                running_loss = 0.0

                speed = (step * batch_size) / elapsed
                remaining_steps = total_steps - step
                eta_sec = remaining_steps * (elapsed / step)
                eta_min = eta_sec / 60

                print(f"Step {step:5,d}/{total_steps:,d} ({(step/total_steps)*100:5.1f}%) | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Speed: {speed:5.1f} pairs/s | "
                      f"Elapsed: {elapsed/60:4.1f}m | "
                      f"ETA: {eta_min:4.1f}m")

    total_training_time = time.time() - start_time

    # 4. Save Final Weights
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n" + "=" * 75)
    print(f"FULL DATASET TRAINING COMPLETE in {total_training_time/60:.2f} minutes ({total_training_time:.1f}s)!")
    print(f"Final fine-tuned weights saved to: {output_dir}")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output", type=str, default="models/esci_cross_encoder_full")
    args = parser.parse_args()

    run_full_fine_tuning(batch_size=args.batch_size, lr=args.lr, output_dir=args.output)
