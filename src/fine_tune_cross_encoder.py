"""
Fine-tune a Cross-Encoder on Amazon ESCI using Native PyTorch & AdamW
Optimizes Binary Cross-Entropy (BCEWithLogitsLoss) against graded ESCI soft targets.
Saves the fine-tuned model checkpoint to models/esci_cross_encoder/
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
import time
import argparse
from pathlib import Path
import polars as pl
import torch
torch.set_num_threads(1)
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from cascade_engine import clean_text, ESCI_GAINS


class ESCIPairDataset(Dataset):
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


def fine_tune_cross_encoder(
    num_train_pairs: int = 600,
    epochs: int = 1,
    batch_size: int = 16,
    lr: float = 3e-5,
    output_dir: str = "models/esci_cross_encoder"
):
    print("=" * 75)
    print("FINE-TUNING CROSS-ENCODER ON AMAZON ESCI RELEVANCE JUDGMENTS")
    print("=" * 75)

    # 1. Load ESCI Judgments
    print("Loading ESCI judgments from local Parquet...")
    examples_df = pl.read_parquet(
        "shopping_queries_dataset/shopping_queries_dataset_examples.parquet",
        columns=["query_id", "query", "product_id", "product_locale", "esci_label", "small_version", "split"]
    ).filter(
        (pl.col("product_locale") == "us") &
        (pl.col("small_version") == 1) &
        (pl.col("split") == "train")
    ).head(num_train_pairs)

    needed_pids = set(examples_df["product_id"].unique().to_list())
    print(f"Loading metadata for {len(needed_pids):,} products...")

    products_raw = pl.scan_parquet(
        "shopping_queries_dataset/shopping_queries_dataset_products.parquet"
    ).filter(
        (pl.col("product_locale") == "us") &
        (pl.col("product_id").is_in(list(needed_pids)))
    ).collect()

    prod_title_map = dict(zip(
        products_raw["product_id"].to_list(),
        [clean_text(t) for t in products_raw["product_title"].to_list()]
    ))

    # 2. Build training lists
    queries = []
    titles = []
    labels = []
    for row in examples_df.to_dicts():
        pid = row["product_id"]
        title = prod_title_map.get(pid, "")
        if not title:
            continue
        queries.append(row["query"])
        titles.append(title)
        labels.append(ESCI_GAINS.get(row["esci_label"], 0.0))

    print(f"Constructed {len(labels)} training pairs with graded soft targets (E=1.0, S=0.1, C=0.01, I=0.0).")

    # 3. Load Model and Tokenizer
    base_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    print(f"Loading base model: {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=1)

    dataset = ESCIPairDataset(queries, titles, labels, tokenizer, max_length=128)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    # 4. Optimizer and Loss Function
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fct = torch.nn.BCEWithLogitsLoss()

    model.train()
    print(f"\nStarting PyTorch training loop: {epochs} epoch(s), {len(dataloader)} batches, lr={lr}...")

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for step, batch in enumerate(dataloader, start=1):
            target_labels = batch.pop("labels")
            outputs = model(**batch)
            logits = outputs.logits.squeeze(-1)

            loss = loss_fct(logits, target_labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            if step % 10 == 0 or step == len(dataloader):
                print(f"  Epoch {epoch}/{epochs} | Step {step:2d}/{len(dataloader)} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} finished with Average Loss: {avg_loss:.4f}")

    train_time = time.time() - t0

    # 5. Save Model and Tokenizer
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n" + "=" * 75)
    print(f"FINE-TUNING COMPLETED in {train_time:.1f} seconds!")
    print(f"Saved custom fine-tuned weights and config to: {output_dir}")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=600, help="Number of training pairs")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--output", type=str, default="models/esci_cross_encoder")
    args = parser.parse_args()

    fine_tune_cross_encoder(num_train_pairs=args.pairs, epochs=args.epochs, output_dir=args.output)
