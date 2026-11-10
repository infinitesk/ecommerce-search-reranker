#!/usr/bin/env bash
# Download Amazon ESCI Shopping Queries Dataset into shopping_queries_dataset/

set -e

DATA_DIR="shopping_queries_dataset"
mkdir -p "$DATA_DIR"

BASE_URL="https://github.com/amazon-science/esci-data/raw/main/shopping_queries_dataset"

echo "Downloading Amazon ESCI Dataset into $DATA_DIR/..."

if [ ! -f "$DATA_DIR/shopping_queries_dataset_examples.parquet" ]; then
    echo "Downloading shopping_queries_dataset_examples.parquet..."
    curl -L "$BASE_URL/shopping_queries_dataset_examples.parquet" -o "$DATA_DIR/shopping_queries_dataset_examples.parquet"
fi

if [ ! -f "$DATA_DIR/shopping_queries_dataset_sources.csv" ]; then
    echo "Downloading shopping_queries_dataset_sources.csv..."
    curl -L "$BASE_URL/shopping_queries_dataset_sources.csv" -o "$DATA_DIR/shopping_queries_dataset_sources.csv"
fi

if [ ! -f "$DATA_DIR/shopping_queries_dataset_products.parquet" ]; then
    echo "Note: shopping_queries_dataset_products.parquet is ~1GB."
    echo "Downloading shopping_queries_dataset_products.parquet from Amazon Science repo..."
    curl -L "$BASE_URL/shopping_queries_dataset_products.parquet" -o "$DATA_DIR/shopping_queries_dataset_products.parquet"
fi

echo "Dataset setup completed successfully."
