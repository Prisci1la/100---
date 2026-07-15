"""
chapter11_utils.py: shared utilities for Chapter 11.
"""

from __future__ import annotations

import csv
import logging
import os
import warnings
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from huggingface_hub.utils import disable_progress_bars
from transformers import AutoTokenizer
from transformers.utils import logging as transformers_logging


BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = "google-bert/bert-base-uncased"
DEFAULT_TRAIN_PATH = BASE_DIR.parent / "Chapter 8" / "data" / "SST-2" / "train.tsv"
DEFAULT_DEV_PATH = BASE_DIR.parent / "Chapter 8" / "data" / "SST-2" / "dev.tsv"
DEFAULT_CHECKPOINT_DIR = BASE_DIR / "checkpoints" / "best_model"
DEFAULT_CUSTOM_CHECKPOINT_DIR = BASE_DIR / "checkpoints" / "best_max_pool_model"
DEFAULT_SENTENCES = [
    "The movie was full of incomprehensibilities.",
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish.",
]


def configure_quiet_mode():
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    warnings.filterwarnings("ignore", message="You are sending unauthenticated requests to the HF Hub.*")
    disable_progress_bars()
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    transformers_logging.disable_progress_bar()
    transformers_logging.set_verbosity_error()


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tokenizer(model_name=MODEL_NAME):
    return AutoTokenizer.from_pretrained(model_name)


def read_sst2_rows(path, max_examples=None):
    rows = []
    with Path(path).open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for row in reader:
            text = row.get("sentence") or row.get("text")
            label = row.get("label")
            if text is None or label is None:
                continue
            rows.append({"text": text.strip(), "label": int(label)})
            if max_examples is not None and len(rows) >= max_examples:
                break
    return rows


class SST2BertDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length=128):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        encoding = self.tokenizer(
            row["text"],
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
        )
        encoding["labels"] = row["label"]
        encoding["text"] = row["text"]
        return encoding


def collate_bert_batch(examples, tokenizer):
    features = [
        {"input_ids": example["input_ids"], "attention_mask": example["attention_mask"]}
        for example in examples
    ]
    batch = tokenizer.pad(features, padding=True, return_tensors="pt")
    batch["labels"] = torch.tensor([example["labels"] for example in examples], dtype=torch.long)
    batch["texts"] = [example["text"] for example in examples]
    return batch


def create_dataset(path, tokenizer, max_length=128, max_examples=None):
    rows = read_sst2_rows(path, max_examples=max_examples)
    return SST2BertDataset(rows, tokenizer, max_length=max_length)


def create_data_loader(dataset, tokenizer, batch_size=16, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda examples: collate_bert_batch(examples, tokenizer),
    )


def save_standard_model(model, tokenizer, output_dir, metrics):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    import json

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def save_custom_model(model, tokenizer, output_dir, metrics):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "pytorch_model.bin")
    tokenizer.save_pretrained(output_dir)
    config = {"model_name": MODEL_NAME, "architecture": "max_pool"}
    import json

    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
