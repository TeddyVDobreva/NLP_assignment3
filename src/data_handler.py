from typing import Dict, Tuple

import pandas as pd
from datasets import Dataset as ds
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

PAD = "<pad>"
UNK = "<unk>"
MASK = "<mask>"
MAX_LEN = 64
BATCH_SIZE = 64

# Subsample for speed
N_TRAIN = 500
N_VAL = 100
N_TEST = 100
MASK_WORDS = [
    "investor",
    "stocks",
    "sales",
    "company",
    "market",
    "internet",
    "microsoft",
    "google",
    "software",
    "technology",
    "game",
    "olympic",
    "coach",
    "season",
    "league",
    "president",
    "world",
    "national",
    "oil",
    "government",
]


def _get_raw_data(path: str) -> Dict[str, ds]:
    """
    Load the raw dataset from CSV files, split the training data into
    training and validation sets, and convert them to HuggingFace datasets.

    Args:
        path: path to the directory containing the dataset CSV files

    Returns:
        Dictionary containing three datasets:
            - train: training dataset
            - validation: validation dataset
            - test: test dataset
    """
    train_data = pd.read_csv(path + "/train.csv")
    test_data = pd.read_csv(path + "/test.csv")

    train_data, validation_data = train_test_split(
        train_data, test_size=0.1, random_state=67
    )

    X_train = pd.DataFrame(
        {
            "text": train_data["Title"] + train_data["Description"],
            "label": [idx - 1 for idx in train_data["Class Index"]],
        }
    )
    X_validation = pd.DataFrame(
        {
            "text": validation_data["Title"] + validation_data["Description"],
            "label": [idx - 1 for idx in validation_data["Class Index"]],
        }
    )
    X_test = pd.DataFrame(
        {
            "text": test_data["Title"] + test_data["Description"],
            "label": [idx - 1 for idx in test_data["Class Index"]],
        }
    )

    return {
        "train": ds.from_pandas(X_train, preserve_index=False),
        "validation": ds.from_pandas(X_validation, preserve_index=False),
        "test": ds.from_pandas(X_test, preserve_index=False),
    }


def _get_smaller_datasets(
    raw: Dict[str, ds],
) -> Tuple[ds, ds, ds]:
    """
    Create smaller subsets of the datasets for faster experimentation.

    Args:
        raw: Dictionary containing the original datasets ("train",
        "validation", and "test").

    Returns:
        A tuple containing:
            - training dataset
            - validation dataset
            - test dataset
    """
    train = raw["train"].shuffle(seed=67).select(range(N_TRAIN))
    validation = raw["validation"].shuffle(seed=67).select(range(N_VAL))
    test = raw["test"].select(range(N_TEST))
    return train, validation, test


def _get_datasets(raw: Dict[str, ds]) -> Tuple[ds, ds, ds]:
    """
    Retrieve the datasets used for training, validation, and testing.

    Args:
        raw: Dictionary containing the datasets ("train",
        "validation", and "test".)

    Returns:
        A tuple containing:
            - training dataset
            - validation dataset
            - test dataset
    """
    train = raw["train"].shuffle(seed=67)  # We only shuffle the training set
    validation = raw["validation"]
    test = raw["test"]
    return train, validation, test


def _tokenize_function(examples):
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
    # print(examples["text"][0])
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def _mask(dataset):
    for text in dataset["text"]:
        for word in MASK_WORDS:
            text.replace(word, MASK)


def get_preprocessed_data(path, small=True):
    raw = _get_raw_data(path)
    if small:
        train_dataset, val_dataset, test_dataset = _get_smaller_datasets(raw)
    else:
        train_dataset, val_dataset, test_dataset = _get_datasets(raw)

    tokenized_train = train_dataset.map(_tokenize_function, batched=True)
    tokenized_val = val_dataset.map(_tokenize_function, batched=True)
    tokenized_test = test_dataset.map(_tokenize_function, batched=True)

    return tokenized_train, tokenized_val, tokenized_test


def get_only_headline_test_dataset(path):
    test_data = pd.read_csv(path + "/test.csv")
    X_test = pd.DataFrame(
        {
            "text": test_data["Title"],
            "label": test_data["Class Index"],
        }
    )
    return X_test.map(_tokenize_function, batched=True)


def get_masked_test_dataset(path):
    test_data = pd.read_csv(path + "/test.csv")
    X_test = pd.DataFrame(
        {
            "text": test_data["Title"] + test_data["Description"],
            "label": test_data["Class Index"],
        }
    )
    X_test = _mask(X_test)
    return X_test.map(_tokenize_function, batched=True)
