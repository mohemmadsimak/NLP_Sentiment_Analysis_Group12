"""
data_utils.py
Shared dataset loading + sentiment labeling utility for the group project.

All three members import this module so that everyone works from an
identical, reproducible train/test split while each member applies their
own preprocessing / feature engineering and trains their own models.

Dataset: Datafiniti Amazon Consumer Reviews of Amazon Products
File   : data/1429_1.csv
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42

# Resolve the data directory relative to the project root rather than
# depending on the current working directory.

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "1429_1.csv"

REQUIRED_COLUMNS = [
    "id",
    "name",
    "reviews.rating",
    "reviews.text",
    "reviews.title",
]


def rating_to_sentiment(rating: float) -> str:
    """
    Maps the 1-5 star reviews.rating value to a 3-class sentiment label.

    1-2 stars -> Negative
    3 stars   -> Neutral
    4-5 stars -> Positive
    """
    if pd.isna(rating):
        return None

    rating = float(rating)

    if rating <= 2:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    else:
        return "Positive"


def load_dataset(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """
    Load and clean the raw review dataset.

    Steps:
    1. Validate the dataset path.
    2. Validate required columns.
    3. Keep project-relevant columns.
    4. Remove rows with missing review text or rating.
    5. Remove duplicate reviews.
    6. Create the sentiment target.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Please make sure 1429_1.csv is inside the data/ directory."
        )

    df = pd.read_csv(path)

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    # Keep only the columns relevant to this project.
    df = df[REQUIRED_COLUMNS].copy()

    df.rename(
        columns={
            "reviews.rating": "rating",
            "reviews.text": "review_text",
            "reviews.title": "review_title",
        },
        inplace=True,
    )

    # Remove rows with missing review text or rating.
    df = df.dropna(subset=["review_text", "rating"])

    # Remove duplicate reviews.
    df = df.drop_duplicates(subset=["review_text"])

    # Create the sentiment label.
    df["sentiment"] = df["rating"].apply(rating_to_sentiment)
    df = df.dropna(subset=["sentiment"])

    return df.reset_index(drop=True)


def get_split(
    df: pd.DataFrame,
    text_col: str = "review_text",
    label_col: str = "sentiment",
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
):
    """
    Create a reproducible stratified train-test split.

    Stratification keeps the sentiment-class proportions similar in
    both the training and test sets.
    """
    if df.empty:
        raise ValueError("Cannot split an empty dataset.")

    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"Expected columns '{text_col}' and '{label_col}' "
            "were not found in the dataset."
        )

    X = df[text_col]
    y = df[label_col]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


if __name__ == "__main__":
    data = load_dataset()

    print(f"Dataset shape: {data.shape}")
    print("\nSentiment distribution:")
    print(data["sentiment"].value_counts())

    print("\nSentiment percentages:")
    print(data["sentiment"].value_counts(normalize=True).mul(100).round(2))