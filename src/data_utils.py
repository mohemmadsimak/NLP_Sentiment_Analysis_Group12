"""
data_utils.py
Shared dataset loading + sentiment labeling utility for the group project.
All three members import this module so that everyone works from an
identical, reproducible train/test split (same rows, same labels),
while each member applies their OWN preprocessing / feature engineering
and trains their OWN unique ML + DL model.

Dataset: Datafiniti Amazon Consumer Reviews of Amazon Products
File   : data/1429_1.csv
Source : https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazonproducts
"""

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
DATA_PATH = "../data/1429_1.csv"


def rating_to_sentiment(rating: float) -> str:
    """
    Maps the 1-5 star `reviews.rating` column to a 3-class sentiment label.
    Mapping agreed by the group:
        1-2 stars -> Negative
        3   stars -> Neutral
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


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Loads the raw CSV, keeps only the columns needed for sentiment analysis,
    drops rows with missing review text or missing rating, removes duplicate
    reviews, and creates the `sentiment` target column.
    """
    df = pd.read_csv(path)

    # Keep only the columns relevant to this project
    df = df[["id", "name", "reviews.rating", "reviews.text", "reviews.title"]].copy()
    df.rename(
        columns={
            "reviews.rating": "rating",
            "reviews.text": "review_text",
            "reviews.title": "review_title",
        },
        inplace=True,
    )

    # Handle missing values (Section 6 of the validation doc: "Missing Values")
    df = df.dropna(subset=["review_text", "rating"])

    # Remove duplicate reviews (Section 6: "Duplicate Data Points")
    df = df.drop_duplicates(subset=["review_text"])

    # Build the sentiment label
    df["sentiment"] = df["rating"].apply(rating_to_sentiment)
    df = df.dropna(subset=["sentiment"])

    df = df.reset_index(drop=True)
    return df


def get_split(df: pd.DataFrame, text_col: str = "review_text", label_col: str = "sentiment",
              test_size: float = 0.2, random_state: int = RANDOM_STATE):
    """
    Common stratified 80/20 train-test split so every member evaluates on
    a comparable holdout set.
    """
    X = df[text_col]
    y = df[label_col]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


if __name__ == "__main__":
    data = load_dataset()
    print(data.shape)
    print(data["sentiment"].value_counts())
