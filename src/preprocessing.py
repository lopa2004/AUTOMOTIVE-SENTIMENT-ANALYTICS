import pandas as pd
import re


INPUT_FILE = "data/automotive_reviews.csv"
OUTPUT_FILE = "data/processed_reviews.csv"


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():

    print("Loading dataset...")

    df = pd.read_csv(INPUT_FILE)

    print("Original dataset shape:", df.shape)

    # Remove missing values
    df = df.dropna(subset=["review", "aspect", "sentiment"])

    # Remove duplicate reviews
    df = df.drop_duplicates()

    # Clean review text
    df["clean_review"] = df["review"].apply(clean_text)

    # Remove empty reviews
    df = df[df["clean_review"].str.len() > 0]

    # Save processed dataset
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nPreprocessing completed successfully!")

    print("Final dataset shape:", df.shape)

    print("\nSample processed data:")
    print(df.head())

    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts())

    print("\nAspect distribution:")
    print(df["aspect"].value_counts())

    print(f"\nProcessed file saved at: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()