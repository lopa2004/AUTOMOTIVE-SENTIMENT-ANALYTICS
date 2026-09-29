import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_FILE = "data/processed_reviews.csv"

TRAIN_FILE = "data/train.csv"
VALIDATION_FILE = "data/validation.csv"
TEST_FILE = "data/test.csv"


def main():

    print("Loading processed dataset...")

    df = pd.read_csv(INPUT_FILE)

    print("Original shape:", df.shape)

    # Create BERT input
    df["bert_text"] = (
        df["aspect"] + ": " + df["clean_review"]
    )

    # Convert sentiment labels to numbers
    label_mapping = {
        "Negative": 0,
        "Neutral": 1,
        "Positive": 2
    }

    df["label"] = df["sentiment"].map(label_mapping)

    # First split:
    # 70% training
    # 30% temporary
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df["label"]
    )

    # Split temporary data:
    # 15% validation
    # 15% testing
    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["label"]
    )

    # Save datasets
    train_df.to_csv(TRAIN_FILE, index=False)
    validation_df.to_csv(VALIDATION_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    print("\nDataset preparation completed!")

    print("\nTraining:", train_df.shape)
    print("Validation:", validation_df.shape)
    print("Testing:", test_df.shape)

    print("\nLabel mapping:")
    print(label_mapping)

    print("\nExample BERT input:")
    print(train_df[["bert_text", "sentiment", "label"]].head())


if __name__ == "__main__":
    main()