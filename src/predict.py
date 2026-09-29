import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from aspect_extractor import extract_aspects


MODEL_PATH = "model/bert_automotive_sentiment"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.eval()


LABEL_NAMES = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}


def predict_sentiment(aspect, review):

    text = aspect + ": " + review

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():

        outputs = model(**inputs)

    prediction = torch.argmax(
        outputs.logits,
        dim=1
    ).item()

    return LABEL_NAMES[prediction]


def analyze_review(review):

    aspects = extract_aspects(review)

    results = []

    for aspect in aspects:

        sentiment = predict_sentiment(
            aspect,
            review
        )

        results.append({
            "aspect": aspect,
            "sentiment": sentiment
        })

    return results


if __name__ == "__main__":

    review = input("\nEnter automotive review: ")

    results = analyze_review(review)

    print("\n================================")
    print("Aspect-Level Sentiment Analysis")
    print("================================")

    print("\nReview:", review)

    if not results:

        print("\nNo known automotive aspect detected.")

    else:

        for result in results:

            print(
                f"{result['aspect']} → "
                f"{result['sentiment']}"
            )