import pandas as pd
import numpy as np

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# --------------------------------------------------
# 1. FILE PATHS
# --------------------------------------------------

TRAIN_FILE = "data/train.csv"
VALIDATION_FILE = "data/validation.csv"
TEST_FILE = "data/test.csv"

MODEL_NAME = "bert-base-uncased"

OUTPUT_DIR = "model/bert_automotive_sentiment"


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("Loading datasets...")

train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)
test_df = pd.read_csv(TEST_FILE)

print("Training samples:", len(train_df))
print("Validation samples:", len(validation_df))
print("Testing samples:", len(test_df))


# --------------------------------------------------
# 3. CONVERT PANDAS → HUGGING FACE DATASET
# --------------------------------------------------

train_dataset = Dataset.from_pandas(
    train_df[["bert_text", "label"]],
    preserve_index=False
)

validation_dataset = Dataset.from_pandas(
    validation_df[["bert_text", "label"]],
    preserve_index=False
)

test_dataset = Dataset.from_pandas(
    test_df[["bert_text", "label"]],
    preserve_index=False
)


# --------------------------------------------------
# 4. LOAD BERT TOKENIZER
# --------------------------------------------------

print("\nLoading BERT tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# --------------------------------------------------
# 5. TOKENIZATION
# --------------------------------------------------

def tokenize_function(examples):

    return tokenizer(
        examples["bert_text"],
        truncation=True,
        max_length=128
    )


print("Tokenizing datasets...")

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True
)

validation_dataset = validation_dataset.map(
    tokenize_function,
    batched=True
)

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True
)


# --------------------------------------------------
# 6. LOAD BERT MODEL
# --------------------------------------------------

print("\nLoading BERT model...")

id2label = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

label2id = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2
}

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)


# --------------------------------------------------
# 7. DATA COLLATOR
# --------------------------------------------------

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# --------------------------------------------------
# 8. EVALUATION METRICS
# --------------------------------------------------

def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    predictions = np.argmax(predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# --------------------------------------------------
# 9. TRAINING CONFIGURATION
# --------------------------------------------------

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    learning_rate=2e-5,

    per_device_train_batch_size=8,

    per_device_eval_batch_size=8,

    num_train_epochs=3,

    weight_decay=0.01,

    eval_strategy="epoch",

    save_strategy="epoch",

    load_best_model_at_end=True,

    metric_for_best_model="f1",

    greater_is_better=True,

    logging_steps=10,

    report_to="none",

    use_cpu=True
)


# --------------------------------------------------
# 10. TRAINER
# --------------------------------------------------

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=validation_dataset,

    processing_class=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics
)


# --------------------------------------------------
# 11. START TRAINING
# --------------------------------------------------

print("\n========================================")
print("Starting BERT training...")
print("========================================\n")

trainer.train()


# --------------------------------------------------
# 12. TEST MODEL
# --------------------------------------------------

print("\n========================================")
print("Evaluating model on test dataset...")
print("========================================\n")

test_results = trainer.evaluate(
    test_dataset
)

print("\nTest Results:")

for key, value in test_results.items():

    if isinstance(value, float):

        print(f"{key}: {value:.4f}")

    else:

        print(f"{key}: {value}")


# --------------------------------------------------
# 13. SAVE MODEL
# --------------------------------------------------

print("\nSaving trained model...")

trainer.save_model(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)

print("\n========================================")
print("BERT training completed successfully!")
print("========================================")

print(f"\nModel saved at: {OUTPUT_DIR}")