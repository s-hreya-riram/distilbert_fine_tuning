"""
Module 3 — Fine-tuning
Trains DistilBERT on silver-labeled sentences.
Eval set is drawn from manually verified rows where possible.
"""

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from constants import LABELED_DATA_PATH, FINETUNING_DATA_PATH, MODEL_PATH

LABEL_TO_ID = {"vague": 0, "quantified": 1}
ID_TO_LABEL = {0: "vague", 1: "quantified"}
MODEL_NAME = "distilbert-base-uncased"


def load_and_split(path):
    df = pd.read_csv(path)
    df["label"] = df["label"].map(LABEL_TO_ID)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    train_df, eval_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )
    print(f"Train: {len(train_df):,} | Eval: {len(eval_df):,}")
    print(f"Train label dist: {train_df['label'].value_counts().to_dict()}")
    print(f"Eval label dist:  {eval_df['label'].value_counts().to_dict()}")
    return train_df.reset_index(drop=True), eval_df.reset_index(drop=True)


def tokenize_dataset(df: pd.DataFrame, tokenizer) -> Dataset:
    ds = Dataset.from_pandas(df[["text", "label"]])
    ds = ds.map(
        lambda batch: tokenizer(
            batch["text"], truncation=True, padding="max_length", max_length=256
        ),
        batched=True,
    )
    # Trainer expects the input to be a tensor, hence the format change
    ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return ds


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary"),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def train():
    train_df, eval_df = load_and_split(LABELED_DATA_PATH)
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    train_ds = tokenize_dataset(train_df, tokenizer)
    eval_ds  = tokenize_dataset(eval_df, tokenizer)

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    # TODO if using a bigger dataset, consider hyperparameter tuning using Optuna
    args = TrainingArguments(
        output_dir=FINETUNING_DATA_PATH,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro", # using f1 macro as we have a class imbalance
        greater_is_better=True,
        learning_rate=2e-5,
        weight_decay=0.01, # regularization term to prevent overfitting
        warmup_ratio=0.1, # 10% of training steps for warmup
        logging_dir="./logs",
        logging_steps=20,
        fp16=False, # set True if you have a GPU
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        # you must make 0.005 improvement in f1_macro to continue training, 
        # if improvement is less than the threshold for 3 consecutive epochs, training will stop
        # otherwise training will continue until all epochs are completed
        # this is to prevent overfitting
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.005)],
    )

    trainer.train()

    # Full classification report on eval set
    preds_output = trainer.predict(eval_ds)
    preds = np.argmax(preds_output.predictions, axis=-1)
    print("\nClassification Report:")
    print(classification_report(eval_df["label"], preds, target_names=["vague", "quantified"]))

    # Save model and tokenizer
    best_path = MODEL_PATH
    trainer.save_model(best_path)
    tokenizer.save_pretrained(best_path)
    print(f"\nModel saved to {best_path}")


if __name__ == "__main__":
    train()