from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification
from src.data_handler import get_preprocessed_data


def main():
    train_dataset, val_dataset, test_dataset = get_preprocessed_data("data")

    model_name = "FacebookAI/roberta-base"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)

    training_args = TrainingArguments(
        eval_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

    evaluation_results = trainer.evaluate(test_dataset)
    print(f"Evaluation Results: {evaluation_results}")


if __name__ == "__main__":
    main()
