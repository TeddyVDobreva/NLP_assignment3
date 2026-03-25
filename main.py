from transformers import (
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.data_handler import get_only_headline_test_dataset, get_preprocessed_data, get_masked_test_dataset
from src.evaluation import plot_confusion_matrix


def main():
    train_dataset, val_dataset, test_dataset = get_preprocessed_data("data")

    model_name = "FacebookAI/roberta-base"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)

    training_args = TrainingArguments(
        eval_strategy="steps",
        eval_steps=6750,
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        metric_for_best_model='eval_loss',
        load_best_model_at_end=True,
        save_steps=6750,
        seed=67,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    
    # accuracy + F1 on validation set
    evaluation_val = trainer.evaluate(val_dataset)
    print(f"Validation Results: {evaluation_val}")
    # confusion matrix on validation set
    plot_confusion_matrix(trainer, val_dataset, model_name, "validation dataset")

    # accuracy + F1 on test set
    evaluation_test = trainer.evaluate(test_dataset)
    print(f"Test Results: {evaluation_test}")
    # confusion matrix on test set
    plot_confusion_matrix(trainer, test_dataset, model_name, "test dataset")

    # Robustness with headlines vs headlines+description
    headlines_test_dataset = get_only_headline_test_dataset("data")
    # accuracy + F1 on test set
    headlines_evaluation_test = trainer.evaluate(headlines_test_dataset)
    print(f"Headlines Results: {headlines_evaluation_test}")
    # confusion matrix on test set
    plot_confusion_matrix(trainer, headlines_test_dataset, model_name, "headlines test")

    # Robustness with keyword masking
    mask_test_dataset = get_masked_test_dataset("/data")
    # accuracy + F1 on test set
    mask_evaluation_test = trainer.evaluate(mask_test_dataset)
    print(f"Masked Results: {mask_evaluation_test}")
    # confusion matrix on test set
    plot_confusion_matrix(trainer, mask_test_dataset, model_name, "mask dataset")



if __name__ == "__main__":
    main()
