from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification
from src.data_handler import get_preprocessed_data, get_only_headline_test_dataset


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
    
    # accuracy + F1 on validation set
    evaluation_val= trainer.evaluate(val_dataset)
    print(f"Evaluation Results: {evaluation_val}")
    # confusion matrix on validation set
    
    # accuracy + F1 on test set
    evaluation_test= trainer.evaluate(test_dataset)
    print(f"Evaluation Results: {evaluation_test}")
    # confusion matrix on test set
    
    # Robustness with headlines vs headlines+description
    headlines_test_dataset = get_only_headline_test_dataset("data")
    
    # accuracy + F1 on test set
    headlines_evaluation_test= trainer.evaluate(headlines_test_dataset)
    print(f"Evaluation Results: {headlines_evaluation_test}")
    # confusion matrix on test set
    
    
    # Robustness with keyword masking 
    mask_test_dataset = None
    # accuracy + F1 on test set
    mask_evaluation_test= trainer.evaluate(mask_test_dataset)
    print(f"Evaluation Results: {mask_evaluation_test}")
    # confusion matrix on test set

if __name__ == "__main__":
    main()

