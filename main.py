from datasets import load_dataset
import torch
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification


def main():
    device = 0 if torch.cuda.is_available() else -1

    dataset = load_dataset("imdb")
    train_dataset = (
        dataset["train"].shuffle(seed=42).select(range(100))
    )  # Using a subset for quick fine-tuning
    test_dataset = dataset["test"].shuffle(seed=42).select(range(100))


    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    def tokenize_function(examples):
        # print(examples["text"][0])
        return tokenizer(examples["text"], padding="max_length", truncation=True)


    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(
    device
    )
    
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
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
    )
    
    trainer.train()
    
    eval_results = trainer.evaluate()
    print(f"Evaluation Results: {eval_results}")
    
    input_string = "I really liked this tutorial!"

    # Tokenize the input string
    inputs = tokenizer(input_string, return_tensors="pt").to(device)

    # Get predictions (logits)
    with torch.no_grad():  # Disable gradient computation since we're just doing inference
        outputs = model(**inputs)
        logits = outputs.logits

    predicted_label = torch.argmax(logits, dim=1).item()


    print(f"Predicted label: {predicted_label}")
    

if __name__ == "__main__":
    main()
