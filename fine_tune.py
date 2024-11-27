import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import load_from_disk
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
import argparse


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, tweets, labels, tokenizer, max_len=128):
        """
        Args:
            tweets (list of str): List of tweet texts.
            labels (list of int): List of labels (e.g., sentiment or other classification labels).
            tokenizer (RobertaTokenizer): The tokenizer used for tokenizing the text.
            max_len (int, optional): The maximum sequence length after padding/truncation. Default is 128.
        """
        self.tweets = tweets
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        tweet = self.tweets[idx]
        label = self.labels[idx]

        # Tokenize the tweet
        inputs = self.tokenizer(
            tweet,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        # Remove the batch dimension (since tokenizer returns a batch of size 1)
        return {
            'input_ids': inputs['input_ids'].squeeze(0),  # Remove the batch dimension
            'attention_mask': inputs['attention_mask'].squeeze(0),  # Remove the batch dimension
            'labels': torch.tensor(label, dtype=torch.long)  # Convert the label to tensor
        }


def compute_metrics(p):
    predictions, labels = p
    preds = predictions.argmax(axis=1)
    return {"accuracy": accuracy_score(labels, preds)}


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = load_from_disk(args.path_to_dataset)

    # Tokenizer initialization
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # Replace label 4 with 1
    train_labels = list(map(lambda label: 1 if label == 4 else label, dataset['train']['label']))
    test_labels = list(map(lambda label: 1 if label == 4 else label, dataset['test']['label']))

    # Create dataset objects
    train_dataset = CustomDataset(dataset['train']['twitter'], train_labels, tokenizer)
    test_dataset = CustomDataset(dataset['test']['twitter'], test_labels, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,  # Output directory
        num_train_epochs=args.epochs,  # Number of training epochs
        per_device_train_batch_size=args.batch_size,  # Batch size for training
        per_device_eval_batch_size=args.batch_size,  # Batch size for evaluation
        warmup_steps=500,  # Number of warmup steps
        weight_decay=0.01,  # Strength of weight decay
        logging_dir="./logs",  # Directory for storing logs
        logging_steps=10,  # Log every 10 steps
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
        save_strategy="epoch",  # Save model at the end of each epoch
        load_best_model_at_end=True,  # Load the best model at the end of training
        metric_for_best_model="accuracy",  # Use accuracy to evaluate the best model
    )

    # Initialize the model
    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
    model.to(device)

    # Define the Trainer
    trainer = Trainer(
        model=model,  # The model to be fine-tuned
        args=training_args,  # Training arguments
        train_dataset=train_dataset,  # Training dataset
        eval_dataset=test_dataset,  # Evaluation dataset
        compute_metrics=compute_metrics,  # Metrics function
    )

    # Start training
    trainer.train()

    # Save the final model
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa on a custom dataset")
    parser.add_argument('--path_to_dataset', type=str, required=True,
                        help="Path to the dataset (Hugging Face dataset format)")
    parser.add_argument('--epochs', type=int, default=1, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for training and evaluation")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the trained model")

    # Parse arguments
    args = parser.parse_args()

    # Run the main function
    main(args)


# python train_roberta.py \
#     --path_to_dataset "huggingface_dataset_twitter" \
#     --epochs 3 \
#     --batch_size 16 \
#     --output_dir "output_model"
