# training/train_bert.py
# Fine-tune BERT for hypothesis validity classification

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import csv
import random
import numpy as np
import torch
import mlflow
import mlflow.pytorch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup


# ── Reproducibility ───────────────────────────────────────────────────────

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ── Config ────────────────────────────────────────────────────────────────

CONFIG = {
    'model_name':    'bert-base-uncased',
    'max_length':    128,
    'batch_size':    16,
    'epochs':        5,
    'learning_rate': 2e-5,
    'warmup_ratio':  0.1,
    'train_split':   0.8,
    'val_split':     0.1,
    'test_split':    0.1,
}


# ── Dataset class ─────────────────────────────────────────────────────────

class HypothesisDataset(Dataset):
    """
    PyTorch Dataset that holds tokenized hypotheses and labels.
    
    Why a custom Dataset?
    - DataLoader needs __len__ and __getitem__ to batch data
    - Tokenization happens once upfront, not every epoch
    """

    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            texts,
            truncation=True,       # cut text longer than max_length
            padding='max_length',  # pad shorter text to max_length
            max_length=max_length,
            return_tensors='pt'    # return PyTorch tensors
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids':      self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels':         self.labels[idx]
        }


# ── Load data ─────────────────────────────────────────────────────────────

def load_data(csv_path):
    """Read CSV and return texts + labels."""
    texts, labels = [], []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row['hypothesis'])
            labels.append(int(row['label']))

    print(f"Loaded {len(texts)} examples ({sum(labels)} valid, {len(labels) - sum(labels)} flawed)")
    return texts, labels


# ── Training loop ─────────────────────────────────────────────────────────

def train_one_epoch(model, dataloader, optimizer, scheduler, device):
    """
    One pass through the entire training set.
    
    For each batch:
    1. Forward pass  → model predicts, computes loss
    2. Backward pass → compute gradients
    3. Optimizer step → update weights
    4. Scheduler step → adjust learning rate
    """
    model.train()
    total_loss = 0

    for batch in dataloader:
        # Move batch to GPU/CPU
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['labels'].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()   # clear old gradients
        loss.backward()         # compute new gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # prevent exploding gradients
        optimizer.step()        # update weights
        scheduler.step()        # adjust learning rate

        total_loss += loss.item()

    return total_loss / len(dataloader)


# ── Evaluation ────────────────────────────────────────────────────────────

def evaluate(model, dataloader, device):
    """Run model on validation/test set, return metrics."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():  # no gradient computation during eval
        for batch in dataloader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    cm = confusion_matrix(all_labels, all_preds)

    return {
        'accuracy':  accuracy,
        'precision': precision,
        'recall':    recall,
        'f1':        f1,
        'confusion_matrix': cm.tolist()
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=== HypothesisValidityBERT Training ===\n")

    # Device
    device = torch.device('cpu')
    print(f"Device: {device}")

    # Load data
    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'hypotheses.csv')
    texts, labels = load_data(csv_path)

    # Split: 80% train, 10% val, 10% test
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=SEED, stratify=temp_labels
    )

    print(f"Train: {len(train_texts)} | Val: {len(val_texts)} | Test: {len(test_texts)}")

    # Tokenizer + datasets
    tokenizer = BertTokenizer.from_pretrained(CONFIG['model_name'])

    train_dataset = HypothesisDataset(train_texts, train_labels, tokenizer, CONFIG['max_length'])
    val_dataset   = HypothesisDataset(val_texts, val_labels, tokenizer, CONFIG['max_length'])
    test_dataset  = HypothesisDataset(test_texts, test_labels, tokenizer, CONFIG['max_length'])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
    test_loader  = DataLoader(test_dataset, batch_size=CONFIG['batch_size'])

    # Model
    model = BertForSequenceClassification.from_pretrained(CONFIG['model_name'], num_labels=2)
    model.to(device)

    # Optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=0.01)
    total_steps = len(train_loader) * CONFIG['epochs']
    warmup_steps = int(total_steps * CONFIG['warmup_ratio'])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # MLflow tracking
    mlflow.set_experiment("hypothesis-validity-bert")

    with mlflow.start_run(run_name="bert-base-v1"):
        # Log config
        mlflow.log_params(CONFIG)
        mlflow.log_param("dataset_size", len(texts))
        mlflow.log_param("device", str(device))

        # Training loop
        best_val_f1 = 0
        model_save_path = os.path.join(os.path.dirname(__file__), 'model')
        os.makedirs(model_save_path, exist_ok=True)

        for epoch in range(CONFIG['epochs']):
            print(f"\n--- Epoch {epoch + 1}/{CONFIG['epochs']} ---")

            # Train
            avg_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
            print(f"  Train loss: {avg_loss:.4f}")
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

            # Validate
            val_metrics = evaluate(model, val_loader, device)
            print(f"  Val accuracy:  {val_metrics['accuracy']:.4f}")
            print(f"  Val precision: {val_metrics['precision']:.4f}")
            print(f"  Val recall:    {val_metrics['recall']:.4f}")
            print(f"  Val F1:        {val_metrics['f1']:.4f}")

            mlflow.log_metric("val_accuracy", val_metrics['accuracy'], step=epoch)
            mlflow.log_metric("val_precision", val_metrics['precision'], step=epoch)
            mlflow.log_metric("val_recall", val_metrics['recall'], step=epoch)
            mlflow.log_metric("val_f1", val_metrics['f1'], step=epoch)

            # Save best model
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                model.save_pretrained(model_save_path)
                tokenizer.save_pretrained(model_save_path)
                print(f"  ✅ New best model saved (F1: {best_val_f1:.4f})")

        # Test evaluation
        print(f"\n--- Final Test Evaluation ---")

        # Reload best model
        model = BertForSequenceClassification.from_pretrained(model_save_path)
        model.to(device)

        test_metrics = evaluate(model, test_loader, device)
        print(f"  Test accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"  Test precision: {test_metrics['precision']:.4f}")
        print(f"  Test recall:    {test_metrics['recall']:.4f}")
        print(f"  Test F1:        {test_metrics['f1']:.4f}")
        print(f"  Confusion matrix: {test_metrics['confusion_matrix']}")

        mlflow.log_metric("test_accuracy", test_metrics['accuracy'])
        mlflow.log_metric("test_precision", test_metrics['precision'])
        mlflow.log_metric("test_recall", test_metrics['recall'])
        mlflow.log_metric("test_f1", test_metrics['f1'])
        mlflow.log_param("confusion_matrix", json.dumps(test_metrics['confusion_matrix']))

        # Log model to MLflow
        mlflow.pytorch.log_model(model, "hypothesis-validity-bert")

        print(f"\n=== Training Complete ===")
        print(f"Best val F1:  {best_val_f1:.4f}")
        print(f"Test F1:      {test_metrics['f1']:.4f}")
        print(f"Model saved:  {model_save_path}")
        print(f"MLflow run:   {mlflow.active_run().info.run_id}")


if __name__ == '__main__':
    main()