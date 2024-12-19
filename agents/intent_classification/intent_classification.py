import json

import torch
from torch.optim import AdamW
import torch.nn as nn
from fpdf import FPDF
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd


# Load JSON files and convert to pandas DataFrame
def load_data(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return pd.DataFrame(data, columns=['sentence', 'intent'])
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return pd.DataFrame(columns=['sentence', 'intent'])

# Load data
train_df = load_data('./is_train.json')
val_df = load_data('./is_val.json')
test_df = load_data('./is_test.json')

# Map intents to numerical labels
intent_labels = {intent: idx for idx, intent in enumerate(train_df['intent'].unique())}
train_df['label'] = train_df['intent'].map(intent_labels)
val_df['label'] = val_df['intent'].map(intent_labels)
test_df['label'] = test_df['intent'].map(intent_labels)

# Prepare data
train_texts = train_df['sentence'].tolist()
train_labels = train_df['label'].tolist()
val_texts = val_df['sentence'].tolist()
val_labels = val_df['label'].tolist()
test_texts = test_df['sentence'].tolist()
test_labels = test_df['label'].tolist()

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Tokenize data
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

# Dataset class
class IntentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

# Create dataset objects
train_dataset = IntentDataset(train_encodings, train_labels)
val_dataset = IntentDataset(val_encodings, val_labels)
test_dataset = IntentDataset(test_encodings, test_labels)

# Specify the directory to save/load model
MODEL_PATH = 'D:/semester 7/FYP/agents/intent_classification/model'

# Model initialization
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(intent_labels))
optimizer = AdamW(model.parameters(), lr=3e-4)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Training loop
def train(model, train_loader, val_loader, optimizer, num_epochs=3):
    for epoch in range(num_epochs):
        model.train()
        total_loss, total_val_loss, total_val_accuracy = 0, 0, 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
                loss = outputs.loss
                logits = outputs.logits
                total_val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                total_val_accuracy += accuracy_score(batch['labels'].cpu(), preds.cpu())

        print(f"Epoch {epoch+1}: Train Loss: {total_loss / len(train_loader)}, Val Loss: {total_val_loss / len(val_loader)}, Val Accuracy: {total_val_accuracy / len(val_loader)}")

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# Training the model
train(model, train_loader, val_loader, optimizer)
model.save_pretrained(MODEL_PATH)
tokenizer.save_pretrained(MODEL_PATH)

# Load the model for evaluation
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)

# Evaluation
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())

    report = classification_report(all_labels, all_preds, target_names=intent_labels.keys())
    print("Classification Report:")
    print(report)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in report.split('\n'):
        pdf.cell(0, 10, line, ln=True)
    pdf.output("classification_report.pdf")

# Evaluate the model on the test set
evaluate(model, test_loader)

# Prediction function
def predict_intent(model, tokenizer, sentence):
    model.eval()
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()
    return list(intent_labels.keys())[predicted_label]

def classify_intent(text):
    # Assume the model and tokenizer are set up correctly
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()
    intent = list(intent_labels.keys())[predicted_label]
    return intent
