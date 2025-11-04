# 
import torch
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch.nn as nn
from config import config

# 
import torch
import torch.nn as nn
import torch.nn.functional as F
import ast
from OCRAttention import OCRAttention

class AnxietyClassifier(nn.Module):
    def __init__(self, config):
        super(AnxietyClassifier, self).__init__()
        self.config = config
        self.hidden_size = config['hidden_size']
        self.num_classes = 6
        self.attention = OCRAttention(config)

        self.classifier = nn.Sequential(
            nn.Linear(2 * self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes)
        )
    
    def forward(self, ocr_repr, k_triples_repr):
        attn_repr = self.attention(ocr_repr, k_triples_repr)  # [batch_size, hidden_size]
        combined = torch.cat((ocr_repr, attn_repr), dim=1)    # [batch_size, 2*hidden_size]
        output = self.classifier(combined)                    # [batch_size, num_classes]
        return output


# 
"""TRAINING LOOP"""

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import torch.optim as optim

train_data = torch.load("train_anxiety_representations.pt")
test_data = torch.load("test_anxiety_representations.pt")

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def prepare_data(data):
    return {
        "ocr_representations": data['ocr_representations'].to(device),
        "triple_representations": data['triple_representations'].to(device),
        "labels": data['labels'].to(device),
        "sample_ids": data['sample_ids'],
        "k_indices": data['k_indices']
    }

train_data = prepare_data(train_data)
test_data = prepare_data(test_data)

model = AnxietyClassifier(config).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5)

def compute_loss(logits, labels):
    ce_loss = F.cross_entropy(logits, labels)
    probs = F.softmax(logits, dim=1)
    one_hot = F.one_hot(labels, num_classes=6).float()
    mse_loss = F.mse_loss(probs, one_hot)
    return 0.5 * ce_loss + 0.5 * mse_loss

import ast
import torch

def get_k_triples_reprs(all_repr, k_indices_batch):
    """
    all_repr: Tensor of shape [N_total_triples, hidden_size] (on device)
    k_indices_batch: List of strings like '[88, 1076, 825]' for each sample

    Returns:
        Tensor of shape [batch_size, k, hidden_size]
    """
    device = all_repr.device
    k_batch = []

    for k_str in k_indices_batch:
        # Parse the string to a list of ints
        k_idx = ast.literal_eval(k_str)  # e.g., [88, 1076, 825]
        k_repr = all_repr[k_idx]         # shape: [k, hidden_size]
        k_batch.append(k_repr)

    return torch.stack(k_batch, dim=0)   # shape: [batch_size, k, hidden_size]

# Training
epochs = 5
batch_size = 32
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for i in range(0, len(train_data['ocr_representations']), batch_size):
        ocr_repr = train_data['ocr_representations'][i:i+batch_size]
        labels = train_data['labels'][i:i+batch_size]
        k_indices = train_data['k_indices'][i:i+batch_size]
        k_triples = get_k_triples_reprs(train_data['triple_representations'], k_indices).to(device)

        logits = model(ocr_repr, k_triples)
        loss = compute_loss(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Train Loss: {total_loss:.4f}")

# Testing
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for i in range(0, len(test_data['ocr_representations']), batch_size):
        ocr_repr = test_data['ocr_representations'][i:i+batch_size]
        labels = test_data['labels'][i:i+batch_size]
        k_indices = test_data['sample_ids'][i:i+batch_size]
        k_triples = get_k_triples_reprs(test_data['triple_representations'], test_data['triple_representations'], k_indices).to(device)

        logits = model(ocr_repr, k_triples)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {accuracy:.4f}")


# 



