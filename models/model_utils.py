import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from config import config

def get_tokenizer():
    return AutoTokenizer.from_pretrained(config.MODEL_NAME)

def get_base_model():
    return AutoModel.from_pretrained(config.MODEL_NAME)

def get_classification_model(num_labels=2):
    return AutoModelForSequenceClassification.from_pretrained(config.MODEL_NAME, num_labels=num_labels)

@torch.no_grad()
def extract_features(model, dataloader, device):
    model.to(device)
    model.eval()  
    all_features = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Extracting features"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_features = last_hidden_state[:, 0, :]
        
        all_features.append(cls_features.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    features = np.concatenate(all_features)
    labels = np.concatenate(all_labels)
    
    return features, labels
