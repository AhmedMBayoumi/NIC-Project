import torch
import numpy as np
import shap
import random
from lime.lime_text import LimeTextExplainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import config

def explain_samples_shap(model, tokenizer, dataset, num_samples=3, device='cpu'):
    model.to(device)
    model.eval()

    def predict_proba(texts):
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        return torch.nn.functional.softmax(outputs.logits, dim=1).detach().cpu().numpy()

    explainer = shap.Explainer(predict_proba, tokenizer, output_names=["Negative", "Positive"])
    
    try:
        all_texts = dataset.texts
        indices = random.sample(range(len(all_texts)), num_samples)
        sample_texts = [str(all_texts[i]) for i in indices]
    except AttributeError:
        indices = random.sample(range(len(dataset)), num_samples)
        sample_texts = [tokenizer.decode(dataset[i]['input_ids'], skip_special_tokens=True) for i in indices]

    shap_values = explainer(sample_texts)
    return shap_values

def explain_samples_lime(model, tokenizer, dataset, num_samples=3, device='cpu'):
    model.to(device)
    model.eval()

    def predict_proba(texts):
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        return torch.nn.functional.softmax(outputs.logits, dim=1).detach().cpu().numpy()

    explainer = LimeTextExplainer(class_names=['Negative', 'Positive'])
    
    try:
        all_texts = dataset.texts
        indices = random.sample(range(len(all_texts)), num_samples)
        sample_texts = [str(all_texts[i]) for i in indices]
    except AttributeError:
        indices = random.sample(range(len(dataset)), num_samples)
        sample_texts = [tokenizer.decode(dataset[i]['input_ids'], skip_special_tokens=True) for i in indices]

    explanations = []
    for text in sample_texts:
        exp = explainer.explain_instance(text, predict_proba, num_features=10)
        explanations.append(exp)
    
    return explanations
