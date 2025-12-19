import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from config import config

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            text, add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True, return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_and_prep_data(file_path=config.DATA_PATH):
    col_names = ['target', 'ids', 'date', 'flag', 'user', 'text']
    try:
        df = pd.read_csv(file_path, encoding='ISO-8859-1', names=col_names)
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}. Please update config/config.py")
        return None, None

    df['target'] = df['target'].replace(4, 1)
    df = df[['text', 'target']]
    
    sample_n = config.SAMPLE_SIZE // 2
    df_pos = df[df['target'] == 1].sample(n=sample_n, random_state=config.RANDOM_STATE)
    df_neg = df[df['target'] == 0].sample(n=sample_n, random_state=config.RANDOM_STATE)
    df = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=config.RANDOM_STATE).reset_index(drop=True)
    
    df_train, df_val = train_test_split(
        df, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=df['target']
    )
    
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    train_dataset = SentimentDataset(
        texts=df_train['text'].to_list(),
        labels=df_train['target'].to_list(),
        tokenizer=tokenizer, max_len=config.MAX_LENGTH
    )
    val_dataset = SentimentDataset(
        texts=df_val['text'].to_list(),
        labels=df_val['target'].to_list(),
        tokenizer=tokenizer, max_len=config.MAX_LENGTH
    )
    
    return train_dataset, val_dataset
