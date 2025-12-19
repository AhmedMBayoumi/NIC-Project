import numpy as np
import copy
import torch
from sklearn.metrics import accuracy_score
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification
from config import config

def train_and_get_accuracy(params, train_dataset, val_dataset):
    """
    Trains the BERT model with given parameters and returns validation accuracy.
    This is a core function used by NIC optimizers for HPO.
    """
    # Clone params to avoid side effects
    run_params = copy.deepcopy(params)
    
    # Ensure integer types for specific params
    if 'num_train_epochs' in run_params:
        run_params['num_train_epochs'] = int(round(run_params['num_train_epochs']))
    if 'per_device_train_batch_size' in run_params:
        run_params['per_device_train_batch_size'] = int(round(run_params['per_device_train_batch_size']))

    model = AutoModelForSequenceClassification.from_pretrained(config.MODEL_NAME, num_labels=2)
    
    training_args = TrainingArguments(
        output_dir='./sa_results',
        learning_rate=run_params.get('learning_rate', 3e-5),
        num_train_epochs=run_params.get('num_train_epochs', 3),
        weight_decay=run_params.get('weight_decay', 0.01),
        warmup_ratio=run_params.get('warmup_ratio', 0.1),
        adam_beta1=run_params.get('adam_beta1', 0.9),
        adam_beta2=run_params.get('adam_beta2', 0.999),
        per_device_train_batch_size=run_params.get('per_device_train_batch_size', 16),
        per_device_eval_batch_size=32,
        logging_steps=500,
        eval_strategy=\"epoch\",
        save_strategy=\"no\",
        report_to=\"none\",
    )

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        acc = accuracy_score(p.label_ids, preds)
        return {'accuracy': acc}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    try:
        trainer.train()
        eval_results = trainer.evaluate()
        return eval_results['eval_accuracy']
    except Exception as e:
        print(f"Training failed: {e}")
        return 0.0
