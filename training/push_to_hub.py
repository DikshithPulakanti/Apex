# training/push_to_hub.py
# Push HypothesisValidityBERT to HuggingFace Hub

import os
from transformers import BertForSequenceClassification, BertTokenizer

model_path = os.path.join(os.path.dirname(__file__), 'model')
repo_name = "Dikshith4500/HypothesisValidityBERT"

print("Loading model...")
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

print(f"Pushing to HuggingFace: {repo_name}")
model.push_to_hub(repo_name, commit_message="feat: HypothesisValidityBERT fine-tuned on 2600 examples, 98% F1")
tokenizer.push_to_hub(repo_name, commit_message="feat: tokenizer for HypothesisValidityBERT")

print(f"\n✅ Model live at: https://huggingface.co/{repo_name}")