# training/predictor.py
# Load HypothesisValidityBERT for inference

import torch
from transformers import BertForSequenceClassification, BertTokenizer


class HypothesisPredictor:
    """
    Loads the fine-tuned model and scores hypotheses locally.
    No API calls needed — runs entirely on your machine.
    """

    def __init__(self, model_name="Dikshith4500/HypothesisValidityBERT"):
        print("[HypothesisPredictor] Loading model...")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device('cpu')
        self.model.to(self.device)
        print("[HypothesisPredictor] Ready.")

    def predict(self, hypothesis: str) -> dict:
        """
        Score a single hypothesis.
        
        Returns:
            {
                'label': 1 or 0,
                'confidence': 0.0-1.0,
                'verdict': 'valid' or 'flawed'
            }
        """
        inputs = self.tokenizer(
            hypothesis,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            label = torch.argmax(probs, dim=1).item()
            confidence = probs[0][label].item()

        return {
            'label': label,
            'confidence': round(confidence, 4),
            'verdict': 'valid' if label == 1 else 'flawed'
        }

    def predict_batch(self, hypotheses: list) -> list:
        """Score multiple hypotheses at once."""
        return [self.predict(h) for h in hypotheses]


# ── Quick test ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    predictor = HypothesisPredictor()

    test_cases = [
        "Neural networks trained on protein folding data can predict binding affinity with 15% higher accuracy when incorporating graph-based molecular representations compared to sequence-only models.",
        "AI will eventually become conscious and understand the true meaning of existence.",
        "Federated learning with differential privacy guarantees achieves within 3% accuracy of centralized training on medical imaging tasks when using adaptive gradient clipping.",
    ]

    print("\n=== HypothesisValidityBERT Predictions ===\n")
    for h in test_cases:
        result = predictor.predict(h)
        print(f"  [{result['verdict'].upper()}] (confidence: {result['confidence']})")
        print(f"  {h[:80]}...\n")