# tests/test_predictor.py
# Unit tests for HypothesisPredictor — BERT model/tokenizer are mocked
# so these tests don't download real weights from HuggingFace.

from unittest.mock import MagicMock

import torch

from training.predictor import HypothesisPredictor


def _build_predictor_with_mocked_model(logits):
    predictor = HypothesisPredictor.__new__(HypothesisPredictor)  # skip __init__ (no HF download)

    fake_inputs = {
        'input_ids': torch.zeros((1, 128), dtype=torch.long),
        'attention_mask': torch.ones((1, 128), dtype=torch.long),
    }
    tokenizer = MagicMock()
    tokenizer.return_value.to.return_value = fake_inputs

    model_output = MagicMock()
    model_output.logits = torch.tensor([logits])
    model = MagicMock(return_value=model_output)

    predictor.tokenizer = tokenizer
    predictor.model = model
    predictor.device = torch.device('cpu')
    return predictor


def test_predict_labels_high_valid_logit_as_valid():
    predictor = _build_predictor_with_mocked_model([-3.0, 4.0])  # class 1 (valid) dominant

    result = predictor.predict('Some hypothesis statement.')

    assert result['label'] == 1
    assert result['verdict'] == 'valid'
    assert result['confidence'] > 0.9


def test_predict_labels_high_flawed_logit_as_flawed():
    predictor = _build_predictor_with_mocked_model([4.0, -3.0])  # class 0 (flawed) dominant

    result = predictor.predict('Some hypothesis statement.')

    assert result['label'] == 0
    assert result['verdict'] == 'flawed'


def test_predict_batch_scores_each_hypothesis():
    predictor = _build_predictor_with_mocked_model([-3.0, 4.0])

    results = predictor.predict_batch(['h1', 'h2', 'h3'])

    assert len(results) == 3
    assert all(r['verdict'] == 'valid' for r in results)
