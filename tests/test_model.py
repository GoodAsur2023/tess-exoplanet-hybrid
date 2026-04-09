"""
tests/test_model.py — Smoke tests for the model and utilities.
Run in CI without data — verifies shapes, loss behaviour, and metrics.
"""
import numpy as np
import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import BaselineCNN, DualViewTransformer, count_parameters
from src.utils import FocalLoss, compute_metrics, find_best_threshold

BATCH      = 8
GLOBAL_LEN = 2048
LOCAL_LEN  = 201


@pytest.fixture
def dummy_batch():
    gv     = torch.randn(BATCH, GLOBAL_LEN)
    lv     = torch.randn(BATCH, LOCAL_LEN)
    labels = torch.randint(0, 2, (BATCH,)).float()
    return gv, lv, labels


class TestDualViewTransformer:
    def test_output_shape(self, dummy_batch):
        model = DualViewTransformer(global_length=GLOBAL_LEN, local_length=LOCAL_LEN)
        model.eval()
        gv, lv, _ = dummy_batch
        with torch.no_grad():
            logits, tokens = model(gv, lv)
        assert logits.shape == (BATCH,)
        assert tokens.ndim == 3
        assert tokens.shape[0] == BATCH

    def test_parameter_count(self):
        model = DualViewTransformer()
        n = count_parameters(model)
        assert 1_000_000 < n < 20_000_000, f"Unexpected param count: {n:,}"

    def test_gradient_flow(self, dummy_batch):
        model = DualViewTransformer(global_length=GLOBAL_LEN, local_length=LOCAL_LEN)
        gv, lv, labels = dummy_batch
        criterion = FocalLoss()
        logits, _ = model(gv, lv)
        loss = criterion(logits, labels)
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad, "No gradients flowed through the model"


class TestBaselineCNN:
    def test_output_shape(self, dummy_batch):
        model = BaselineCNN(global_length=GLOBAL_LEN, local_length=LOCAL_LEN)
        model.eval()
        gv, lv, _ = dummy_batch
        with torch.no_grad():
            logits, tokens = model(gv, lv)
        assert logits.shape == (BATCH,)
        assert tokens is None


class TestFocalLoss:
    def test_reduces_to_bce_at_gamma_zero(self):
    	torch.manual_seed(0)
    	logits  = torch.randn(32)
    	targets = torch.randint(0, 2, (32,)).float()
    	fl  = FocalLoss(alpha=0.5, gamma=0.0)
    	bce = torch.nn.BCEWithLogitsLoss()
    	# With gamma=0 and alpha=0.5, alpha_t=0.5 for every sample,
    	# so FocalLoss = 0.5 * BCE exactly
    	assert abs(fl(logits, targets).item() - 0.5 * bce(logits, targets).item()) < 1e-4

    def test_focal_weights_downweight_easy_examples(self):
        logits_hard = torch.tensor([0.0])
        logits_easy = torch.tensor([-10.0])
        targets     = torch.tensor([0.0])
        fl = FocalLoss(alpha=0.5, gamma=2.0)
        assert fl(logits_hard, targets).item() > fl(logits_easy, targets).item()


class TestMetrics:
    def test_compute_metrics_perfect(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        m = compute_metrics(y_true, y_prob)
        assert m["roc_auc"] == 1.0

    def test_find_best_threshold(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        thresh = find_best_threshold(y_true, y_prob, metric="f1")
        assert 0.1 <= thresh <= 0.9