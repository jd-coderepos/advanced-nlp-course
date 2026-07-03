"""Small hand-checkable tests for the retrieval objective.

Run from the repository root:

    python -m unittest tests/test_objective.py
"""

import unittest

import torch

from src.objective import retrieval_hinge_loss


class RetrievalObjectiveTest(unittest.TestCase):
    def test_zero_loss_when_positive_is_better_by_margin(self):
        q = torch.tensor([[1.0, 0.0]])
        pos = torch.tensor([[1.0, 0.0]])
        neg = torch.tensor([[0.0, 1.0]])
        loss = retrieval_hinge_loss(q, pos, neg, margin=0.2)
        self.assertAlmostEqual(float(loss), 0.0, places=6)

    def test_positive_loss_when_negative_scores_higher(self):
        q = torch.tensor([[1.0, 0.0]])
        pos = torch.tensor([[0.0, 1.0]])
        neg = torch.tensor([[1.0, 0.0]])
        loss = retrieval_hinge_loss(q, pos, neg, margin=0.2)
        self.assertAlmostEqual(float(loss), 1.2, places=6)


if __name__ == "__main__":
    unittest.main()
