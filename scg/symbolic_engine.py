import torch
import torch.nn as nn


class GodelLogic(nn.Module):
    """
    Approximation différentiable de la t-norme de Gödel via Smooth-Min (Log-Sum-Exp).
    """
    def __init__(self, tau: float = 0.1):
        super().__init__()
        self.tau = tau

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return -self.tau * torch.logsumexp(torch.stack([-a / self.tau, -b / self.tau], dim=0), dim=0)


class NeuroSymbolicEngine(nn.Module):
    """
    Évaluation d'admissibilité logique par règles latentes.
    """
    def __init__(self, emb_dim: int, num_rules: int = 16):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_rules = num_rules
        self.rule_embeddings = nn.Parameter(torch.randn(num_rules, emb_dim))
        self.rule_evaluator = nn.Linear(emb_dim, num_rules)
        self.godel = GodelLogic()

    def forward(self, context_emb: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.rule_evaluator(context_emb))
