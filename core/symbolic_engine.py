import torch
import torch.nn as nn


class GodelLogic(nn.Module):
    """
    Implémentation de la logique de Gödel (fuzzy logic).
    Utilisée pour combiner des évaluations booléennes floues.
    """
    
    def __init__(self, tau: float = 0.1):
        """
        Args:
            tau: Température de lissage (plus petit = plus dur)
        """
        super().__init__()
        self.tau = tau
    
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Conjonction floue de Gödel : min(a, b) approximée.
        Utilise l'approximation log-sum-exp pour stabilité numérique.
        
        Args:
            a: (*, ) - Valeur floue A
            b: (*, ) - Valeur floue B
        
        Returns:
            result: (*, ) - Conjonction a ∧ b en logique floue
        """
        # min(a, b) ≈ -tau * log(exp(-a/tau) + exp(-b/tau))
        return -self.tau * torch.logsumexp(
            torch.stack([-a / self.tau, -b / self.tau], dim=0), 
            dim=0
        )


class NeuroSymbolicEngine(nn.Module):
    """
    Moteur neuro-symbolique pour évaluation de règles et contraintes.
    Combine apprentissage neural avec raisonnement symbolique.
    """
    
    def __init__(self, emb_dim: int, num_rules: int = 16):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_rules = num_rules
        
        # Évaluateur de règles
        self.rule_evaluator = nn.Linear(emb_dim, num_rules)
        
        # Opérateur logique de Gödel
        self.godel = GodelLogic(tau=0.1)
    
    def forward(self, context_emb: torch.Tensor) -> torch.Tensor:
        """
        Évalue les règles symboliques sur le contexte.
        
        Args:
            context_emb: (batch_size, emb_dim) - Embedding du contexte
        
        Returns:
            rule_scores: (batch_size, num_rules) - Score de chaque règle [0, 1]
        """
        logits = self.rule_evaluator(context_emb)
        rule_scores = torch.sigmoid(logits)
        return rule_scores
    
    def evaluate_conjunction(self, rule_scores: torch.Tensor) -> torch.Tensor:
        """
        Évalue la conjonction de plusieurs règles avec logique de Gödel.
        
        Args:
            rule_scores: (batch_size, num_rules) - Score de chaque règle
        
        Returns:
            conjunction: (batch_size, ) - Conjonction globale
        """
        # Conjonction itérative : min_i(rule_i)
        result = rule_scores[:, 0]
        for i in range(1, rule_scores.shape[1]):
            result = self.godel(result, rule_scores[:, i])
        return result
