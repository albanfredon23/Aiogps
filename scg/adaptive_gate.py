import torch
import torch.nn as nn


class AdaptiveComputeGate(nn.Module):
    """
    Évalue la complexité de la requête et module l'activation des nœuds de graphe.
    """
    def __init__(self, emb_dim: int):
        super().__init__()
        self.complexity_evaluator = nn.Linear(emb_dim, 1)
        self.graph_gate = nn.Linear(emb_dim, emb_dim)

    def forward(self, query_emb: torch.Tensor, graph_nodes: torch.Tensor):
        complexity_score = torch.sigmoid(self.complexity_evaluator(query_emb))
        gate_activation = torch.sigmoid(self.graph_gate(query_emb))
        gated_nodes = graph_nodes * gate_activation.unsqueeze(1)
        return gated_nodes, complexity_score
