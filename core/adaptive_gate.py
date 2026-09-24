import torch
import torch.nn as nn


class AdaptiveComputeGate(nn.Module):
    """
    Modulation adaptative de l'effort de calcul.
    
    Évalue la complexité intrinsèque de la requête et ajuste
    l'activation des nœuds du graphe en conséquence.
    """
    
    def __init__(self, emb_dim: int):
        super().__init__()
        self.emb_dim = emb_dim
        self.complexity_evaluator = nn.Linear(emb_dim, 1)
        self.graph_gate = nn.Linear(emb_dim, emb_dim)
    
    def forward(self, query_emb: torch.Tensor, graph_nodes: torch.Tensor):
        """
        Args:
            query_emb: (batch_size, emb_dim) - Embedding de la requête
            graph_nodes: (batch_size, num_nodes, emb_dim) - Nœuds du graphe
        
        Returns:
            gated_nodes: (batch_size, num_nodes, emb_dim) - Nœuds filtrés
            complexity_score: (batch_size, 1) - Score de complexité [0, 1]
        """
        # Évaluation de la complexité
        complexity_logits = self.complexity_evaluator(query_emb)
        complexity_score = torch.sigmoid(complexity_logits)
        
        # Gating appliqué à chaque nœud
        gate_activation = torch.sigmoid(self.graph_gate(query_emb))
        
        # Application du gate
        gated_nodes = graph_nodes * gate_activation.unsqueeze(1)
        
        return gated_nodes, complexity_score
    
    def estimate_compute_budget(self, complexity_score: torch.Tensor, max_flops: float = 100.0) -> torch.Tensor:
        """Estime le budget de calcul en fonction de la complexité."""
        min_flops = max_flops * 0.3  # Min 30% de calcul
        estimated_flops = min_flops + complexity_score * (max_flops - min_flops)
        return estimated_flops
