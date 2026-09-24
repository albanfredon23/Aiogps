import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableBeamSearch(nn.Module):
    """
    Recherche en faisceau différentiable utilisant Gumbel-Softmax.
    Support multi-mode : entraînement (stochastique) vs inférence (déterministe).
    """
    
    def __init__(self, emb_dim: int, beam_width: int = 4):
        """
        Args:
            emb_dim: Dimension des embeddings
            beam_width: Largeur du faisceau (nombre de trajectoires à conserver)
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.beam_width = beam_width
        self.score_head = nn.Linear(emb_dim, 1)
    
    def forward(self, candidate_nodes: torch.Tensor):
        """
        Sélection différentiable des meilleures trajectoires.
        
        Args:
            candidate_nodes: (batch_size, num_nodes, emb_dim) - Candidats
        
        Returns:
            trajectories: (batch_size, beam_width, emb_dim) - Trajectoires sélectionnées
            scores: (batch_size, beam_width) - Scores normalisés
        """
        batch_size, num_nodes, emb_dim = candidate_nodes.shape
        
        # Calcul des scores pour chaque nœud
        logits = self.score_head(candidate_nodes).squeeze(-1)  # (batch_size, num_nodes)
        
        if self.training:
            # Mode entraînement : Gumbel-Softmax + Straight-Through Estimator
            weights = F.gumbel_softmax(logits, tau=1.0, hard=False)  # Soft
            
            # Sélection top-k avec probabilités
            k = min(self.beam_width, num_nodes)
            topk_weights, topk_indices = torch.topk(weights, k=k, dim=-1)
            
            # Reconstruction des trajectoires
            idx_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, emb_dim)
            trajectories = torch.gather(candidate_nodes, 1, idx_expanded)
            scores = topk_weights
        else:
            # Mode inférence : Top-k déterministe
            k = min(self.beam_width, num_nodes)
            topk_scores, topk_indices = torch.topk(logits, k=k, dim=-1)
            
            # Reconstruction des trajectoires
            idx_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, emb_dim)
            trajectories = torch.gather(candidate_nodes, 1, idx_expanded)
            
            # Normalisation des scores
            scores = F.softmax(topk_scores, dim=-1)
        
        return trajectories, scores
    
    def estimate_trajectory_quality(self, trajectories: torch.Tensor) -> torch.Tensor:
        """Estime la qualité moyenne des trajectoires sélectionnées."""
        scores = self.score_head(trajectories).squeeze(-1)
        return scores.mean(dim=-1)
