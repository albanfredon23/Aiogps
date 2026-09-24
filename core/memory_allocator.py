import torch
import torch.nn as nn


class DifferentialMemoryAllocator(nn.Module):
    """
    Allocateur mémoire adaptatif basé sur la qualité des trajectoires.
    Réduit la fenêtre de contexte injectée en fonction du score de confiance.
    """
    
    def __init__(self, emb_dim: int, base_k: int = 2, max_k: int = 15):
        """
        Args:
            emb_dim: Dimension des embeddings
            base_k: Minimum de tokens de contexte à conserver
            max_k: Maximum de tokens de contexte
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.base_k = base_k
        self.max_k = max_k
        
        # Gate pour modulation du budget
        self.gate = nn.Linear(emb_dim, 1)
    
    def forward(self, trajectory_scores: torch.Tensor, retrieved_docs: torch.Tensor):
        """
        Alloue un budget de contexte adaptatif.
        
        Args:
            trajectory_scores: (batch_size, beam_width) ou (batch_size, beam_width, 1)
                Score de confiance des trajectoires
            retrieved_docs: (batch_size, num_docs, emb_dim)
                Documents disponibles
        
        Returns:
            allocated_memory: (batch_size, k_max, emb_dim) - Contexte alloué
            budget_k: (batch_size, ) - Budget effectif k par exemple
            padding_mask: (batch_size, k_max) - Masque de padding
        """
        batch_size, num_docs, emb_dim = retrieved_docs.shape
        
        # Normalisation du score
        score_scalar = trajectory_scores.mean(dim=-1, keepdim=True)
        if score_scalar.dim() > 2:
            score_scalar = score_scalar.squeeze(-1)
        
        # Modulation du budget en fonction du score
        norm_score = torch.sigmoid(score_scalar).squeeze(-1)  # [0, 1]
        
        # Budget adaptatif : base_k -> max_k en fonction du score
        dynamic_k = (self.base_k + norm_score * (self.max_k - self.base_k)).long()
        dynamic_k = torch.clamp(dynamic_k, min=self.base_k, max=min(self.max_k, num_docs))
        
        # Sélection des documents jusqu'au budget max
        k_max_batch = int(dynamic_k.max().item())
        sliced_context = retrieved_docs[:, :k_max_batch, :]
        
        # Masque de padding pour chaque exemple
        range_tensor = torch.arange(k_max_batch, device=retrieved_docs.device).unsqueeze(0).expand(batch_size, -1)
        padding_mask = range_tensor < dynamic_k.unsqueeze(-1)
        
        # Application du masque
        sliced_context = sliced_context * padding_mask.unsqueeze(-1).float()
        
        return sliced_context, dynamic_k, padding_mask
    
    def estimate_memory_savings(self, budget_k: torch.Tensor, num_docs: int) -> torch.Tensor:
        """
        Estime l'économie mémoire réalisée.
        
        Args:
            budget_k: (batch_size, ) - Budget alloué
            num_docs: Nombre total de documents
        
        Returns:
            savings_ratio: (batch_size, ) - Ratio de compression
        """
        return 1.0 - (budget_k.float() / num_docs)
