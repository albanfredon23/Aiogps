import torch
import torch.nn as nn
from .adaptive_gate import AdaptiveComputeGate
from .symbolic_engine import NeuroSymbolicEngine
from .beam_planner import DifferentiableBeamSearch
from .dynamic_allocator import DynamicAgentAllocator
from .memory_allocator import DifferentialMemoryAllocator


class AIOTECH44_EnergyCore(nn.Module):
    """
    Orchestrateur principal du middleware AIOTECH44.
    Coordonne tous les composants : gate adaptatif, TAP-NN, SCG, 
    allocation mémoire et allocateur d'agents.
    """
    
    def __init__(self, emb_dim: int = 256, num_nodes: int = 50, num_agents: int = 4):
        """
        Args:
            emb_dim: Dimension des embeddings (par défaut 256)
            num_nodes: Nombre de nœuds du graphe
            num_agents: Nombre d'agents spécialisés
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.num_nodes = num_nodes
        
        # Composants du pipeline
        self.adaptive_gate = AdaptiveComputeGate(emb_dim)
        self.symbolic_engine = NeuroSymbolicEngine(emb_dim, num_rules=16)
        self.beam_planner = DifferentiableBeamSearch(emb_dim, beam_width=4)
        self.memory_allocator = DifferentialMemoryAllocator(emb_dim, base_k=2, max_k=15)
        self.agent_allocator = DynamicAgentAllocator(emb_dim, num_agents=num_agents)
        
        # Tête de décision politique
        # Entrée : contexte du graphe (emb_dim) + contexte trajectoire (emb_dim) + résumé mémoire (emb_dim)
        self.policy_head = nn.Linear(emb_dim * 3, num_nodes)
    
    def forward(self, query_emb: torch.Tensor, retrieved_docs: torch.Tensor, 
                graph_nodes: torch.Tensor, constraints: torch.Tensor) -> dict:
        """
        Exécution complète du pipeline AIOTECH44.
        
        Args:
            query_emb: (batch_size, emb_dim) - Embedding de la requête
            retrieved_docs: (batch_size, seq_len, emb_dim) - Documents récupérés
            graph_nodes: (batch_size, num_nodes, emb_dim) - Nœuds du graphe
            constraints: (batch_size, emb_dim) - Contraintes latentes (pour SCG)
        
        Returns:
            dict avec sorties de tous les composants
        """
        batch_size = query_emb.shape[0]
        
        # 1. Gate adaptatif - Module l'effort
        gated_nodes, complexity_score = self.adaptive_gate(query_emb, graph_nodes)
        
        # 2. Sélection différentiable (TAP-NN)
        trajectories, beam_scores = self.beam_planner(gated_nodes)
        
        # 3. Evaluation des règles symboliques
        rule_scores = self.symbolic_engine(query_emb)
        
        # 4. Allocation mémoire différentielle
        allocated_memory, budget_k, padding_mask = self.memory_allocator(beam_scores, retrieved_docs)
        
        # 5. Allocateur d'agents dynamique
        agent_weights = self.agent_allocator(query_emb)
        
        # 6. Synthèse des contextes pour la tête politique
        graph_context = gated_nodes.mean(dim=1)
        trajectory_context = trajectories.mean(dim=1)
        memory_summary = allocated_memory.mean(dim=1)
        
        # Fusion des contextes
        fused = torch.cat([graph_context, trajectory_context, memory_summary], dim=-1)
        policy_logits = self.policy_head(fused)
        
        # Métriques d'efficacité
        total_docs = retrieved_docs.size(1)
        allocated_tokens = budget_k.detach().float().mean().item()
        allocated_tokens_total = padding_mask.sum().detach().item()
        allocated_memory_ratio = (allocated_tokens / max(1, total_docs)) * 100.0
        
        # Retour complet
        return {
            "policy": policy_logits,                              # Décision politique
            "trajectories": trajectories,                         # Trajectoires sélectionnées
            "complexity_score": complexity_score,                 # Score de complexité
            "rule_scores": rule_scores,                           # Scores des règles (nouveau)
            "active_agents": agent_weights,                       # Poids des agents
            "budget_k": budget_k,                                 # Budget de contexte par exemple
            "allocated_tokens": allocated_tokens,                 # Budget moyen
            "allocated_tokens_total": allocated_tokens_total,     # Total tokens alloués
            "allocated_memory_ratio": allocated_memory_ratio,     # Ratio en %
            "padding_mask": padding_mask,                         # Masque de padding
            "agent_priors": self.agent_allocator.agent_priors     # Priors adaptatifs (nouveau)
        }
    
    def get_model_size(self) -> dict:
        """Retourne des infos sur la taille du modèle."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 ** 2)  # Approximation en float32
        }
