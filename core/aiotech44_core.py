from typing import Dict, Any
import torch
import torch.nn as nn

from .adaptive_gate import AdaptiveComputeGate
from .symbolic_engine import NeuroSymbolicEngine
from .beam_planner import DifferentiableBeamSearch
from .dynamic_allocator import DynamicAgentAllocator
from .memory_allocator import DifferentialMemoryAllocator
from scg.pruner import SCGEnergyPruner


class AIOTECH44_EnergyCore(nn.Module):
    """
    Orchestrateur central AIOTECH44.
    Assure le couplage causal Mémoire -> Décision et l'élagage géométrique SCG.
    """
    def __init__(self, emb_dim: int = 256, num_nodes: int = 50, num_agents: int = 4):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_nodes = num_nodes
        self.num_agents = num_agents

        self.adaptive_gate = AdaptiveComputeGate(emb_dim)
        self.symbolic_engine = NeuroSymbolicEngine(emb_dim)
        self.beam_planner = DifferentiableBeamSearch(emb_dim, beam_width=4)
        self.scg_pruner = SCGEnergyPruner()
        self.memory_allocator = DifferentialMemoryAllocator(emb_dim)
        self.agent_allocator = DynamicAgentAllocator(emb_dim=emb_dim, num_agents=num_agents)

        # policy_head : graph_context (emb_dim) + trajectory_context (emb_dim) + memory_summary (emb_dim)
        self.policy_head = nn.Linear(emb_dim * 3, num_nodes)

    def forward(
        self,
        query_emb: torch.Tensor,
        retrieved_docs: torch.Tensor,
        graph_nodes: torch.Tensor,
        constraints: torch.Tensor
    ) -> Dict[str, Any]:
        # 1. Gate adaptatif
        gated_nodes, complexity_score = self.adaptive_gate(query_emb, graph_nodes)

        # 2. Planification de trajectoires
        trajectories, beam_scores = self.beam_planner(gated_nodes)

        # 3. Élagage SCG (3 sorties synchronisées)
        surviving_trajectories, active_mask, scg_loss = self.scg_pruner(trajectories, constraints)

        # 4. Allocation mémoire différentielle (déballage des 3 retours)
        allocated_memory, budget_k, padding_mask = self.memory_allocator(active_mask, retrieved_docs)

        # 5. Sélection et pondération des agents
        agent_weights = self.agent_allocator(query_emb)

        # 6. Agrégation causale et décision
        graph_context = gated_nodes.mean(dim=1)
        trajectory_context = surviving_trajectories.mean(dim=1)
        memory_summary = allocated_memory.mean(dim=1)

        fused_context = torch.cat([graph_context, trajectory_context, memory_summary], dim=-1)
        policy_logits = self.policy_head(fused_context)

        total_docs = retrieved_docs.size(1)
        allocated_tokens = budget_k.detach().float().mean()
        allocated_memory_ratio = (allocated_tokens / max(1, total_docs)) * 100.0
        allocated_tokens_total = padding_mask.sum().detach()

        return {
            "policy": policy_logits,
            "trajectories": surviving_trajectories,
            "complexity_score": complexity_score,
            "active_agents": agent_weights,
            "budget_k": budget_k,
            "allocated_tokens": allocated_tokens,
            "allocated_tokens_total": allocated_tokens_total,
            "allocated_memory_ratio": allocated_memory_ratio,
            "padding_mask": padding_mask,
            "scg_loss": scg_loss
        }
