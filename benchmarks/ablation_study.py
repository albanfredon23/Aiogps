import time
from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn

from core.aiotech44_core import AIOTECH44_EnergyCore


class BaselineDense(nn.Module):
    """Modèle dense statique de référence (calcul uniforme sans élagage)."""
    def __init__(self, emb_dim: int, num_nodes: int):
        super().__init__()
        self.head = nn.Linear(emb_dim * 2, num_nodes)

    def forward(self, query: torch.Tensor, docs: torch.Tensor, graph: torch.Tensor):
        fused = torch.cat([query, docs.mean(dim=1)], dim=-1)
        return self.head(fused)


class AIOTECHWithoutSCG(AIOTECH44_EnergyCore):
    """Variante ablative désactivant le filtrage géométrique SCG."""
    def forward(
        self,
        query_emb: torch.Tensor,
        retrieved_docs: torch.Tensor,
        graph_nodes: torch.Tensor,
        constraints: torch.Tensor
    ) -> Dict[str, Any]:
        gated_nodes, complexity_score = self.adaptive_gate(query_emb, graph_nodes)
        trajectories, _ = self.beam_planner(gated_nodes)

        # Bypass SCG : aucune pénalité ni élagage, masque unitaire
        surviving_trajectories = trajectories
        active_mask = torch.ones(trajectories.shape[:2], device=trajectories.device)
        scg_loss = torch.tensor(0.0, device=trajectories.device)

        allocated_memory, budget_k, padding_mask = self.memory_allocator(active_mask, retrieved_docs)
        agent_weights = self.agent_allocator(query_emb)

        graph_context = gated_nodes.mean(dim=1)
        trajectory_context = surviving_trajectories.mean(dim=1)
        memory_summary = allocated_memory.mean(dim=1)

        fused_context = torch.cat([graph_context, trajectory_context, memory_summary], dim=-1)
        policy_logits = self.policy_head(fused_context)

        total_docs = retrieved_docs.size(1)
        allocated_tokens = budget_k.detach().float().mean()
        allocated_memory_ratio = (allocated_tokens / max(1, total_docs)) * 100.0

        return {
            "policy": policy_logits,
            "trajectories": surviving_trajectories,
            "active_mask": active_mask,
            "budget_k": budget_k,
            "allocated_tokens": allocated_tokens,
            "allocated_memory_ratio": allocated_memory_ratio,
            "scg_loss": scg_loss
        }


def run_ablation(iterations: int = 100, warmup: int = 15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Lancement de l'étude d'ablation SCG sur périphérique : {device}")

    emb_dim, num_nodes, seq_len, batch_size = 256, 50, 20, 4

    models = {
        "1. Baseline Dense": BaselineDense(emb_dim, num_nodes).to(device),
        "2. AIOTECH (Sans SCG)": AIOTECHWithoutSCG(emb_dim, num_nodes, num_agents=4).to(device),
        "3. AIOTECH (Complet)": AIOTECH44_EnergyCore(emb_dim, num_nodes, num_agents=4).to(device)
    }

    for m in models.values():
        m.eval()

    # Jeux d'entrée synthétiques avec contraintes adverses injectées
    q = torch.randn(batch_size, emb_dim, device=device)
    docs = torch.randn(batch_size, seq_len, emb_dim, device=device)
    graph = torch.randn(batch_size, num_nodes, emb_dim, device=device)
    # Contraintes polarisées pour induire des violations partielles mesurables
    c = torch.randn(batch_size, emb_dim, device=device)

    # Phase de chauffe
    with torch.no_grad():
        for _ in range(warmup):
            _ = models["1. Baseline Dense"](q, docs, graph)
            _ = models["2. AIOTECH (Sans SCG)"](q, docs, graph, c)
            _ = models["3. AIOTECH (Complet)"](q, docs, graph, c)

    metrics: Dict[str, Dict[str, float]] = {}

    for name, model in models.items():
        latencies = []
        tokens_alloc = []
        pruned_ratios = []

        with torch.no_grad():
            for _ in range(iterations):
                t0 = time.perf_counter()
                if isinstance(model, BaselineDense):
                    _ = model(q, docs, graph)
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    tok = float(seq_len)
                    pruned = 0.0
                else:
                    out = model(q, docs, graph, c)
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    tok = float(out["allocated_tokens"])
                    # Taux de trajectoires invalidées (actives < 0.5)
                    active = (out["trajectories"].abs().sum(dim=-1) > 1e-5).float()
                    pruned = float((1.0 - active.mean()) * 100.0)

                latencies.append(dt_ms)
                tokens_alloc.append(tok)
                pruned_ratios.append(pruned)

        metrics[name] = {
            "latency_mean": float(np.mean(latencies)),
            "latency_std": float(np.std(latencies)),
            "tokens_mean": float(np.mean(tokens_alloc)),
            "context_retention": float(np.mean(tokens_alloc) / seq_len * 100.0),
            "pruned_trajectories": float(np.mean(pruned_ratios))
        }

    # Restitution tabulaire
    print("\n" + "=" * 90)
    print(" RÉSULTATS COMPARATIFS DE L'ÉTUDE D'ABLATION (ISOLATION SCG)")
    print("=" * 90)
    print(f"{'Configuration':<25} | {'Latence (ms)':<16} | {'Budget Tokens':<15} | {'Rétention':<12} | {'Élagage SCG':<12}")
    print("-" * 90)

    for name, data in metrics.items():
        lat = f"{data['latency_mean']:5.2f} ± {data['latency_std']:4.2f}"
        tok = f"{data['tokens_mean']:4.1f} / {seq_len}"
        ret = f"{data['context_retention']:5.1f} %"
        prun = f"{data['pruned_trajectories']:5.1f} %"
        print(f"{name:<25} | {lat:<16} | {tok:<15} | {ret:<12} | {prun:<12}")

    print("=" * 90)
    
    # Interprétation différentielle
    delta_pruning = metrics["3. AIOTECH (Complet)"]["pruned_trajectories"]
    lat_diff = (
        (metrics["3. AIOTECH (Complet)"]["latency_mean"] - metrics["2. AIOTECH (Sans SCG)"]["latency_mean"])
        / metrics["2. AIOTECH (Sans SCG)"]["latency_mean"] * 100.0
    )
    print(f"\n[Diagnostic SCG]")
    print(f" • Taux d'élagage géométrique direct : {delta_pruning:.1f} % des trajectoires aberrantes éliminées.")
    print(f" • Surcoût temporel du filtre SCG   : {lat_diff:+.1f} % par rapport à la version sans contraintes.")


if __name__ == "__main__":
    run_ablation()
