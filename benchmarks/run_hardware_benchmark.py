import time
import torch
import torch.nn as nn
from core.aiotech44_core import AIOTECH44_EnergyCore


class BaselineDense(nn.Module):
    """Modèle dense statique de référence sans calcul adaptatif."""
    def __init__(self, emb_dim: int, num_nodes: int):
        super().__init__()
        self.head = nn.Linear(emb_dim * 2, num_nodes)

    def forward(self, query: torch.Tensor, docs: torch.Tensor, graph: torch.Tensor):
        fused = torch.cat([query, docs.mean(dim=1)], dim=-1)
        return self.head(fused)


def run_benchmark(warmup: int = 10, iterations: int = 50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Exécution du benchmark matériel sur périphérique : {device}")

    emb_dim, num_nodes, seq_len, batch_size = 256, 50, 20, 4

    baseline = BaselineDense(emb_dim, num_nodes).to(device)
    aiotech = AIOTECH44_EnergyCore(emb_dim=emb_dim, num_nodes=num_nodes, num_agents=4).to(device)
    baseline.eval()
    aiotech.eval()

    q = torch.randn(batch_size, emb_dim, device=device)
    docs = torch.randn(batch_size, seq_len, emb_dim, device=device)
    graph = torch.randn(batch_size, num_nodes, emb_dim, device=device)
    c = torch.randn(batch_size, emb_dim, device=device)

    # Phase de chauffe (Warmup)
    with torch.no_grad():
        for _ in range(warmup):
            _ = baseline(q, docs, graph)
            _ = aiotech(q, docs, graph, c)

    # 1. Mesure Baseline
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            _ = baseline(q, docs, graph)
    t_base = (time.perf_counter() - t0) * 1000.0 / iterations

    # 2. Mesure AIOTECH
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            out = aiotech(q, docs, graph, c)
    t_aio = (time.perf_counter() - t0) * 1000.0 / iterations

    # Extraction sécurisée des scalaires (sans appel .item() sur des types float natifs)
    allocated_tokens = float(out["allocated_tokens"])
    memory_ratio = float(out["allocated_memory_ratio"])
    token_economy = max(0.0, (1.0 - (allocated_tokens / seq_len)) * 100.0)

    print("\n" + "=" * 60)
    print(" RÉSULTATS DU BENCHMARK MATÉRIEL")
    print("=" * 60)
    print(f"Latence moyenne Baseline : {t_base:6.2f} ms")
    print(f"Latence moyenne AIOTECH  : {t_aio:6.2f} ms")
    print(f"Budget tokens moyen      : {allocated_tokens:6.1f} / {seq_len}")
    print(f"Ratio de contexte actif  : {memory_ratio:6.1f} %")
    print(f"Économie contextuelle    : {token_economy:6.1f} %")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()
