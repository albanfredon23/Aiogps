import time
import torch
import torch.nn as nn
from core.aiotech44_core import AIOTECH44_EnergyCore


class BaselineDense(nn.Module):
    """Modèle dense statique de référence (calcul uniforme)."""
    def __init__(self, emb_dim: int, num_nodes: int):
        super().__init__()
        self.head = nn.Linear(emb_dim * 2, num_nodes)

    def forward(self, query: torch.Tensor, docs: torch.Tensor, graph: torch.Tensor):
        fused = torch.cat([query, docs.mean(dim=1)], dim=-1)
        return self.head(fused)


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Exécution du benchmark matériel sur : {device}")

    emb_dim, num_nodes, seq_len, batch_size = 256, 50, 20, 4

    baseline = BaselineDense(emb_dim, num_nodes).to(device)
    aiotech = AIOTECH44_EnergyCore(emb_dim=emb_dim, num_nodes=num_nodes).to(device)
    baseline.eval()
    aiotech.eval()

    q = torch.randn(batch_size, emb_dim, device=device)
    docs = torch.randn(batch_size, seq_len, emb_dim, device=device)
    graph = torch.randn(batch_size, num_nodes, emb_dim, device=device)
    c = torch.randn(batch_size, emb_dim, device=device)

    # Benchmark Baseline
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(50):
            _ = baseline(q, docs, graph)
    t_base = (time.perf_counter() - t0) * 1000 / 50

    # Benchmark AIOTECH
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(50):
            out = aiotech(q, docs, graph, c)
    t_aio = (time.perf_counter() - t0) * 1000 / 50

    reduction = (1.0 - (out["allocated_tokens"].item() / seq_len)) * 100.0

    print("=" * 60)
    print(" RÉSULTATS DU BENCHMARK COMPARATIF")
    print("=" * 60)
    print(f"Latence moyenne Baseline : {t_base:.2f} ms")
    print(f"Latence moyenne AIOTECH  : {t_aio:.2f} ms")
    print(f"Budget moyen alloué      : {out['allocated_tokens'].item():.1f} / {seq_len} fenêtres")
    print(f"Économie contextuelle    : {reduction:.1f} %")
    print("=" * 60)


if __name__ == "__main__":
    run()
