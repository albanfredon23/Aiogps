import torch
import pytest
from core.aiotech44_core import AIOTECH44_EnergyCore


def test_aiotech44_smoke_and_shapes():
    """Vérifie la cohérence des dimensions et la présence de toutes les clés de sortie."""
    emb_dim = 128
    num_nodes = 30
    batch_size = 2
    seq_len = 10

    core = AIOTECH44_EnergyCore(emb_dim=emb_dim, num_nodes=num_nodes, num_agents=4)
    core.eval()

    query = torch.randn(batch_size, emb_dim)
    docs = torch.randn(batch_size, seq_len, emb_dim)
    graph = torch.randn(batch_size, num_nodes, emb_dim)
    constraints = torch.randn(batch_size, emb_dim)

    with torch.no_grad():
        out = core(query, docs, graph, constraints)

    # 1. Vérification de la présence des clés d'interface
    assert "policy" in out
    assert "trajectories" in out
    assert "scg_loss" in out
    assert "budget_k" in out
    assert "allocated_tokens" in out
    assert "allocated_memory_ratio" in out
    assert "padding_mask" in out

    # 2. Vérification des dimensions
    assert out["policy"].shape == (batch_size, num_nodes)
    assert out["trajectories"].shape[0] == batch_size
    assert out["budget_k"].shape == (batch_size,)

    # 3. Contrôle des bornes du budget dynamique
    assert torch.all(out["budget_k"] <= 15)
    assert torch.all(out["budget_k"] >= 2)
    assert 0.0 <= float(out["allocated_memory_ratio"]) <= 100.0


def test_aiotech44_autograd_and_finiteness():
    """Valide la différentiabilité de bout en bout avec gradients finis et non nuls."""
    emb_dim = 128
    num_nodes = 30
    batch_size = 2
    seq_len = 10

    core = AIOTECH44_EnergyCore(emb_dim=emb_dim, num_nodes=num_nodes, num_agents=4)
    core.train()

    query = torch.randn(batch_size, emb_dim, requires_grad=True)
    docs = torch.randn(batch_size, seq_len, emb_dim, requires_grad=True)
    graph = torch.randn(batch_size, num_nodes, emb_dim, requires_grad=True)
    constraints = torch.randn(batch_size, emb_dim)

    out = core(query, docs, graph, constraints)

    # Fonction de coût combinée : tâche + pénalité géométrique SCG
    loss = out["policy"].sum() + out["scg_loss"]
    loss.backward()

    # Assertions strictes de différentiabilité et de non-nullité
    for tensor_name, grad in [
        ("query", query.grad),
        ("docs", docs.grad),
        ("graph", graph.grad),
        ("policy_head.weight", core.policy_head.weight.grad),
    ]:
        assert grad is not None, f"Gradient manquant pour {tensor_name}"
        assert torch.isfinite(grad).all(), f"Gradient non fini (NaN/Inf) pour {tensor_name}"
        assert grad.abs().sum() > 0, f"Gradient nul pour {tensor_name}"
