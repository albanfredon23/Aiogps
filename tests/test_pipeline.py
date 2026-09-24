import torch
from core.aiotech44_core import AIOTECH44_EnergyCore


def test_pipeline_integration_and_gradients():
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

    outputs = core(query, docs, graph, constraints)

    # Vérification des clés indispensables
    assert "policy" in outputs
    assert "budget_k" in outputs
    assert "allocated_tokens" in outputs
    assert "allocated_memory_ratio" in outputs
    assert "scg_loss" in outputs

    # Vérification des bornes
    assert torch.all(outputs["budget_k"] <= 15)
    assert torch.all(outputs["budget_k"] >= 2)
    assert 0.0 <= outputs["allocated_memory_ratio"].item() <= 100.0

    # Rétropropagation de bout en bout
    loss = outputs["policy"].sum() + outputs["scg_loss"]
    loss.backward()

    # Vérification stricte des gradients
    assert query.grad is not None
    assert torch.isfinite(query.grad).all()
    assert query.grad.abs().sum() > 0

    assert docs.grad is not None
    assert torch.isfinite(docs.grad).all()
    assert docs.grad.abs().sum() > 0

    assert core.policy_head.weight.grad is not None
    assert torch.isfinite(core.policy_head.weight.grad).all()
