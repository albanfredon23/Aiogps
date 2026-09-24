import torch
from core.aiotech44_core import AIOTECH44_EnergyCore


def main():
    print("=" * 60)
    print("  AIOTECH44 : EXÉCUTION DU PIPELINE ADAPTATIF & TEST D'INTÉGRITÉ")
    print("=" * 60)

    emb_dim = 256
    num_nodes = 50
    seq_len = 20
    batch_size = 2

    # Instanciation du cœur sans modules orphelins
    core = AIOTECH44_EnergyCore(emb_dim=emb_dim, num_nodes=num_nodes, num_agents=4)

    # 1. Inférence nominale (Smoke test)
    core.eval()
    query = torch.randn(batch_size, emb_dim)
    docs = torch.randn(batch_size, seq_len, emb_dim)
    graph = torch.randn(batch_size, num_nodes, emb_dim)
    constraints = torch.randn(batch_size, emb_dim)

    with torch.no_grad():
        output = core(query, docs, graph, constraints)

    print("\n✓ Inférence exécutée sans erreur de dimension ni de déballage :")
    print(f" - Complexité évaluée     : {output['complexity_score'].mean().item():.3f}")
    print(f" - Forme de la politique  : {list(output['policy'].shape)}")
    print(f" - Budget moyen alloué    : {output['allocated_tokens'].item():.1f} fenêtres")
    print(f" - Ratio mémoire utilisé  : {output['allocated_memory_ratio'].item():.1f} %")
    print(f" - Perte géométrique SCG  : {output['scg_loss'].item():.4f}")

    # 2. Test de la boucle de régression continue
    print("\n[Test d'adaptation continue des agents]")
    priors_init = core.agent_allocator.agent_priors.clone()
    print(f" - Poids initiaux   : {priors_init.numpy().round(3).tolist()}")

    # Simulation d'un signal de gradient de perte
    simulated_loss = output["active_agents"].mean(dim=0) * 0.5
    core.agent_allocator.update_continuous_weights(simulated_loss)

    priors_updated = core.agent_allocator.agent_priors.clone()
    print(f" - Poids actualisés : {priors_updated.numpy().round(3).tolist()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
