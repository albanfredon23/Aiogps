import torch
import numpy as np
from core.aiotech44_core import AIOTECH44_EnergyCore
from scg.pruner import SCGEnergyPruner


def run_adversarial_scg_test():
    print("=" * 70)
    print(" BENCHMARK ADVERSARIAL SCG : VALIDATION DU FILTRAGE GÉOMÉTRIQUE")
    print("=" * 70)

    emb_dim = 256
    batch_size = 4
    beam_width = 4

    # Calibrage corrigé : lambda = 1.0, seuil de violation à 0.1 (rejette si dot < -0.1)
    pruner = SCGEnergyPruner(lam=1.0, energy_threshold=0.1)
    pruner.eval()

    # Définition d'un vecteur de contrainte de référence
    constraints = torch.randn(batch_size, emb_dim)
    constraints = torch.nn.functional.normalize(constraints, p=2, dim=-1)

    # 4 faisceaux de test par batch :
    # 0: Alignée (dot ≈ +1.0) -> Valide
    # 1: Neutre/Orthogonale (dot ≈ 0.0) -> Tolérée
    # 2: Déviante légère (dot ≈ -0.3) -> Non admissible
    # 3: Directement opposée (dot ≈ -1.0) -> Violation critique
    c_exp = constraints.unsqueeze(1)
    noise = torch.randn(batch_size, beam_width, emb_dim) * 0.05

    t_aligned = c_exp + noise[:, 0:1, :]
    t_orthogonal = torch.randn(batch_size, 1, emb_dim)
    t_orthogonal -= (t_orthogonal * c_exp).sum(dim=-1, keepdim=True) * c_exp
    t_deviant = -0.3 * c_exp + noise[:, 2:3, :]
    t_opposite = -1.0 * c_exp + noise[:, 3:4, :]

    trajectories = torch.cat([t_aligned, t_orthogonal, t_deviant, t_opposite], dim=1)

    with torch.no_grad():
        surviving, active_mask, _ = pruner(trajectories, constraints)

    # Analyse des résultats
    print(f"\nProfil des 4 trajectoires testées par échantillon :")
    labels = ["Alignée (+1.0)", "Orthogonale (0.0)", "Déviante (-0.3)", "Opposée (-1.0)"]
    retention_per_type = active_mask.mean(dim=0).numpy()

    for idx, (label, ret) in enumerate(zip(labels, retention_per_type)):
        status = "CONSERVÉE" if ret > 0.5 else "ÉLAGUÉE"
        print(f" • Trajectoire {idx} [{label:<18}] : Taux survie = {ret*100:5.1f} % -> {status}")

    pruned_total = (1.0 - active_mask.mean().item()) * 100.0
    print("-" * 70)
    print(f"Taux d'élagage global mesuré : {pruned_total:.1f} %")
    print(f"Comportement attendu        : 50.0 % (trajectoires déviante et opposée éliminées)")
    print("=" * 70)


if __name__ == "__main__":
    run_adversarial_scg_test()
