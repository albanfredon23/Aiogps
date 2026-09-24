import torch
from aiotech_position.core import AIOTECH16DCore
from simulation.tunnel_monte_carlo import run_tunnel_benchmark
from simulation.spoofing_benchmark import run_spoofing_benchmark


def smoke_test_core():
    print("\n[1/3] Exécution du Smoke Test unitaire sur l'état 16D...")
    core = AIOTECH16DCore(dt=0.05)

    f_b = torch.tensor([0.1, 0.05, 9.85], dtype=torch.float64)
    omega_b = torch.tensor([0.01, -0.01, 0.02], dtype=torch.float64)

    state_pred = core.predict(f_b, omega_b, enable_scg=True)
    assert state_pred.shape[0] == 16, f"Dimension d'état attendue 16, reçue {state_pred.shape[0]}"

    pos_clean = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    accepted, gamma = core.update_gnss(pos_clean)
    assert accepted, "La mesure propre aurait dû être validée par le test Chi-deux."

    print("✓ Smoke Test validé sans erreur d'interface.")


def main():
    print("=" * 65)
    print("   AIOTECH-POSITION : SYSTÈME DE NAVIGATION SOUS CONTRAINTES")
    print("=" * 65)

    smoke_test_core()
    run_tunnel_benchmark(n_runs=15)
    run_spoofing_benchmark(n_runs=15, spoof_offset=25.0)

    print("\n" + "=" * 65)
    print("   TOUTES LES VALIDATIONS DU DÉPÔT SE SONT ACHEVÉES AVEC SUCCÈS")
    print("=" * 65)


if __name__ == "__main__":
    main()
