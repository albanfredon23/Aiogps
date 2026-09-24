from typing import Tuple
import torch
import torch.nn as nn


class SCGEnergyPruner(nn.Module):
    """
    Garde-fou géométrique sphérique (SCG).
    Pénalise continûment les violations à l'entraînement et applique un élagage franc en inférence.
    """
    def __init__(self, lam: float = 0.2, energy_threshold: float = 0.5, temperature: float = 0.1):
        super().__init__()
        self.lam = lam
        self.energy_threshold = energy_threshold
        self.temperature = temperature

    def forward(
        self, trajectories: torch.Tensor, constraints: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Normalisation unitaire sur la sphère S^(D-1)
        norm_trajectories = torch.nn.functional.normalize(trajectories, p=2, dim=-1)
        norm_constraints = torch.nn.functional.normalize(constraints, p=2, dim=-1).unsqueeze(1)

        # Violation angulaire le long de la trajectoire
        violations = torch.relu(-norm_trajectories * norm_constraints).sum(dim=-1)
        scg_scores = self.lam * violations

        if self.training:
            active_mask = torch.sigmoid((self.energy_threshold - scg_scores) / self.temperature)
            surviving_trajectories = trajectories * active_mask.unsqueeze(-1)
            scg_loss = scg_scores.mean()
        else:
            active_mask = (scg_scores < self.energy_threshold).float()
            surviving_trajectories = trajectories * active_mask.unsqueeze(-1)
            scg_loss = torch.tensor(0.0, device=trajectories.device)

        return surviving_trajectories, active_mask, scg_loss

    def prune_trajectories(
        self, trajectories: torch.Tensor, constraints: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        surviving, mask, _ = self.forward(trajectories, constraints)
        return surviving, mask
