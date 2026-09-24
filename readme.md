# AIOTECH44 : Middleware Neuro-Symbolique & Calcul Adaptatif (Green AI)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)

AIOTECH44 est un middleware cognitif conçu pour optimiser l'efficience énergétique des inférences de modèles de langage. Il remplace le calcul dense et systématique par une **exploration de trajectoires admissibles (TAP)**, un **élagage géométrique par contraintes sphériques (SCG)** et une **allocation de mémoire différentielle**.

---

## 🏛 Architecture & Flux d'Exécution

```text
               ┌───────────────────────────────┐
               │      Requête Utilisateur      │
               └──────────────┬────────────────┘
                              │
               ┌──────────────▼────────────────┐
               │  1. AdaptiveComputeGate       │
               │  (Modulation de l'effort)     │
               └──────────────┬────────────────┘
                              │
               ┌──────────────▼────────────────┐
               │  2. Moteur TAP-NN             │
               │  (Sélection Gumbel / Soft-Gödel)
               └──────────────┬────────────────┘
                              │
               ┌──────────────▼────────────────┐
               │  3. SCGEnergyPruner           │
               │  (Filtrage géométrique)       │
               └──────────────┬────────────────┘
                              │
               ┌──────────────▼────────────────┐
               │  4. DifferentialMemoryAllocator
               │  (Tronquage contextuel k)     │
               └──────────────┬────────────────┘
                              │
               ┌──────────────▼────────────────┐
               │  5. DynamicAgentAllocator     │
               │  (Mise à jour adaptative)     │
               └──────────────┬────────────────┘
                              │
               ┌──────────────▼────────────────┐
               │    Décision & Trace Auditable │
               └───────────────────────────────┘
