# Aiogps · Le GPS du raisonnement

[![Site](https://img.shields.io/badge/site-albanfredon23.github.io%2FAiogps-34d399.svg)](https://albanfredon23.github.io/Aiogps/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)

🌐 **Site de présentation 3D : [albanfredon23.github.io/Aiogps](https://albanfredon23.github.io/Aiogps/)**
(pipeline en scène Three.js, démo interactive, résultats mesurés ; sources dans [`site/`](site/)).

**Chaque requête prend la trajectoire la plus sobre.** Aiogps, propulsé par le moteur **AIOTECH44**, est un
middleware neuro-symbolique Green AI. Il se place autour de votre modèle de langage et décide, pour chaque
requête, de l'effort à fournir, des pistes à explorer, du contexte à injecter et des agents à solliciter. Une
question simple ne coûte plus le prix d'une démonstration.

| Paramètre du code | Valeur |
|---|---|
| Effort de calcul pour une requête jugée simple | 30 % (100 % pour un raisonnement) |
| Documents de contexte retenus | 2 à 15, au lieu de la fenêtre complète |
| Agents spécialisés pondérés à chaque requête | 4 |

---

## Le principe

Un modèle de langage dépense le même effort pour « Quels sont vos horaires ? » que pour analyser un contrat
clause par clause. Il relit toute sa fenêtre de contexte et explore ses pistes sans tenir compte des contraintes
à respecter.

Aiogps remplace ce calcul dense et systématique par une navigation : estimer la difficulté du trajet, ne tracer
que les itinéraires plausibles, écarter ceux qui sortent des règles, puis n'emporter que les documents utiles.
Comme un GPS, il ne roule pas à votre place : il choisit la route.

## Le pipeline AIOTECH44, en six étapes

```
Requête ─▶ 1. Porte adaptative ─▶ 2. Faisceau TAP ─▶ 3. Sphère SCG ─▶ 4. Mémoire différentielle ─▶ 5. Agents ─▶ 6. Décision + trace
```

1. **Porte de calcul adaptative** (`core/adaptive_gate.py`, `AdaptiveComputeGate`) : estime la complexité de la
   requête entre 0 et 1. Le budget de calcul suit cette estimation, de 30 % pour une recherche simple à 100 %
   pour un raisonnement, et la requête module l'activation de chaque nœud du graphe de connaissances.
2. **Faisceau de trajectoires TAP** (`core/beam_planner.py`, `DifferentiableBeamSearch`) : retient les k
   meilleures trajectoires parmi les nœuds activés. À l'entraînement, la sélection passe par Gumbel-Softmax
   pour garder des gradients stables ; en inférence, c'est un top-k déterministe, donc reproductible.
3. **Élagage géométrique SCG** (`scg/pruner.py`, `SCGEnergyPruner`) : projette trajectoires et contraintes sur
   la sphère unité et mesure la violation angulaire `relu(−t·c)`. À l'entraînement, elle devient une pénalité
   continue ; en inférence, toute trajectoire au-delà du seuil d'énergie est élaguée.
4. **Mémoire différentielle** (`core/memory_allocator.py`, `DifferentialMemoryAllocator`) : fixe un budget de
   contexte k entre 2 et 15 documents selon la confiance des trajectoires survivantes ; le reste de la fenêtre
   est masqué.
5. **Agents spécialisés** (`core/dynamic_allocator.py`, `DynamicAgentAllocator`) : pondère quatre agents,
   raisonnement, mathématiques, logique et critique. Leurs priors s'ajustent en continu à partir du signal de
   perte, sans réentraînement complet.
6. **Décision et trace auditable** (`core/aiotech44_core.py`, `AIOTECH44_EnergyCore`) : graphe filtré,
   trajectoires survivantes et mémoire allouée sont fusionnés en une décision. Chaque inférence renvoie sa
   trace : complexité, budget k, poids des agents, perte géométrique.

La logique continue de Gödel utilisée pour combiner les évaluations floues est dans `core/symbolic_engine.py`.

## Démo interactive

- **Dans le navigateur** : la [démo du site](https://albanfredon23.github.io/Aiogps/#demo) applique les règles
  du dépôt (bornes de la porte, seuil du SCG, budget de 2 à 15 documents, formule de calcul économisé) à la
  requête que vous saisissez, sans rien envoyer. La complexité y est estimée par une heuristique lisible.
- **Avec le vrai modèle PyTorch** : `streamlit run app.py`.

## Ce qui est mesuré, ce qui reste à prouver

Mesures du 7 octobre 2026 sur processeur, sans GPU, données synthétiques aléatoires, modules non entraînés,
3 exécutions chacune.

**Benchmark matériel** (`benchmarks/run_hardware_benchmark.py`, lot de 4 requêtes, 50 nœuds, 20 documents,
dimension 256)

| Mesure | Référence dense | AIOTECH44 |
|---|---|---|
| Documents de contexte retenus | 20 / 20 | **11 / 20** |
| Économie de contexte | 0 % | **45 %** |
| Latence par lot (CPU) | 0,02 ms | 0,4 à 0,6 ms |

La référence est une simple couche linéaire : sa latence n'est pas celle d'un LLM, la comparaison de temps n'a
donc pas de sens ici. Ce qui compte est le contexte que le LLM n'aura pas à relire.

**Garde-fou SCG** (`benchmarks/adversarial_scg_benchmark.py`)

| Trajectoire | Alignement avec les contraintes | Survie |
|---|---|---|
| Alignée | +1,0 | 100 %, conservée |
| Orthogonale | 0,0 | 0 %, élaguée |
| Déviante | −0,3 | 0 %, élaguée |
| Opposée | −1,0 | 0 %, élaguée |

Élagage global mesuré : 75 %, contre 50 % attendus par le script. Avec ces réglages (λ = 1, seuil 0,1), le SCG
écarte aussi les trajectoires orthogonales : le garde-fou est plus strict que prévu et son seuil reste à calibrer.

**Objectifs, présentés comme tels.** L'objectif du projet est de 25 à 40 % d'énergie économisée sans perte de
qualité perceptible. Il ne pourra être confirmé qu'en branchant Aiogps sur un vrai modèle de langage, avec un
trafic réel et une mesure de qualité des réponses. Les chiffres ci-dessus valident la mécanique, pas encore le
gain final.

## Pourquoi Aiogps

- **Sobriété** : moins de nœuds activés, moins de trajectoires explorées, moins de contexte relu ; chaque requête
  consomme selon sa difficulté.
- **Conformité** : les contraintes sont une géométrie, pas une consigne dans le prompt ; une trajectoire hors
  règles est élaguée avant la réponse.
- **Traçabilité** : complexité, budget de contexte, agents sollicités et perte géométrique sont renvoyés à chaque
  inférence.
- **Ouverture** : code Python et PyTorch, branché autour de votre modèle, sur votre infrastructure.

## Cas d'usage

- **Assistants et support client** : la majorité des questions sont simples ; la porte adaptative les traite à
  effort réduit et garde la pleine puissance pour les cas difficiles.
- **RAG documentaire d'entreprise** : la mémoire différentielle n'injecte que les documents utiles, donc moins de
  tokens facturés et des réponses moins diluées.
- **Agents métiers encadrés** (juridique, finance, santé) : le SCG écarte les raisonnements qui violent les règles
  du domaine, et la trace documente chaque réponse.

## Installation et exécution

```bash
git clone https://github.com/albanfredon23/Aiogps.git
cd Aiogps
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt

python main.py                                   # smoke test du pipeline
pytest tests/ -v                                 # tests
python benchmarks/run_hardware_benchmark.py      # benchmark matériel
python benchmarks/adversarial_scg_benchmark.py   # garde-fou SCG
python benchmarks/ablation_study.py              # étude d'ablation
streamlit run app.py                             # démonstrateur interactif
```

## Intégration

```python
from core.aiotech44_core import AIOTECH44_EnergyCore
import torch

core = AIOTECH44_EnergyCore(emb_dim=256, num_nodes=50, num_agents=4)
core.eval()

query = torch.randn(1, 256)        # requête
docs = torch.randn(1, 20, 256)     # documents récupérés
graph = torch.randn(1, 50, 256)    # nœuds du graphe de connaissances
constraints = torch.randn(1, 256)  # contraintes latentes

out = core(query, docs, graph, constraints)

out["complexity_score"]  # complexité estimée, 0 à 1
out["budget_k"]          # documents de contexte retenus
out["active_agents"]     # poids des 4 agents
out["scg_loss"]          # violation géométrique
```

## Structure du dépôt

```
Aiogps/
├── core/
│   ├── adaptive_gate.py       # porte de calcul adaptative
│   ├── beam_planner.py        # faisceau TAP différentiable
│   ├── memory_allocator.py    # mémoire différentielle (k de 2 à 15)
│   ├── dynamic_allocator.py   # 4 agents spécialisés
│   ├── symbolic_engine.py     # logique continue de Gödel
│   └── aiotech44_core.py      # orchestration du pipeline
├── scg/pruner.py              # élagage géométrique SCG
├── benchmarks/                # benchmark matériel, SCG adversarial, ablation
├── tests/                     # tests du pipeline
├── app.py                     # démonstrateur Streamlit
├── main.py                    # smoke test
└── site/                      # site de présentation 3D (Vite + Three.js)
```



## Auteur

Architecture et conception : Alban.
