import streamlit as st
import torch
import numpy as np

# Import du cœur AIOTECH44
from core.aiotech44_core import AIOTECH44_EnergyCore

st.set_page_config(
    page_title="AIOTECH44 — Démonstrateur Interactif",
    page_icon="⚡",
    layout="wide"
)

# ---------------------------------------------------------
# Initialisation en cache du modèle (évite de réinstancier)
# ---------------------------------------------------------
@st.cache_resource
def load_core(emb_dim: int = 256, num_nodes: int = 50, num_agents: int = 4):
    model = AIOTECH44_EnergyCore(emb_dim=emb_dim, num_nodes=num_nodes, num_agents=num_agents)
    model.eval()
    return model

emb_dim = 256
num_nodes = 50
core = load_core(emb_dim=emb_dim, num_nodes=num_nodes)

# ---------------------------------------------------------
# Interface Utilisateur : Barre latérale (Paramètres)
# ---------------------------------------------------------
st.sidebar.header("⚙️ Configuration de la Requête")

prompt_input = st.sidebar.text_area(
    "Requête utilisateur :",
    value="Résous ce problème de logique formelle en appliquant les contraintes strictes."
)

num_docs = st.sidebar.slider(
    "Documents contextuels initiaux (Fenêtre RAG) :",
    min_value=5,
    max_value=30,
    value=20,
    step=1
)

constraint_severity = st.sidebar.slider(
    "Sévérité des contraintes géométriques (SCG) :",
    min_value=0.0,
    max_value=2.0,
    value=0.8,
    step=0.1
)

run_button = st.sidebar.button("🚀 Exécuter l'inférence adaptative", type="primary")

# ---------------------------------------------------------
# En-tête principal
# ---------------------------------------------------------
st.title("⚡ AIOTECH44 : Middleware de Raisonnement Adaptatif (Green AI)")
st.markdown(
    "Démonstration en temps réel du couplage **Porte de complexité**, "
    "**Garde-fou géométrique SCG** et **Allocation différentielle de mémoire**."
)

if run_button or "first_run" not in st.session_state:
    st.session_state["first_run"] = True

    # 1. Génération synthétique contrôlée selon la saisie
    torch.manual_seed(len(prompt_input))
    query_emb = torch.randn(1, emb_dim)
    docs_emb = torch.randn(1, num_docs, emb_dim)
    graph_nodes = torch.randn(1, num_nodes, emb_dim)
    
    # Injection du vecteur de contraintes modulé par la sévérité
    constraints = torch.randn(1, emb_dim) * constraint_severity

    # 2. Inférence dans le pipeline AIOTECH44
    with torch.no_grad():
        out = core(query_emb, docs_emb, graph_nodes, constraints)

    complexity = float(out["complexity_score"].mean().item())
    budget_k = int(out["budget_k"].mean().item())
    memory_ratio = float(out["allocated_memory_ratio"])
    scg_loss = float(out["scg_loss"].item())
    agent_weights = out["active_agents"].squeeze(0).numpy()

    # Estimation des économies de calcul (FLOPs simulés)
    flops_baseline = (num_nodes ** 2) * emb_dim + (num_docs ** 2) * emb_dim
    flops_aiotech = (int(num_nodes * complexity) ** 2) * emb_dim + (budget_k ** 2) * emb_dim
    energy_saved = max(0.0, (1.0 - (flops_aiotech / flops_baseline)) * 100.0)

    # ---------------------------------------------------------
    # Affichage des métriques clés
    # ---------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Complexité Évaluée",
            value=f"{complexity:.2f}",
            delta="Frugal" if complexity < 0.5 else "Complexe",
            delta_color="inverse"
        )
    with col2:
        st.metric(
            label="Contexte Retenu (k)",
            value=f"{budget_k} / {num_docs} docs",
            delta=f"-{100.0 - memory_ratio:.1f}% volume"
        )
    with col3:
        st.metric(
            label="Violation SCG résiduelle",
            value=f"{scg_loss:.4f}",
            delta="Conforme" if scg_loss < 0.1 else "Élagué"
        )
    with col4:
        st.metric(
            label="Charge Calcul Économisée",
            value=f"-{energy_saved:.1f} %",
            delta="Gain Green AI"
        )

    st.divider()

    # ---------------------------------------------------------
    # Détail des étapes du pipeline
    # ---------------------------------------------------------
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("1. Répartition des Rôles d'Agents")
        roles = ["Raisonnement (Reasoning)", "Mathématiques (Math)", "Logique Formelle (Logic)", "Critique (Critic)"]
        chart_data = {roles[i]: float(agent_weights[i]) for i in range(len(roles))}
        st.bar_chart(chart_data)
        st.caption("Pondération des modules experts régulée par la boucle d'adaptation continue.")

    with col_right:
        st.subheader("2. Audit de l'Élagage & Trajectoires")
        active_trajs = out["trajectories"].squeeze(0)
        norm_scores = torch.norm(active_trajs, dim=-1).numpy()
        
        st.write(f"• **Forme des trajectoires survivantes :** `{list(active_trajs.shape)}`")
        st.write(f"• **Énergie résiduelle par trajectoire :**")
        for idx, score in enumerate(norm_scores):
            status = "Admissible" if score > 1e-4 else "Élaguée par SCG"
            st.progress(min(1.0, float(score) / 5.0), text=f"Trajectoire {idx+1} : {status}")

    # ---------------------------------------------------------
    # Sortie décisionnelle finale
    # ---------------------------------------------------------
    st.subheader("3. Décision du Modèle (Policy Head)")
    st.info(
        f"Politique calculée sur un tenseur de dimension **{list(out['policy'].shape)}**. "
        f"Le contexte documentaire a été physiquement tronqué à {budget_k} documents, "
        f"limitant les calculs matriciels en aval."
    )
