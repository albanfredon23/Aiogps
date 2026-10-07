/**
 * Démo interactive : simulation pédagogique du pipeline AIOTECH44, exécutée
 * entièrement dans le navigateur (aucune donnée envoyée).
 *
 * Elle reprend les règles du dépôt sans charger le modèle PyTorch :
 * - porte adaptative : effort = 30 % + 70 % × complexité (AdaptiveComputeGate) ;
 * - faisceau TAP : 4 trajectoires (beam_width = 4) ;
 * - SCG : violation angulaire relu(−t·c) pondérée par la sévérité, élagage franc ;
 * - mémoire différentielle : k entre 2 et 15, borné par le nombre de documents ;
 * - calcul économisé : même formule de FLOPs que le démonstrateur Streamlit (app.py).
 * La complexité est ici estimée par une heuristique lisible, pas par le réseau.
 */
import { track } from './analytics.js';

const EMB = 256;
const NODES = 50;
const BEAM = 4;
const K_MIN = 2;
const K_MAX = 15;

const PRESETS = {
  faq: "Quels sont vos horaires d'ouverture ?",
  raisonnement:
    "Explique pourquoi cette clause du contrat s'applique, puis déduis les obligations de chaque partie si le délai n'est pas respecté.",
  calcul: 'Calcule le coût total si 3 serveurs consomment 450 W pendant 18 heures à 0,21 € le kWh, puis vérifie le résultat.',
};

const AGENTS = [
  { id: 'reasoning', label: 'Raisonnement', words: ['pourquoi', 'explique', 'déduis', 'analyse', 'compare', 'cause', 'raison'] },
  { id: 'math', label: 'Math', words: ['calcule', 'combien', 'somme', 'total', 'moyenne', '%', 'kwh', 'coût'] },
  { id: 'logic', label: 'Logique', words: ['si', 'alors', 'prouve', 'démontre', 'règle', 'clause', 'contrainte', 'logique'] },
  { id: 'critic', label: 'Critique', words: ['vérifie', 'valide', 'critique', 'relis', 'contrôle', 'erreur'] },
];

const fmt = (v, digits = 0) => v.toLocaleString('fr-FR', { maximumFractionDigits: digits, minimumFractionDigits: digits });
const sigmoid = (x) => 1 / (1 + Math.exp(-x));
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

/** Empreinte stable du texte (FNV-1a), pour un résultat reproductible. */
function hash(text) {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i += 1) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h;
}

function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function simulate({ query, docs, severity }) {
  const text = query.trim().toLowerCase();
  const tokens = text.split(/[^\p{L}\p{N}%]+/u).filter(Boolean);
  const hits = AGENTS.map((a) => tokens.filter((t) => a.words.includes(t)).length + (a.id === 'math' ? (text.match(/\d+/g) || []).length * 0.5 : 0));
  const signal = hits.reduce((s, v) => s + v, 0);

  // 1. Porte adaptative
  const complexity = clamp(sigmoid(-2.2 + tokens.length * 0.06 + signal * 0.45), 0.02, 0.98);
  const effort = 0.3 + 0.7 * complexity;
  const activeNodes = Math.round(NODES * effort);

  // 2. Faisceau TAP : 4 trajectoires, alignement t·c sur la sphère
  const rand = rng(hash(text));
  const trajectories = Array.from({ length: BEAM }, (_, i) => {
    const alignment = clamp([0.85, 0.25, -0.35, -0.9][i] + (rand() - 0.5) * 0.3, -1, 1);
    // 3. SCG : énergie de violation, élagage franc au-delà du seuil 0,5
    const energy = severity * (1 - alignment) * 0.5;
    return { id: i + 1, alignment, energy, kept: energy < 0.5 };
  });
  const kept = trajectories.filter((t) => t.kept);

  // 4. Mémoire différentielle : k selon la confiance du faisceau survivant
  const confidence = kept.length ? kept.reduce((s, t) => s + (t.alignment + 1) / 2, 0) / kept.length : 0;
  const norm = (kept.length / BEAM) * (0.35 + 0.65 * complexity) * (0.6 + 0.4 * confidence);
  const budgetK = clamp(Math.floor(K_MIN + norm * (K_MAX - K_MIN)), K_MIN, Math.min(K_MAX, docs));

  // 5. Agents : softmax des affinités
  const logits = hits.map((h, i) => h * 0.9 + (i === 3 ? 0.4 : 0.6));
  const maxL = Math.max(...logits);
  const exps = logits.map((l) => Math.exp(l - maxL));
  const sum = exps.reduce((s, v) => s + v, 0);
  const agents = AGENTS.map((a, i) => ({ id: a.id, label: a.label, weight: exps[i] / sum }));

  // Calcul économisé : formule de FLOPs du démonstrateur (app.py)
  const flopsBaseline = NODES ** 2 * EMB + docs ** 2 * EMB;
  const flopsAio = Math.floor(NODES * complexity) ** 2 * EMB + budgetK ** 2 * EMB;
  const saved = Math.max(0, 1 - flopsAio / flopsBaseline);

  return {
    complexity,
    effort,
    activeNodes,
    trajectories,
    kept: kept.length,
    budgetK,
    docs,
    agents,
    saved,
    decision: kept.length ? 'REPONSE' : 'REFUS_CONTRAINTES',
  };
}

export function initDemo() {
  const form = document.getElementById('demo-form');
  if (!form) return;
  const query = form.elements.query;
  const docs = form.elements.docs;
  const severity = form.elements.severity;
  const docsOut = document.getElementById('demo-docs-out');
  const sevOut = document.getElementById('demo-sev-out');
  const status = document.getElementById('demo-status');
  const kpis = document.getElementById('demo-kpis');
  const bars = document.getElementById('demo-agents');
  const beams = document.getElementById('demo-beams');
  const memory = document.getElementById('demo-memory');
  const trace = document.getElementById('demo-trace');
  let tracked = false;

  const kpi = (key, value) => {
    const el = kpis.querySelector(`[data-kpi="${key}"]`);
    if (el) el.textContent = value;
  };

  function render() {
    docsOut.value = docs.value;
    sevOut.value = fmt(Number(severity.value), 1);
    if (!query.value.trim()) {
      status.textContent = 'Saisissez une requête pour lancer la simulation.';
      return;
    }
    const r = simulate({ query: query.value, docs: Number(docs.value), severity: Number(severity.value) });

    kpi('complexity', fmt(r.complexity, 2));
    kpi('effort', `${fmt(r.effort * 100)} %`);
    kpi('nodes', `${r.activeNodes} / ${NODES}`);
    kpi('pruned', `${BEAM - r.kept} / ${BEAM}`);
    kpi('k', `${r.budgetK} / ${r.docs}`);
    kpi('saved', `${fmt(r.saved * 100, 1)} %`);
    kpis.querySelector('[data-kpi="saved"]')?.classList.toggle('kpi-ok', r.saved > 0.3);

    bars.replaceChildren(
      ...r.agents.map((a) => {
        const li = document.createElement('li');
        const name = document.createElement('span');
        name.textContent = a.label;
        const track = document.createElement('span');
        track.className = 'bar-track';
        const fill = document.createElement('span');
        fill.className = 'bar-fill';
        fill.style.width = `${(a.weight * 100).toFixed(1)}%`;
        track.append(fill);
        const val = document.createElement('span');
        val.className = 'bar-value';
        val.textContent = `${fmt(a.weight * 100)} %`;
        li.append(name, track, val);
        return li;
      }),
    );

    beams.replaceChildren(
      ...r.trajectories.map((t) => {
        const li = document.createElement('li');
        li.className = t.kept ? 'beam beam-kept' : 'beam beam-pruned';
        li.textContent = `Trajectoire ${t.id} : alignement ${fmt(t.alignment, 2)}, énergie ${fmt(t.energy, 2)}, ${t.kept ? 'conservée' : 'élaguée'}`;
        return li;
      }),
    );

    memory.replaceChildren(
      ...Array.from({ length: r.docs }, (_, i) => {
        const cell = document.createElement('span');
        cell.className = i < r.budgetK ? 'doc doc-kept' : 'doc';
        return cell;
      }),
    );
    memory.setAttribute('aria-label', `${r.budgetK} documents retenus sur ${r.docs}`);

    trace.textContent = JSON.stringify(
      {
        complexity_score: Number(r.complexity.toFixed(3)),
        compute_effort: Number(r.effort.toFixed(3)),
        active_graph_nodes: r.activeNodes,
        trajectories: r.trajectories.map((t) => ({ id: t.id, alignment: Number(t.alignment.toFixed(3)), scg_energy: Number(t.energy.toFixed(3)), kept: t.kept })),
        budget_k: r.budgetK,
        retrieved_docs: r.docs,
        active_agents: Object.fromEntries(r.agents.map((a) => [a.id, Number(a.weight.toFixed(3))])),
        estimated_compute_saved: Number(r.saved.toFixed(3)),
        decision: r.decision,
      },
      null,
      2,
    );

    status.textContent =
      r.kept === 0
        ? 'Toutes les trajectoires violent les contraintes : le pipeline refuse de répondre plutôt que de produire une réponse non conforme.'
        : `Complexité ${fmt(r.complexity, 2)} : ${fmt(r.saved * 100, 1)} % de calcul économisé, ${BEAM - r.kept} trajectoire(s) élaguée(s), ${r.budgetK} documents retenus.`;
  }

  form.addEventListener('click', (event) => {
    const preset = event.target.closest('[data-preset]')?.dataset.preset;
    if (!preset) return;
    query.value = PRESETS[preset];
    render();
  });
  form.addEventListener('input', () => {
    render();
    if (!tracked) {
      tracked = true;
      track('demo_utilisee');
    }
  });
  form.addEventListener('submit', (event) => {
    event.preventDefault();
    render();
  });
  render();
}
