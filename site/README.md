# Site vitrine Aiogps

Site statique (Vite + Three.js + GSAP) qui présente le pipeline AIOTECH44 en scène 3D pilotée par le défilement.
Publié sur GitHub Pages par `.github/workflows/pages.yml` à chaque push sur `main` touchant `site/`.

```bash
cd site
npm ci
npm run dev        # http://localhost:5173
npx vite build --base=/Aiogps/ && npx vite preview --base=/Aiogps/
```

- **3D** : `src/graph/scene.js`, chargée à la demande après le premier rendu ; repli SVG si WebGL est absent,
  l'appareil modeste ou le contexte perdu ; pixel ratio adaptatif (cible 60 FPS) ; `dispose()` à la sortie de page.
- **Consentement** : `src/consent.js` (« Tout accepter » et « Tout refuser » au même niveau, aucun script optionnel
  avant le choix, choix gardé 6 mois). Mesure d'audience : renseigner `aiogps:analytics-src` dans `index.html` et
  autoriser son domaine dans la CSP (`vite.config.js`).
- **CSP** : posée en balise meta au build (GitHub Pages ne permet pas d'en-têtes HTTP) ; aucun script, style ou
  connexion hors de l'origine.
- **Contact** : renseigner `aiogps:contact-email` (ouverture de la messagerie) ou `aiogps:contact-endpoint`
  (API de même origine) dans `index.html`.
- **Avant mise en ligne commerciale** : compléter les éléments surlignés des pages légales (éditeur, contact).
