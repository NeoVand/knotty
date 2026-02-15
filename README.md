# Knotty

Knotty is a SvelteKit + TypeScript + WebGPU prototype for knot-theory gameplay:

- Closed curves (mathematical knots) represented as inextensible, self-avoiding loops.
- Player interactions via direct drag moves on knot segments.
- Continuous relaxation toward low-energy embeddings.
- Crossing-count objective as the first puzzle metric.

## Run

```sh
npm install
npm run dev
```

## MVP Status

Implemented:

- SvelteKit app scaffolded with TypeScript and Vite.
- WebGPU 3D viewport using `three` WebGPU renderer, with WebGL fallback.
- Knot preset generator (`T(p,q)` torus knots + random torus challenge).
- Simple rod-like solver (stretch + bend + self-avoidance repulsion + edge collision correction).
- Interactive drag controls on knot segments.
- Approximate crossing-count estimator and live energy score.
- "Show solution" fast relaxation action.

Not yet implemented:

- Certified topological invariants (Alexander/Jones, exact unknot tests).
- Puzzle progression, scoring system, replay system, multiplayer.
- GPU compute solver kernels (current solver is CPU-based).

## Key Files

- `src/routes/+page.svelte`: game UI and control panel.
- `src/lib/knot/engine.ts`: WebGPU scene, interaction, render/update loop.
- `src/lib/knot/solver.ts`: knot dynamics and self-collision handling.
- `src/lib/knot/presets.ts`: knot generators and challenge state creation.
- `src/lib/knot/metrics.ts`: crossing-count estimator.
- `docs/mvp-and-research.md`: architecture and source-backed research notes.

## Scripts

- `npm run dev`: start dev server.
- `npm run check`: type and Svelte checks.
- `npm run build`: production build.
