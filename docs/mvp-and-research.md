# Knotty MVP and Research Notes

## Product Direction

Core gameplay loop for V1:

1. Player receives a closed knot embedding with extra geometric complexity.
2. Player drags local segments through smooth, non-cutting deformations.
3. Goal is to reduce to a low-crossing, low-energy embedding.
4. Optional "show solution" performs aggressive relaxation as a hint.

## Why This Engine Shape

### Representation

- Knot is a closed polygonal chain (`N` nodes, periodic edges).
- A rendered tube is generated from a smooth interpolation of nodes.
- Physics keeps edge lengths near a rest length and discourages self-intersection.

This gives:

- Fast interactive updates.
- Stable dragging behavior.
- A clear path to future GPU compute kernels.

### Solver (Current MVP)

- Stretch term: preserves rope length.
- Bend term: penalizes high curvature.
- Vertex repulsion term: discourages near-contact.
- Edge-edge collision correction: avoids segment penetration.

The current solver is CPU-side for faster iteration. WebGPU currently handles rendering.

## MVP Architecture

- UI and state: `src/routes/+page.svelte`
- Scene and interaction: `src/lib/knot/engine.ts`
- Knot generation: `src/lib/knot/presets.ts`
- Dynamics and constraints: `src/lib/knot/solver.ts`
- Diagram metric approximation: `src/lib/knot/metrics.ts`

## Reuse vs Reinvent

### What We Can Reuse

- Web stack: SvelteKit + Vite + Three WebGPU renderer.
- Knot tables and metadata: KnotInfo / Knot Atlas style datasets.
- External invariant tooling (future): SnapPy, Regina, pyknotid (mostly Python-side).

### What We Should Not Copy Directly

- KnotPlot is not open source and has restrictive redistribution terms.
- Ridgerunner and several tightening codes are GPL-licensed; direct embedding in proprietary/commercial code would require GPL compliance.

For now, this project uses an original TypeScript solver implementation guided by published methods.

## Next Milestones

1. Deterministic challenge generator: fixed seed + known target crossings.
2. Accurate topological checks via worker/WASM bridge to invariant tooling.
3. Compute-shader solver pass for larger node counts.
4. Puzzle modes:
   - "Minimize this knot"
   - "Knotty or unknotty?"
   - Time/score ladder
5. Content layer: curated knot/link table with unlock progression.

## Primary Sources Consulted

- SvelteKit introduction and tooling:
  - https://svelte.dev/docs/kit/introduction
  - https://svelte.dev/docs/cli/sv-create
- Three.js WebGPU usage notes:
  - https://threejs.org/manual/en/webgpu.html
- WebGPU platform/API reference:
  - https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API
- KnotPlot terms/site:
  - https://knotplot.com/
- Ridgerunner and license:
  - https://github.com/designbynumbers/ridgerunner
- Minimum ropelength optimization paper:
  - https://arxiv.org/abs/math/0202284
- Discrete elastic rods:
  - https://www.cs.columbia.edu/cg/rods/
- DisMech simulator (open-source implementation notes):
  - https://github.com/StructuresComp/dismech-rods
- Knot table/invariant references:
  - https://knotinfo.math.indiana.edu/
  - https://katlas.org/wiki/Main_Page
  - https://www.math.uic.edu/t3m/SnapPy/
  - https://regina-normal.github.io/
  - https://github.com/SPOCKnots/pyknotid
