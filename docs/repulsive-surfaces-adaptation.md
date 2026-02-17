# Repulsive Surfaces Adaptation Notes (for Knotty)

## Sources reviewed

- CMU project page: https://www.cs.cmu.edu/~kmcrane/Projects/RepulsiveSurfaces/index.html
- Paper abstract page: https://arxiv.org/abs/2107.01664
- Codebase: https://github.com/ythea/repulsive-surfaces

Local code inspected from cloned repository:

- `/tmp/knot-research/repulsive-surfaces/src/surface_flow.cpp`
- `/tmp/knot-research/repulsive-surfaces/src/line_search.cpp`
- `/tmp/knot-research/repulsive-surfaces/src/sobolev/hs.cpp`
- `/tmp/knot-research/repulsive-surfaces/src/sobolev/hs_iterative.cpp`
- `/tmp/knot-research/repulsive-surfaces/src/energy/tpe_barnes_hut_0.cpp`
- `/tmp/knot-research/repulsive-surfaces/src/energy/tp_obstacle_barnes_hut_0.cpp`
- `/tmp/knot-research/repulsive-surfaces/src/remeshing/dynamic_remesher.cpp`
- `/tmp/knot-research/repulsive-surfaces/scenes/FORMAT.txt`

## What matters most for Knotty

The key reusable pattern is not "mesh code", but the optimization architecture:

1. Assemble total L2 gradient from all energy terms.
2. Project/precondition into Sobolev metric (Hs/H1 style).
3. Take Armijo backtracking step.
4. Project constraints (pins, barycenter, area/volume equivalents).
5. Repeat with consistent near-field interactions enabled.

This architecture appears in:

- `SurfaceFlow::StepProjectedGradient*` in `src/surface_flow.cpp`
- `LineSearch::BacktrackingLineSearch` in `src/line_search.cpp`
- `HsMetric` and Schur projection in `src/sobolev/hs.cpp` and `include/sobolev/hs_schur.h`

## Immediate applicability to a knot centerline engine

### Keep and adapt directly

1. `SurfaceEnergy`-style modular energy interface.
2. Armijo backtracking with state restore-on-failure (do not force fallback moves).
3. Constraint separation:
   - hard/simple projectors (drag pin, recenter)
   - saddle/Schur constraints (global rope length, optional fixed-length segments)
4. Optional hierarchical acceleration path (Barnes-Hut/BVH) once node count grows.

### Reinterpret for a 1D closed curve + tube radius

1. Replace triangle-face primitives with centerline segments.
2. Use segment-segment minimum distance and tube radius `r` to enforce clearance.
3. Near-field threshold should be `>= 2r` (plus safety margin), not `~r`.
4. Add continuous constraints so drag cannot elongate rope beyond elastic tolerance.

## Critical mismatch in current Knotty code

Current solver thresholds are below tube diameter:

- Edge collision projection currently uses approximately `1.04r..1.08r`.
- Energy collision penalty currently starts around `1.22r`.

For a rendered tube of radius `r`, self-intersection avoidance requires centerline spacing near `2r` (or larger safety margin). Anything much below `2r` allows surface interpenetration.

## Proposed implementation sequence for Knotty

## Phase A: Safety floor (short-term, highest impact)

1. Raise dynamic collision/barrier distances to `>= 2r`.
2. Remove unconditional fallback step that can accept non-improving moves.
3. Add one diagnostic panel:
   - min non-adjacent segment distance
   - ratio `(min distance) / (2r)`
   - accepted/rejected line-search steps

Acceptance:

- No visible tube self-intersection on generation + auto-relax + drag stress tests.

## Phase B: Repulsive-Surfaces-style solver loop

1. Refactor into explicit energy modules:
   - repulsion term
   - stretch term
   - bend/smoothing term
   - collision barrier term
2. Build projected descent loop:
   - L2 gradient assembly
   - Sobolev/H1 preconditioned direction solve
   - Armijo backtracking
   - constraint projection

Acceptance:

- Monotone (or near-monotone) energy decrease with clear convergence/stall reason.

## Phase C: Constraint quality and interaction realism

1. Add global length constraint via Schur/saddle-style correction (curve version).
2. Convert drag to target constraint (not direct point displacement).
3. Add edge-local inextensibility projection passes after drag updates.

Acceptance:

- Pulling a segment feels transmitted around the loop; no rubber-band stretching.

## Phase D: Performance scaling

1. Start with exact near-field + medium-range pair list.
2. Add Barnes-Hut/BVH approximation for far-field repulsion.
3. Consider WebGPU compute for pairwise kernels once CPU profile justifies it.

Acceptance:

- Stable 200-1000 centerline nodes at interactive rates.

## What not to port now

1. Full triangle remeshing machinery (your game state is curve-first).
2. Scene parser and desktop viewer integration.
3. Surface-specific potentials (Willmore, volume growth) unless needed for special modes.

## Licensing note

`ythea/repulsive-surfaces` is MIT licensed (`LICENSE`), so algorithmic and code adaptation into this project is allowed with attribution and license preservation where required.
