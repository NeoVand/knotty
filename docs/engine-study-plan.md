# Knot Engine Study Plan

## Scope

Build a reliable knot creation + manipulation engine for gameplay:

- Generate arbitrary-complexity closed knots/links from planar shadows.
- Preserve loop length and self-avoidance during user manipulation.
- Relax toward lower-energy embeddings without topology changes.
- Keep UI simple enough to validate generation/physics quickly.

## Current Gaps (from code audit)

1. Generator quality is better than before, but not yet validated by hard invariants per seed.
2. Solver is a practical hybrid (Sobolev-like descent + projections), not a faithful Repulsive Curves pipeline.
3. Drag interaction still injects displacement directly; this can look stretchy under heavy pulls.
4. Convergence behavior is opaque (no explicit residual/step-size diagnostics in UI).
5. Naming/docs drift created confusion (fixed: `random_shadow` now replaces `random_torus`).

## Guiding Constraints

1. Topology must never change from accidental intersections.
2. Arc length should be near-inextensible (small elastic tolerance only).
3. Crossing neighborhoods must spawn with clearance >= `2 * tubeRadius`.
4. Every algorithmic claim must be testable with deterministic seeds.

## Phase 1: Deterministic Diagnostics First

Deliverables:

- Add a `KnotDiagnostics` stream (per frame or sampled):
  - length drift (% from rest)
  - minimum non-adjacent edge distance
  - collision pair count
  - gradient norm
  - accepted/rejected line-search steps
  - crossing estimate
- Add a developer overlay toggle for these metrics.
- Add deterministic replay harness:
  - fixed seed
  - fixed parameter pack
  - optional scripted drag sequence

Acceptance:

- Same seed + params reproduces geometry within tolerance.
- We can tell exactly why auto-relax “stops” (converged vs stalled).

## Phase 2: Generator Hardening (Shadow -> Embedding)

Deliverables:

- Keep current primal/medial construction as base.
- Tighten candidate rejection with hard constraints:
  - crossing angle lower bound
  - local trim radius lower bound
  - initial 3D edge-edge clearance >= `2.1 * tubeRadius`
- Add post-sample arc-length projection and curvature limiter pass.
- Add seed corpus (`3..128` crossings) and save quality stats.

Acceptance:

- For corpus seeds, zero initial self-intersections.
- No visible “wiggle” artifacts from initial spline conversion.

## Phase 3: Solver Refactor Toward Repulsive Curves Style

Deliverables:

- Separate energy terms cleanly:
  - repulsive energy (tangent-point style)
  - stretch penalty
  - bending regularization
  - collision barrier
- Replace fallback unconditional move with guarded update:
  - if line search fails repeatedly, switch to tiny projected step only when energy decreases.
- Add explicit stopping criteria:
  - gradient norm threshold
  - minimum relative energy decrease
  - max stall iterations
- Expose controls that map to physical meaning:
  - repulsion
  - stiffness/stretch
  - damping/smoothing
  - iterations per frame

Acceptance:

- Auto-relax converges smoothly without runaway stretching.
- Length drift remains bounded (target < 2% during normal play).

## Phase 4: Interaction Model (Physically Plausible Pulling)

Deliverables:

- Replace direct point move with soft constraint:
  - dragged handle target
  - local influence kernel
  - internal resistance via stretch constraints each substep
- Add optional interaction modes:
  - `Grab` (local shape edit)
  - `Move Arc` (broader influence, stronger length preservation)
- Keep manipulations reversible and stable under fast mouse motion.

Acceptance:

- User pulls feel transmitted through the loop, not like rubber-band elongation.
- No sudden jitter spikes when starting drag.

## Phase 5: UI Simplification for Testing

Deliverables:

- Keep only essential controls for this stage:
  - crossings
  - seed
  - generate
  - auto-relax on/off
  - solid/line view
  - repulsion, stretch stiffness, iterations/frame
- Add one-click presets:
  - `Stable`
  - `Fast simplify`
  - `Diagnostic`

Acceptance:

- Regenerating a challenge always updates immediately.
- Parameter changes show immediate, understandable effect.

## Phase 6: Regression Suite + Content Pipeline

Deliverables:

- Seed regression pack with screenshots + metric snapshots.
- Fail build if:
  - initial collision count > 0
  - length drift exceeds threshold in benchmark run
  - generation failure rate exceeds threshold
- Prepare knot/link table ingestion format for future puzzle modes.

Acceptance:

- Engine changes stop regressing quality between iterations.

## Recommended Execution Order (next 3 work blocks)

1. Phase 1 diagnostics + replay harness.
2. Phase 2 generator hardening with seed corpus.
3. Phase 3 solver refactor (then revisit drag model in Phase 4).
