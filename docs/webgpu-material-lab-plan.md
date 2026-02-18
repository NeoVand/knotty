# WebGPU Material Lab Plan

## Goals

1. Upgrade knot rendering fidelity (materials, lighting, reflections, animated effects).
2. Keep a WebGPU-first renderer path with graceful WebGL fallback.
3. Add a user-facing Material Lab for rapid visual iteration.
4. Prepare for future GPU/compute-style field simulation.

## Primary References

- Three.js WebGPU manual: https://threejs.org/manual/en/webgpu.html
- Three.js MeshPhysicalMaterial docs: https://threejs.org/docs/#api/en/materials/MeshPhysicalMaterial
- Three.js PMREMGenerator docs: https://threejs.org/docs/#api/en/extras/PMREMGenerator
- WebGPU API reference (MDN): https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API

## Architecture (Parallel Workstreams)

### Workstream A: Material Pipeline

- Unified `KnotMaterialSettings` API in engine.
- Presets: `rope`, `glass`, `liquid_metal`, `energy_field`, `vector_field`.
- Procedural texture library (brushed metal, energy streaks, vector arrows, frosted normal).
- Animated per-material texture offsets for "flow on each ring."

### Workstream B: Lighting + Environment

- Unified `KnotLightingSettings` API in engine.
- Key + fill + rim + hemisphere light rig with runtime controls.
- PMREM environment map generation using `RoomEnvironment`.
- Tone mapping exposure exposed in UI.

### Workstream C: UX / Lab Controls

- New Material Lab control section in `+page.svelte`.
- Preset switching + physically based sliders (roughness/metalness/transmission/clearcoat).
- Lighting sliders (exposure, ambient, key, fill, rim).
- Real-time updates through Svelte effects.

### Workstream D: Future WebGPU Shader/Compute Expansion

- Move procedural field generation from CPU canvas textures to GPU-driven buffers/textures.
- Add vector-field modes from actual solver dynamics (velocity, curvature, torsion, repulsion force).
- Optionally add WebGPU compute pass for field advection and temporal smoothing.

## Phase Plan

1. Phase 1 (done): Material Lab controls + physically based material presets + animated texture flow.
2. Phase 2: TSL/NodeMaterial upgrade for richer GPU-native shading and fresnel/iridescence controls.
3. Phase 3: Derived-field overlays from solver metrics (per-vertex curvature/strain/repulsion).
4. Phase 4: WebGPU compute prototypes for dynamic field simulation and post effects.

## Acceptance Criteria

1. Visual quality clearly improves in all presets at interactive frame rates.
2. User can tune materials/lights without restarting or regenerating.
3. Energy/vector presets visibly animate along each link component.
4. System remains stable under repeated preset/slider changes.
