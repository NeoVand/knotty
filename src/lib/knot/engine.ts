import type { WebGLRenderer } from 'three';
import * as THREE from 'three/webgpu';
import { WebGLRenderer as CoreWebGLRenderer } from 'three';
import WebGPU from 'three/examples/jsm/capabilities/WebGPU.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { estimateCrossings } from './metrics';
import {
	copyPoints,
	createKnotState,
	PRESET_ORDER,
	type KnotPresetName,
	type KnotState
} from './presets';
import { KnotSolver, type SolverOptions } from './solver';

const TUBE_COLOR = '#ffcf73';
const LINE_COLOR = '#8cf7df';
const COMPONENT_TUBE_COLORS = ['#ffcf73', '#79dfc4', '#f3a66a', '#73a5ff', '#d6e178', '#f288b4', '#b39dff'];
const COMPONENT_LINE_COLORS = ['#8cf7df', '#61f2c6', '#ffc17d', '#8fc4ff', '#e2f28f', '#ff9ec8', '#c8b2ff'];

export interface KnotMetrics {
	preset: KnotPresetName;
	label: string;
	crossings: number;
	targetCrossings: number | null;
	energy: number;
	nodeCount: number;
	minClearance: number;
	clearanceRatio: number;
	maxLengthDrift: number;
	edgeUniformity: number;
	acceptedSteps: number;
	rejectedSteps: number;
}

export interface KnotEngineOptions {
	container: HTMLElement;
	onMetrics?: (metrics: KnotMetrics) => void;
	onStatus?: (message: string) => void;
}

interface DragSelection {
	componentIndex: number;
	pointIndex: number;
}

export class KnotEngine {
	private static readonly DEFAULT_WEBGPU_TEXTURE_LIMIT = 8192;
	private static readonly MAX_TARGET_PIXEL_RATIO = 2;

	private readonly container: HTMLElement;
	private readonly onMetrics?: (metrics: KnotMetrics) => void;
	private readonly onStatus?: (message: string) => void;
	private readonly raycaster = new THREE.Raycaster();
	private readonly pointer = new THREE.Vector2();
	private readonly dragPlane = new THREE.Plane();
	private readonly dragPoint = new THREE.Vector3();
	private readonly cameraDirection = new THREE.Vector3();
	private readonly tempObject = new THREE.Object3D();

	private renderer: THREE.WebGPURenderer | WebGLRenderer | null = null;
	private scene: THREE.Scene | null = null;
	private camera: THREE.PerspectiveCamera | null = null;
	private controls: OrbitControls | null = null;
	private resizeObserver: ResizeObserver | null = null;

	private state: KnotState | null = null;
	private points: Float32Array | null = null;
	private passiveComponents: Float32Array[] = [];
	private solver: KnotSolver | null = null;
	private passiveSolvers: KnotSolver[] = [];
	private tubeMesh: THREE.Mesh | null = null;
	private lineMesh: THREE.LineLoop | null = null;
	private passiveTubeMeshes: THREE.Mesh[] = [];
	private passiveLineMeshes: THREE.LineLoop[] = [];
	private controlMesh: THREE.InstancedMesh | null = null;
	private autoRelax = true;
	private showSolidLink = true;
	private showControlPoints = false;
	private colorizeLinks = false;
	private useArcGuideLayout = true;
	private repulsionStrength = 1;
	private relaxSmoothness = 0.62;
	private relaxIterations = 3;
	private dragIndex: number | null = null;
	private dragComponentIndex: number | null = null;
	private readonly dragTarget = new THREE.Vector3();
	private geometryDirty = true;
	private disposed = false;
	private running = false;
	private lastFrameAt = 0;
	private lastMetricAt = 0;
	private lastBoundsAt = 0;
	private relaxTick = 0;
	private maxRenderDimension2D = KnotEngine.DEFAULT_WEBGPU_TEXTURE_LIMIT;

	constructor(options: KnotEngineOptions) {
		this.container = options.container;
		this.onMetrics = options.onMetrics;
		this.onStatus = options.onStatus;
	}

	async init(): Promise<void> {
		if (this.disposed) return;

		this.scene = new THREE.Scene();
		this.scene.background = new THREE.Color('#051d1a');
		this.scene.fog = new THREE.Fog('#051d1a', 7, 18);

		this.camera = new THREE.PerspectiveCamera(43, 1, 0.1, 120);
		this.camera.position.set(0, 0, 7.8);

		this.renderer = await this.createRenderer();
		this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
		this.renderer.toneMappingExposure = 1.1;
		this.renderer.domElement.style.display = 'block';
		this.renderer.domElement.style.width = '100%';
		this.renderer.domElement.style.height = '100%';
		this.container.appendChild(this.renderer.domElement);

		this.controls = new OrbitControls(this.camera, this.renderer.domElement);
		this.controls.enableDamping = true;
		this.controls.dampingFactor = 0.09;
		this.controls.minDistance = 4.2;
		this.controls.maxDistance = 42;

		const hemi = new THREE.HemisphereLight('#7effdd', '#071511', 0.85);
		const key = new THREE.DirectionalLight('#fff6cf', 1.35);
		key.position.set(5, 7, 6);
		const rim = new THREE.PointLight('#ff9f4a', 15, 20, 2);
		rim.position.set(-3.6, 1.4, -4.2);
		this.scene.add(hemi, key, rim);

		this.attachInteractionHandlers();
		this.setShowControlPoints(false);
		this.running = true;
		this.lastFrameAt = performance.now();
		this.lastBoundsAt = this.lastFrameAt;
		this.frame(this.lastFrameAt);
		this.observeResize();
		this.notifyStatus(
			this.renderer instanceof CoreWebGLRenderer
				? 'WebGL fallback active. Drag points to apply smooth knot moves.'
				: 'WebGPU active. Drag points to apply smooth knot moves.'
		);
	}

	dispose(): void {
		if (this.disposed) return;
		this.disposed = true;
		this.running = false;
		this.detachInteractionHandlers();
		this.resizeObserver?.disconnect();
		this.controls?.dispose();
		this.controls = null;
		this.disposeGeometry();
		this.renderer?.dispose();
		if (this.renderer?.domElement.parentElement === this.container) {
			this.container.removeChild(this.renderer.domElement);
		}
		this.renderer = null;
		this.scene = null;
		this.camera = null;
	}

	setPreset(name: KnotPresetName): void {
		this.createState(name);
	}

	setCrossingChallenge(crossings: number, seed: number): void {
		const targetCrossings = clampInteger(crossings, 3, 128);
		const normalizedSeed = seed >>> 0;
		const state = createKnotState('random_shadow', {
			seed: normalizedSeed === 0 ? 1 : normalizedSeed,
			crossings: targetCrossings,
			useArcGuideLayout: this.useArcGuideLayout
		});
		this.applyState(state);
		const loopLabel = state.componentCount === 1 ? 'single loop' : `${state.componentCount} loops`;
		this.notifyStatus(
			`Generated ${targetCrossings}-crossing challenge (${loopLabel}, seed ${normalizedSeed || 1}).`
		);
	}

	setAutoRelax(value: boolean): void {
		this.autoRelax = value;
	}

	setShowControlPoints(value: boolean): void {
		this.showControlPoints = value;
		if (this.controlMesh) this.controlMesh.visible = value;
	}

	setSolidLink(value: boolean): void {
		this.showSolidLink = value;
		if (this.tubeMesh) this.tubeMesh.visible = value;
		if (this.lineMesh) this.lineMesh.visible = !value;
		for (const mesh of this.passiveTubeMeshes) mesh.visible = value;
		for (const mesh of this.passiveLineMeshes) mesh.visible = !value;
	}

	setColorizeLinks(value: boolean): void {
		this.colorizeLinks = value;
		this.applyComponentColors();
	}

	setArcGuideLayout(value: boolean): void {
		this.useArcGuideLayout = value;
	}

	setRepulsionStrength(value: number): void {
		this.repulsionStrength = clampNumber(value, 0.2, 3.5);
		this.applySolverTuning();
	}

	setRelaxSmoothness(value: number): void {
		this.relaxSmoothness = clampNumber(value, 0, 1);
		this.applySolverTuning();
	}

	setRelaxIterations(value: number): void {
		this.relaxIterations = clampInteger(value, 1, 8);
	}

	stepRelax(iterations = 12): void {
		if (!this.points || !this.solver || !this.state) return;
		this.stepAllComponents(iterations, 1 / 120, null);
		this.geometryDirty = true;
	}

	showSolution(iterations = 220): void {
		if (!this.points || !this.solver || !this.state) return;
		this.stepAllComponents(iterations, 1 / 100, null);
		this.geometryDirty = true;
		this.notifyStatus('Relaxed toward a low-energy embedding.');
	}

	getAvailablePresets(): KnotPresetName[] {
		return PRESET_ORDER;
	}

	private frame = (now: number): void => {
		if (!this.running || this.disposed) return;
		requestAnimationFrame(this.frame);

		const renderer = this.renderer;
		const scene = this.scene;
		const camera = this.camera;
		const controls = this.controls;
		const points = this.points;
		const solver = this.solver;
		const state = this.state;
		if (!renderer || !scene || !camera || !controls || !points || !solver || !state) return;

		const dt = Math.min(1 / 25, Math.max(1 / 200, (now - this.lastFrameAt) / 1000));
		this.lastFrameAt = now;

		if (this.autoRelax && this.dragIndex === null) {
			this.stepAllComponents(this.relaxIterations, dt, null);
			this.geometryDirty = true;
		}

		if (this.dragIndex !== null && this.dragComponentIndex !== null) {
			this.applyDragTarget();
			const dragIterations = Math.max(2, this.relaxIterations);
			this.stepAllComponents(dragIterations, dt, {
				componentIndex: this.dragComponentIndex,
				pointIndex: this.dragIndex
			});
			this.geometryDirty = true;
		}

		if (this.geometryDirty) this.rebuildGeometry();
		if (now - this.lastBoundsAt > 220) {
			this.lastBoundsAt = now;
			this.updateCameraForCurrentBounds();
		}
		controls.update();
		renderer.render(scene, camera);

		if (now - this.lastMetricAt > 220) {
			this.lastMetricAt = now;
			this.emitMetrics();
		}
	};

	private createState(name: KnotPresetName): void {
		const nextState = createKnotState(name, { useArcGuideLayout: this.useArcGuideLayout });
		this.applyState(nextState);
		this.notifyStatus(`Loaded ${nextState.label}.`);
	}

	private applyState(nextState: KnotState): void {
		this.state = nextState;
		this.points = copyPoints(this.state.points);
		this.passiveComponents = this.state.componentsPoints.slice(1).map((component) => copyPoints(component));
		this.passiveSolvers = [];
		this.relaxTick = 0;
		const multipleComponents = this.state.componentCount > 1;
		const activeRestLength = this.state.componentRestLengths[0] ?? this.state.restLength;
		this.solver = new KnotSolver(this.points.length / 3, activeRestLength, this.currentSolverOptions());
		this.solver.setKeepCentered(!multipleComponents);
		for (let componentIndex = 0; componentIndex < this.passiveComponents.length; componentIndex += 1) {
			const componentPoints = this.passiveComponents[componentIndex];
			const componentRestLength =
				this.state.componentRestLengths[componentIndex + 1] ?? averageEdgeLength(componentPoints);
			const componentSolver = new KnotSolver(
				componentPoints.length / 3,
				componentRestLength,
				this.currentSolverOptions()
			);
			componentSolver.setKeepCentered(false);
			this.passiveSolvers.push(componentSolver);
		}
		this.applySolverTuning();
		this.dragIndex = null;
		this.dragComponentIndex = null;
		this.geometryDirty = true;
		this.rebuildGeometry(true);
		this.fitViewToState();
		this.emitMetrics();
	}

	private currentSolverOptions(): Partial<SolverOptions> {
		const tension = this.relaxSmoothness;
		const surfaceBarrierWeight = 0.6 + this.repulsionStrength * 1.5;
		const surfacePenetrationWeight = 160 + this.repulsionStrength * 240;
		const stretchWeight = 220 + tension * 430;
		const bendingWeight = 10 + tension * 78;
		const constraintPasses = 10 + Math.round(this.repulsionStrength * 2 + tension * 6);
		const edgeCollisionPasses = 2 + Math.round(this.repulsionStrength * 0.7);
		return {
			repulsionWeight: this.repulsionStrength,
			smoothingWeight: 0.01 + tension * 0.16,
			highOrderWeight: 0.2 + tension * 0.52,
			bendingWeight,
			stretchWeight,
			constraintPasses,
			edgeCollisionPasses,
			surfaceBarrierWeight,
			surfacePenetrationWeight,
			surfaceClearanceFactor: 2.06,
			surfaceNearFieldFactor: 1.4 + tension * 0.33
		};
	}

	private applySolverTuning(): void {
		if (!this.solver) return;
		const tension = this.relaxSmoothness;
		const surfaceBarrierWeight = 0.6 + this.repulsionStrength * 1.5;
		const surfacePenetrationWeight = 160 + this.repulsionStrength * 240;
		const stretchWeight = 220 + tension * 430;
		const bendingWeight = 10 + tension * 78;
		const constraintPasses = 10 + Math.round(this.repulsionStrength * 2 + tension * 6);
		const edgeCollisionPasses = 2 + Math.round(this.repulsionStrength * 0.7);
		this.solver.setTuning({
			repulsionWeight: this.repulsionStrength,
			smoothingWeight: 0.01 + tension * 0.16,
			highOrderWeight: 0.2 + tension * 0.52,
			bendingWeight,
			stretchWeight,
			constraintPasses,
			edgeCollisionPasses,
			surfaceBarrierWeight,
			surfacePenetrationWeight,
			surfaceClearanceFactor: 2.06,
			surfaceNearFieldFactor: 1.4 + tension * 0.33
		});
		for (const solver of this.passiveSolvers) {
			solver.setTuning({
				repulsionWeight: this.repulsionStrength,
				smoothingWeight: 0.01 + tension * 0.16,
				highOrderWeight: 0.2 + tension * 0.52,
				bendingWeight,
				stretchWeight,
				constraintPasses,
				edgeCollisionPasses,
				surfaceBarrierWeight,
				surfacePenetrationWeight,
				surfaceClearanceFactor: 2.06,
				surfaceNearFieldFactor: 1.4 + tension * 0.33
			});
		}
	}

	private stepAllComponents(iterations: number, dt: number, dragSelection: DragSelection | null): void {
		if (!this.state) return;
		const steps = Math.max(1, iterations);
		for (let iteration = 0; iteration < steps; iteration += 1) {
			this.relaxTick += 1;
			this.forEachDynamicComponent((componentIndex, points, solver) => {
				const pinnedIndex =
					dragSelection && dragSelection.componentIndex === componentIndex ? dragSelection.pointIndex : null;
				solver.step(points, dt, pinnedIndex, this.state!.thickness);
			});
			this.applyInterComponentRepulsion(1, dragSelection);
			this.forEachDynamicComponent((componentIndex, points, solver) => {
				const pinnedIndex =
					dragSelection && dragSelection.componentIndex === componentIndex ? dragSelection.pointIndex : null;
				solver.enforceInextensibility(points, pinnedIndex, this.state!.thickness, 1);
			});
			if (dragSelection === null && this.relaxTick % 3 === 0) {
				this.reparameterizeAllComponents(null);
			}
			if (dragSelection === null && this.state.componentCount > 1) {
				this.recenterAllComponents();
			}
		}
	}

	private forEachDynamicComponent(
		visitor: (componentIndex: number, points: Float32Array, solver: KnotSolver) => void
	): void {
		if (this.points && this.solver) visitor(0, this.points, this.solver);
		for (let componentIndex = 0; componentIndex < this.passiveComponents.length; componentIndex += 1) {
			const points = this.passiveComponents[componentIndex];
			const solver = this.passiveSolvers[componentIndex];
			if (!points || !solver) continue;
			visitor(componentIndex + 1, points, solver);
		}
	}

	private rebuildGeometry(forceControlRebuild = false): void {
		const scene = this.scene;
		const state = this.state;
		const points = this.points;
		if (!scene || !state || !points) return;

		const tubeGeometry = createTubeGeometry(points, state.thickness);
		if (!this.tubeMesh) {
			const tubeMaterial = new THREE.MeshPhysicalMaterial({
				color: TUBE_COLOR,
				roughness: 0.23,
				metalness: 0.06,
				clearcoat: 0.7,
				clearcoatRoughness: 0.24
			});
			this.tubeMesh = new THREE.Mesh(tubeGeometry, tubeMaterial);
			this.tubeMesh.castShadow = false;
			this.tubeMesh.receiveShadow = false;
			scene.add(this.tubeMesh);
		} else {
			this.tubeMesh.geometry.dispose();
			this.tubeMesh.geometry = tubeGeometry;
		}
		this.tubeMesh.visible = this.showSolidLink;

		const lineGeometry = createLineGeometry(points);
		if (!this.lineMesh) {
			const lineMaterial = new THREE.LineBasicMaterial({
				color: LINE_COLOR,
				transparent: true,
				opacity: 0.92
			});
			this.lineMesh = new THREE.LineLoop(lineGeometry, lineMaterial);
			this.lineMesh.visible = !this.showSolidLink;
			scene.add(this.lineMesh);
		} else {
			this.lineMesh.geometry.dispose();
			this.lineMesh.geometry = lineGeometry;
			this.lineMesh.visible = !this.showSolidLink;
		}
		this.syncPassiveComponentGeometry(forceControlRebuild);

		const expectedCount = points.length / 3;
		if (
			forceControlRebuild ||
			!this.controlMesh ||
			this.controlMesh.count !== expectedCount ||
			!(this.controlMesh.geometry instanceof THREE.SphereGeometry)
		) {
			if (this.controlMesh) {
				scene.remove(this.controlMesh);
				this.controlMesh.geometry.dispose();
				(this.controlMesh.material as THREE.Material).dispose();
			}
			const pointGeometry = new THREE.SphereGeometry(state.thickness * 0.45, 8, 8);
			const pointMaterial = new THREE.MeshBasicMaterial({
				color: '#46f5ce',
				transparent: true,
				opacity: 0.35
			});
			this.controlMesh = new THREE.InstancedMesh(pointGeometry, pointMaterial, expectedCount);
			this.controlMesh.visible = this.showControlPoints;
			this.controlMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
			scene.add(this.controlMesh);
		}

		this.applyComponentColors();
		this.updateControlPoints();
		this.geometryDirty = false;
	}

	private updateControlPoints(): void {
		if (!this.controlMesh || !this.points) return;
		const count = this.points.length / 3;
		for (let i = 0; i < count; i += 1) {
			const idx = i * 3;
			this.tempObject.position.set(this.points[idx], this.points[idx + 1], this.points[idx + 2]);
			this.tempObject.updateMatrix();
			this.controlMesh.setMatrixAt(i, this.tempObject.matrix);
		}
		this.controlMesh.instanceMatrix.needsUpdate = true;
	}

	private syncPassiveComponentGeometry(forceRebuild = false): void {
		const scene = this.scene;
		const state = this.state;
		if (!scene || !state) return;
		const expectedCount = this.passiveComponents.length;
		const needsRebuild =
			forceRebuild ||
			this.passiveTubeMeshes.length !== expectedCount ||
			this.passiveLineMeshes.length !== expectedCount;
		if (needsRebuild) {
			this.clearPassiveGeometry();
			for (let i = 0; i < this.passiveComponents.length; i += 1) {
				const points = this.passiveComponents[i];
				const tubeMaterial = new THREE.MeshPhysicalMaterial({
					color: TUBE_COLOR,
					roughness: 0.23,
					metalness: 0.06,
					clearcoat: 0.7,
					clearcoatRoughness: 0.24
				});
				const tubeMesh = new THREE.Mesh(createTubeGeometry(points, state.thickness), tubeMaterial);
				tubeMesh.castShadow = false;
				tubeMesh.receiveShadow = false;
				tubeMesh.visible = this.showSolidLink;
				scene.add(tubeMesh);
				this.passiveTubeMeshes.push(tubeMesh);

				const lineMaterial = new THREE.LineBasicMaterial({
					color: LINE_COLOR,
					transparent: true,
					opacity: 0.92
				});
				const lineMesh = new THREE.LineLoop(createLineGeometry(points), lineMaterial);
				lineMesh.visible = !this.showSolidLink;
				scene.add(lineMesh);
				this.passiveLineMeshes.push(lineMesh);
			}
			return;
		}

		for (let i = 0; i < this.passiveComponents.length; i += 1) {
			const points = this.passiveComponents[i];
			const tubeMesh = this.passiveTubeMeshes[i];
			const lineMesh = this.passiveLineMeshes[i];
			tubeMesh.geometry.dispose();
			tubeMesh.geometry = createTubeGeometry(points, state.thickness);
			tubeMesh.visible = this.showSolidLink;
			lineMesh.geometry.dispose();
			lineMesh.geometry = createLineGeometry(points);
			lineMesh.visible = !this.showSolidLink;
		}
	}

	private clearPassiveGeometry(): void {
		for (const mesh of this.passiveTubeMeshes) {
			this.scene?.remove(mesh);
			mesh.geometry.dispose();
			(mesh.material as THREE.Material).dispose();
		}
		for (const mesh of this.passiveLineMeshes) {
			this.scene?.remove(mesh);
			mesh.geometry.dispose();
			(mesh.material as THREE.Material).dispose();
		}
		this.passiveTubeMeshes = [];
		this.passiveLineMeshes = [];
	}

	private applyComponentColors(): void {
		if (this.tubeMesh) {
			const material = this.tubeMesh.material as THREE.MeshPhysicalMaterial;
			material.color.set(componentTubeColor(this.colorizeLinks, 0));
		}
		if (this.lineMesh) {
			const material = this.lineMesh.material as THREE.LineBasicMaterial;
			material.color.set(componentLineColor(this.colorizeLinks, 0));
		}
		for (let i = 0; i < this.passiveTubeMeshes.length; i += 1) {
			const material = this.passiveTubeMeshes[i].material as THREE.MeshPhysicalMaterial;
			material.color.set(componentTubeColor(this.colorizeLinks, i + 1));
		}
		for (let i = 0; i < this.passiveLineMeshes.length; i += 1) {
			const material = this.passiveLineMeshes[i].material as THREE.LineBasicMaterial;
			material.color.set(componentLineColor(this.colorizeLinks, i + 1));
		}
	}

	private getComponentPoints(componentIndex: number): Float32Array | null {
		if (componentIndex === 0) return this.points;
		return this.passiveComponents[componentIndex - 1] ?? null;
	}

	private getComponentSolver(componentIndex: number): KnotSolver | null {
		if (componentIndex === 0) return this.solver;
		return this.passiveSolvers[componentIndex - 1] ?? null;
	}

	private getComponentRestLength(componentIndex: number): number {
		if (!this.state) return 0;
		if (componentIndex === 0) return this.state.componentRestLengths[0] ?? this.state.restLength;
		const points = this.passiveComponents[componentIndex - 1];
		const fallback = points ? averageEdgeLength(points) : 0;
		return this.state.componentRestLengths[componentIndex] ?? fallback;
	}

	private resolveComponentIndexFromTubeObject(object: THREE.Object3D): number | null {
		let current: THREE.Object3D | null = object;
		while (current) {
			if (current === this.tubeMesh) return 0;
			const passiveIndex = this.passiveTubeMeshes.findIndex((mesh) => mesh === current);
			if (passiveIndex >= 0) return passiveIndex + 1;
			current = current.parent;
		}
		return null;
	}

	private emitMetrics(): void {
		if (!this.onMetrics || !this.points || !this.solver || !this.state) return;
		const totalNodes =
			this.points.length / 3 + this.passiveComponents.reduce((sum, component) => sum + component.length / 3, 0);
		let energy = this.solver.measureEnergy(this.points, this.state.thickness);
		const primaryDiagnostics = this.solver.getDiagnostics();
		let minClearance = primaryDiagnostics.lastMinEdgeDistance;
		let clearanceRatio = primaryDiagnostics.lastClearanceRatio;
		let acceptedSteps = primaryDiagnostics.acceptedSteps;
		let rejectedSteps = primaryDiagnostics.rejectedSteps;
		for (let componentIndex = 0; componentIndex < this.passiveComponents.length; componentIndex += 1) {
			const passiveSolver = this.passiveSolvers[componentIndex];
			const passivePoints = this.passiveComponents[componentIndex];
			if (!passiveSolver || !passivePoints) continue;
			energy += passiveSolver.measureEnergy(passivePoints, this.state.thickness);
			const diagnostics = passiveSolver.getDiagnostics();
			if (diagnostics.lastMinEdgeDistance < minClearance) minClearance = diagnostics.lastMinEdgeDistance;
			if (diagnostics.lastClearanceRatio < clearanceRatio) clearanceRatio = diagnostics.lastClearanceRatio;
			acceptedSteps += diagnostics.acceptedSteps;
			rejectedSteps += diagnostics.rejectedSteps;
		}
		const interComponentMin = this.measureInterComponentMinDistance();
		if (interComponentMin < minClearance) minClearance = interComponentMin;
		clearanceRatio = minClearance / Math.max(1e-6, this.state.thickness * 2.06);
		const maxLengthDrift = this.measureComponentLengthDrift();
		const edgeUniformity = this.measureComponentEdgeUniformity();
		this.onMetrics({
			preset: this.state.name,
			label: this.state.label,
			crossings: estimateCrossings(this.points),
			targetCrossings: this.state.targetCrossings,
			energy,
			nodeCount: totalNodes,
			minClearance,
			clearanceRatio,
			maxLengthDrift,
			edgeUniformity,
			acceptedSteps,
			rejectedSteps
		});
	}

	private observeResize(): void {
		this.resizeObserver = new ResizeObserver(() => this.resize());
		this.resizeObserver.observe(this.container);
		this.resize();
	}

	private async createRenderer(): Promise<THREE.WebGPURenderer | WebGLRenderer> {
		try {
			if (!WebGPU.isAvailable()) throw new Error('WebGPU unavailable');
			const renderer = new THREE.WebGPURenderer({
				antialias: true
			});
			await renderer.init();
			const deviceLimit = (
				renderer as unknown as { backend?: { device?: GPUDevice & { limits?: GPUSupportedLimits } } }
			).backend?.device?.limits?.maxTextureDimension2D;
			if (typeof deviceLimit === 'number' && Number.isFinite(deviceLimit) && deviceLimit > 0) {
				this.maxRenderDimension2D = Math.min(
					Math.floor(deviceLimit),
					KnotEngine.DEFAULT_WEBGPU_TEXTURE_LIMIT
				);
			} else {
				this.maxRenderDimension2D = KnotEngine.DEFAULT_WEBGPU_TEXTURE_LIMIT;
			}
			return renderer;
		} catch {
			const webglRenderer = new CoreWebGLRenderer({ antialias: true });
			this.maxRenderDimension2D = Math.min(
				KnotEngine.DEFAULT_WEBGPU_TEXTURE_LIMIT,
				webglRenderer.capabilities.maxTextureSize
			);
			return webglRenderer;
		}
	}

	private resize(): void {
		if (!this.renderer || !this.camera) return;
		const width = Math.max(1, Math.floor(this.container.clientWidth));
		const height = Math.max(1, Math.floor(this.container.clientHeight));
		const basePixelRatio = Math.min(
			KnotEngine.MAX_TARGET_PIXEL_RATIO,
			Math.max(1, window.devicePixelRatio || 1)
		);
		const textureLimit = Math.max(1024, Math.min(this.maxRenderDimension2D, 8192) - 1);

		let renderWidth = Math.max(1, Math.floor(width * basePixelRatio));
		let renderHeight = Math.max(1, Math.floor(height * basePixelRatio));
		const scaleToLimit = Math.min(1, textureLimit / renderWidth, textureLimit / renderHeight);
		if (scaleToLimit < 1) {
			renderWidth = Math.max(1, Math.floor(renderWidth * scaleToLimit));
			renderHeight = Math.max(1, Math.floor(renderHeight * scaleToLimit));
		}

		this.renderer.setPixelRatio(1);
		this.camera.aspect = width / height;
		this.camera.updateProjectionMatrix();
		this.renderer.setSize(renderWidth, renderHeight, false);
	}

	private attachInteractionHandlers(): void {
		this.renderer?.domElement.addEventListener('pointerdown', this.handlePointerDown);
		window.addEventListener('pointermove', this.handlePointerMove);
		window.addEventListener('pointerup', this.handlePointerUp);
		window.addEventListener('pointercancel', this.handlePointerUp);
	}

	private detachInteractionHandlers(): void {
		this.renderer?.domElement.removeEventListener('pointerdown', this.handlePointerDown);
		window.removeEventListener('pointermove', this.handlePointerMove);
		window.removeEventListener('pointerup', this.handlePointerUp);
		window.removeEventListener('pointercancel', this.handlePointerUp);
	}

	private handlePointerDown = (event: PointerEvent): void => {
		if (!this.camera || !this.renderer || !this.points || !this.controls) return;
		const dom = this.renderer.domElement;
		const rect = dom.getBoundingClientRect();
		const x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
		const y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
		this.pointer.set(x, y);
		this.raycaster.setFromCamera(this.pointer, this.camera);

		let pickedComponent: number | null = null;
		let pickedPoint: number | null = null;
		if (this.controlMesh && this.showControlPoints) {
			const pointHits = this.raycaster.intersectObject(this.controlMesh, false);
			const instanceId = pointHits[0]?.instanceId;
			if (typeof instanceId === 'number') {
				pickedComponent = 0;
				pickedPoint = instanceId;
			}
		}

		if (pickedPoint === null) {
			const targets: THREE.Object3D[] = [];
			if (this.tubeMesh) targets.push(this.tubeMesh);
			for (const mesh of this.passiveTubeMeshes) targets.push(mesh);
			const hits = this.raycaster.intersectObjects(targets, false);
			for (const hit of hits) {
				const componentIndex = this.resolveComponentIndexFromTubeObject(hit.object);
				if (componentIndex === null) continue;
				const points = this.getComponentPoints(componentIndex);
				if (!points) continue;
				pickedComponent = componentIndex;
				pickedPoint = nearestPointIndex(points, hit.point);
				break;
			}
		}

		if (pickedComponent === null || pickedPoint === null) return;
		const points = this.getComponentPoints(pickedComponent);
		const solver = this.getComponentSolver(pickedComponent);
		if (!points || !solver) return;

		this.dragComponentIndex = pickedComponent;
		this.dragIndex = pickedPoint;
		solver.zeroVelocity(pickedPoint);
		this.dragPoint.copy(getPoint(points, pickedPoint));
		this.dragTarget.copy(this.dragPoint);
		this.camera.getWorldDirection(this.cameraDirection);
		this.dragPlane.setFromNormalAndCoplanarPoint(this.cameraDirection, this.dragPoint);
		this.controls.enabled = false;
		dom.setPointerCapture(event.pointerId);
		this.notifyStatus(
			pickedComponent === 0 ? 'Dragging a knot segment.' : `Dragging link ${pickedComponent + 1} segment.`
		);
	};

	private handlePointerMove = (event: PointerEvent): void => {
		if (
			this.dragIndex === null ||
			this.dragComponentIndex === null ||
			!this.camera ||
			!this.renderer ||
			!this.state
		) {
			return;
		}
		const rect = this.renderer.domElement.getBoundingClientRect();
		const x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
		const y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
		this.pointer.set(x, y);
		this.raycaster.setFromCamera(this.pointer, this.camera);
		if (!this.raycaster.ray.intersectPlane(this.dragPlane, this.dragPoint)) return;
		this.dragTarget.copy(this.dragPoint);
	};

	private handlePointerUp = (): void => {
		if (this.dragIndex === null || this.dragComponentIndex === null || !this.controls) return;
		this.dragIndex = null;
		this.dragComponentIndex = null;
		this.controls.enabled = true;
		this.notifyStatus('Released segment. Keep relaxing or continue dragging.');
	};

	private disposeGeometry(): void {
		if (this.tubeMesh) {
			this.scene?.remove(this.tubeMesh);
			this.tubeMesh.geometry.dispose();
			(this.tubeMesh.material as THREE.Material).dispose();
			this.tubeMesh = null;
		}
		if (this.lineMesh) {
			this.scene?.remove(this.lineMesh);
			this.lineMesh.geometry.dispose();
			(this.lineMesh.material as THREE.Material).dispose();
			this.lineMesh = null;
		}
		this.clearPassiveGeometry();
		this.passiveSolvers = [];
		if (this.controlMesh) {
			this.scene?.remove(this.controlMesh);
			this.controlMesh.geometry.dispose();
			(this.controlMesh.material as THREE.Material).dispose();
			this.controlMesh = null;
		}
	}

	private applyDragTarget(): void {
		if (this.dragIndex === null || this.dragComponentIndex === null || !this.state) return;
		const points = this.getComponentPoints(this.dragComponentIndex);
		const solver = this.getComponentSolver(this.dragComponentIndex);
		const restLength = this.getComponentRestLength(this.dragComponentIndex);
		if (!points || !solver || !Number.isFinite(restLength) || restLength <= 0) return;
		const count = points.length / 3;
		const idx = this.dragIndex * 3;
		const current = new THREE.Vector3(points[idx], points[idx + 1], points[idx + 2]);
		const delta = this.dragTarget.clone().sub(current);
		const maxMove = restLength * 0.52;
		const length = delta.length();
		if (length < 1e-6) return;
		if (length > maxMove) delta.multiplyScalar(maxMove / length);

		const radius = 7;
		const sigma = radius * 0.44;
		for (let i = 0; i < count; i += 1) {
			const d = ringDistance(i, this.dragIndex, count);
			if (d > radius) continue;
			const w = Math.exp(-(d * d) / (2 * sigma * sigma));
			const base = i * 3;
			points[base] += delta.x * w;
			points[base + 1] += delta.y * w;
			points[base + 2] += delta.z * w;
		}

		for (let pass = 0; pass < 3; pass += 1) {
			for (let offset = -radius - 1; offset <= radius + 1; offset += 1) {
				const a = modIndex(this.dragIndex + offset, count);
				const b = (a + 1) % count;
				projectEdgeLength(points, a, b, restLength);
			}
		}
		projectComponentEdgeLengths(points, restLength, 8, this.dragIndex);
		solver.enforceInextensibility(points, this.dragIndex, this.state.thickness, 1);

		solver.zeroVelocity(this.dragIndex);
	}

	private applyInterComponentRepulsion(passes: number, dragSelection: DragSelection | null): void {
		if (!this.state || !this.points || this.passiveComponents.length === 0) return;
		const components: Float32Array[] = [this.points, ...this.passiveComponents];
		const clearance = this.state.thickness * 2.06;
		const nearField = clearance * (1.38 + this.relaxSmoothness * 0.34);
		const softness = Math.max(1e-4, nearField - clearance);
		const penetrationScale = 0.34 + this.repulsionStrength * 0.26;
		const nearScale = 0.012 + this.repulsionStrength * 0.018;

		for (let pass = 0; pass < passes; pass += 1) {
			for (let componentA = 0; componentA < components.length; componentA += 1) {
				const pointsA = components[componentA];
				const countA = pointsA.length / 3;
				if (countA < 2) continue;
				for (let componentB = componentA + 1; componentB < components.length; componentB += 1) {
					const pointsB = components[componentB];
					const countB = pointsB.length / 3;
					if (countB < 2) continue;
					for (let edgeA = 0; edgeA < countA; edgeA += 1) {
						const edgeANext = (edgeA + 1) % countA;
						const a0 = edgeA * 3;
						const a1 = edgeANext * 3;
						const minAx = Math.min(pointsA[a0], pointsA[a1]) - nearField;
						const minAy = Math.min(pointsA[a0 + 1], pointsA[a1 + 1]) - nearField;
						const minAz = Math.min(pointsA[a0 + 2], pointsA[a1 + 2]) - nearField;
						const maxAx = Math.max(pointsA[a0], pointsA[a1]) + nearField;
						const maxAy = Math.max(pointsA[a0 + 1], pointsA[a1 + 1]) + nearField;
						const maxAz = Math.max(pointsA[a0 + 2], pointsA[a1 + 2]) + nearField;

						for (let edgeB = 0; edgeB < countB; edgeB += 1) {
							const edgeBNext = (edgeB + 1) % countB;
							const b0 = edgeB * 3;
							const b1 = edgeBNext * 3;

							const minBx = Math.min(pointsB[b0], pointsB[b1]);
							const minBy = Math.min(pointsB[b0 + 1], pointsB[b1 + 1]);
							const minBz = Math.min(pointsB[b0 + 2], pointsB[b1 + 2]);
							const maxBx = Math.max(pointsB[b0], pointsB[b1]);
							const maxBy = Math.max(pointsB[b0 + 1], pointsB[b1 + 1]);
							const maxBz = Math.max(pointsB[b0 + 2], pointsB[b1 + 2]);
							if (
								maxBx < minAx ||
								maxBy < minAy ||
								maxBz < minAz ||
								minBx > maxAx ||
								minBy > maxAy ||
								minBz > maxAz
							) {
								continue;
							}

							const contact = closestPointsBetweenComponentEdges(
								pointsA,
								edgeA,
								edgeANext,
								pointsB,
								edgeB,
								edgeBNext
							);
							const dist = Math.sqrt(contact.distSq);
							if (!Number.isFinite(dist) || dist >= nearField) continue;

							let nx = contact.dx;
							let ny = contact.dy;
							let nz = contact.dz;
							let normalLength = Math.hypot(nx, ny, nz);
							if (normalLength < 1e-7) {
								nx = pointsA[a1] - pointsA[a0];
								ny = pointsA[a1 + 1] - pointsA[a0 + 1];
								nz = pointsA[a1 + 2] - pointsA[a0 + 2];
								normalLength = Math.hypot(nx, ny, nz) || 1;
							}
							nx /= normalLength;
							ny /= normalLength;
							nz /= normalLength;

							let push = 0;
							if (dist < clearance) {
								push = (clearance - dist) * penetrationScale;
							} else {
								const blend = (nearField - dist) / softness;
								push = clearance * blend * blend * nearScale;
							}
							if (push <= 0) continue;

							const a0Weight = 1 - contact.s;
							const a1Weight = contact.s;
							const b0Weight = 1 - contact.t;
							const b1Weight = contact.t;

								displaceComponentPoint(
									pointsA,
									edgeA,
									nx * push * a0Weight,
									ny * push * a0Weight,
									nz * push * a0Weight,
									dragSelection && dragSelection.componentIndex === componentA
										? dragSelection.pointIndex
										: null
								);
								displaceComponentPoint(
									pointsA,
									edgeANext,
									nx * push * a1Weight,
									ny * push * a1Weight,
									nz * push * a1Weight,
									dragSelection && dragSelection.componentIndex === componentA
										? dragSelection.pointIndex
										: null
								);
								displaceComponentPoint(
									pointsB,
								edgeB,
									-nx * push * b0Weight,
									-ny * push * b0Weight,
									-nz * push * b0Weight,
									dragSelection && dragSelection.componentIndex === componentB
										? dragSelection.pointIndex
										: null
								);
								displaceComponentPoint(
									pointsB,
								edgeBNext,
									-nx * push * b1Weight,
									-ny * push * b1Weight,
									-nz * push * b1Weight,
									dragSelection && dragSelection.componentIndex === componentB
										? dragSelection.pointIndex
										: null
								);
						}
					}
				}
			}

			for (let componentIndex = 0; componentIndex < components.length; componentIndex += 1) {
				const points = components[componentIndex];
				const restLength = this.getComponentRestLength(componentIndex);
				const pinnedIndex =
					dragSelection && dragSelection.componentIndex === componentIndex ? dragSelection.pointIndex : null;
				projectComponentEdgeLengths(points, restLength, 8, pinnedIndex);
			}
		}
	}

	private reparameterizeAllComponents(dragSelection: DragSelection | null): void {
		const components: Float32Array[] = [];
		if (this.points) components.push(this.points);
		for (const component of this.passiveComponents) components.push(component);
		for (let componentIndex = 0; componentIndex < components.length; componentIndex += 1) {
			const points = components[componentIndex];
			if (!needsArclengthRedistribution(points)) continue;
			const pinnedIndex =
				dragSelection && dragSelection.componentIndex === componentIndex ? dragSelection.pointIndex : null;
			redistributeClosedCurveArclength(points, pinnedIndex);
			const restLength = this.getComponentRestLength(componentIndex);
			projectComponentEdgeLengths(points, restLength, 3, pinnedIndex);
		}
	}

	private recenterAllComponents(): void {
		const components: Float32Array[] = [];
		if (this.points) components.push(this.points);
		for (const component of this.passiveComponents) components.push(component);
		if (components.length === 0) return;

		let count = 0;
		let cx = 0;
		let cy = 0;
		let cz = 0;
		for (const points of components) {
			const pointCount = points.length / 3;
			for (let i = 0; i < pointCount; i += 1) {
				const base = i * 3;
				cx += points[base];
				cy += points[base + 1];
				cz += points[base + 2];
				count += 1;
			}
		}
		if (count === 0) return;
		cx /= count;
		cy /= count;
		cz /= count;

		for (const points of components) {
			const pointCount = points.length / 3;
			for (let i = 0; i < pointCount; i += 1) {
				const base = i * 3;
				points[base] -= cx;
				points[base + 1] -= cy;
				points[base + 2] -= cz;
			}
		}
	}

	private measureInterComponentMinDistance(): number {
		if (!this.points || this.passiveComponents.length === 0) return Number.POSITIVE_INFINITY;
		const components: Float32Array[] = [this.points, ...this.passiveComponents];
		let minDistance = Number.POSITIVE_INFINITY;
		for (let componentA = 0; componentA < components.length; componentA += 1) {
			const pointsA = components[componentA];
			const countA = pointsA.length / 3;
			if (countA < 2) continue;
			for (let componentB = componentA + 1; componentB < components.length; componentB += 1) {
				const pointsB = components[componentB];
				const countB = pointsB.length / 3;
				if (countB < 2) continue;
				for (let edgeA = 0; edgeA < countA; edgeA += 1) {
					const edgeANext = (edgeA + 1) % countA;
					for (let edgeB = 0; edgeB < countB; edgeB += 1) {
						const edgeBNext = (edgeB + 1) % countB;
						const contact = closestPointsBetweenComponentEdges(
							pointsA,
							edgeA,
							edgeANext,
							pointsB,
							edgeB,
							edgeBNext
						);
						const dist = Math.sqrt(contact.distSq);
						if (!Number.isFinite(dist)) continue;
						if (dist < minDistance) minDistance = dist;
					}
				}
			}
		}
		return minDistance;
	}

	private measureComponentLengthDrift(): number {
		if (!this.state || !this.points) return 0;
		const components: Float32Array[] = [this.points, ...this.passiveComponents];
		let worst = 0;
		for (let componentIndex = 0; componentIndex < components.length; componentIndex += 1) {
			const points = components[componentIndex];
			const count = points.length / 3;
			if (count < 2) continue;
			const restLength = this.getComponentRestLength(componentIndex);
			const restTotal = restLength * count;
			if (!Number.isFinite(restTotal) || restTotal <= 1e-9) continue;
			let total = 0;
			for (let i = 0; i < count; i += 1) {
				const j = (i + 1) % count;
				const ia = i * 3;
				const ja = j * 3;
				total += Math.hypot(
					points[ia] - points[ja],
					points[ia + 1] - points[ja + 1],
					points[ia + 2] - points[ja + 2]
				);
			}
			const drift = Math.abs(total / restTotal - 1);
			if (drift > worst) worst = drift;
		}
		return worst;
	}

	private measureComponentEdgeUniformity(): number {
		const components: Float32Array[] = [];
		if (this.points) components.push(this.points);
		for (const component of this.passiveComponents) components.push(component);
		let worstRatio = 1;
		for (const points of components) {
			const count = points.length / 3;
			if (count < 2) continue;
			let minEdge = Number.POSITIVE_INFINITY;
			let maxEdge = 0;
			for (let i = 0; i < count; i += 1) {
				const j = (i + 1) % count;
				const ia = i * 3;
				const ja = j * 3;
				const edgeLength = Math.hypot(
					points[ia] - points[ja],
					points[ia + 1] - points[ja + 1],
					points[ia + 2] - points[ja + 2]
				);
				if (!Number.isFinite(edgeLength)) continue;
				if (edgeLength < minEdge) minEdge = edgeLength;
				if (edgeLength > maxEdge) maxEdge = edgeLength;
			}
			if (!Number.isFinite(minEdge) || minEdge <= 1e-8) continue;
			const ratio = maxEdge / minEdge;
			if (ratio > worstRatio) worstRatio = ratio;
		}
		return worstRatio;
	}

	private fitViewToState(): void {
		if (!this.camera || !this.controls || !this.points) return;
		const bounds = computeBoundsForComponents([this.points, ...this.passiveComponents]);
		const center = bounds.center;
		const radius = bounds.radius;

		const direction = this.camera.position.clone().sub(this.controls.target);
		if (direction.lengthSq() < 1e-8) direction.set(0, 0, 1);
		direction.normalize();
		const distance = Math.max(6.5, radius * 3.2);
		this.controls.target.copy(center);
		this.camera.position.copy(center).addScaledVector(direction, distance);
		this.controls.minDistance = Math.max(2.2, radius * 0.7);
		this.controls.maxDistance = Math.max(90, radius * 80);
		const far = Math.max(260, distance + radius * 130 + 8);
		this.camera.near = Math.max(0.02, Math.min(0.25, far / 5000));
		this.camera.far = far;
		this.camera.updateProjectionMatrix();
		this.controls.update();
	}

	private updateCameraForCurrentBounds(): void {
		if (!this.camera || !this.controls || !this.points) return;
		const bounds = computeBoundsForComponents([this.points, ...this.passiveComponents]);
		const center = bounds.center;
		const radius = bounds.radius;
		const cameraDistance = this.camera.position.distanceTo(this.controls.target);
		const farNeeded = Math.max(260, cameraDistance + radius * 130 + 8);
		const nearNeeded = Math.max(0.02, Math.min(0.25, farNeeded / 5000));

		let projectionDirty = false;
		if (Math.abs(this.camera.far - farNeeded) > 6) {
			this.camera.far = farNeeded;
			projectionDirty = true;
		}
		if (Math.abs(this.camera.near - nearNeeded) > 1e-3) {
			this.camera.near = nearNeeded;
			projectionDirty = true;
		}
		if (projectionDirty) {
			this.camera.updateProjectionMatrix();
		}

		this.controls.minDistance = Math.max(2.2, radius * 0.7);
		this.controls.maxDistance = Math.max(90, radius * 80);

		// Keep target glued to the evolving knot center so orbit/zoom stay intuitive.
		this.controls.target.lerp(center, 0.18);
	}

	private notifyStatus(message: string): void {
		if (this.onStatus) this.onStatus(message);
	}
}

function createTubeGeometry(points: Float32Array, thickness: number): THREE.BufferGeometry {
	const count = points.length / 3;
	const vectors: THREE.Vector3[] = [];
	for (let i = 0; i < count; i += 1) {
		const idx = i * 3;
		vectors.push(new THREE.Vector3(points[idx], points[idx + 1], points[idx + 2]));
	}
	const path = new THREE.CatmullRomCurve3(vectors, true, 'centripetal', 0.2);
	const tubularSegments = Math.max(280, count * 4);
	return new THREE.TubeGeometry(path, tubularSegments, thickness, 20, true);
}

function createLineGeometry(points: Float32Array): THREE.BufferGeometry {
	const geometry = new THREE.BufferGeometry();
	geometry.setAttribute('position', new THREE.BufferAttribute(points.slice(), 3));
	return geometry;
}

function computeBoundsForComponents(components: Float32Array[]): { center: THREE.Vector3; radius: number } {
	const center = new THREE.Vector3();
	let totalCount = 0;
	for (const points of components) {
		const count = points.length / 3;
		for (let i = 0; i < count; i += 1) {
			const base = i * 3;
			center.x += points[base];
			center.y += points[base + 1];
			center.z += points[base + 2];
			totalCount += 1;
		}
	}
	if (totalCount === 0) return { center, radius: 0.1 };
	center.multiplyScalar(1 / totalCount);

	let radius = 0.1;
	for (const points of components) {
		const count = points.length / 3;
		for (let i = 0; i < count; i += 1) {
			const base = i * 3;
			const dx = points[base] - center.x;
			const dy = points[base + 1] - center.y;
			const dz = points[base + 2] - center.z;
			radius = Math.max(radius, Math.hypot(dx, dy, dz));
		}
	}

	return { center, radius };
}

function nearestPointIndex(points: Float32Array, target: THREE.Vector3): number {
	const count = points.length / 3;
	let best = 0;
	let bestDistSq = Number.POSITIVE_INFINITY;
	for (let i = 0; i < count; i += 1) {
		const idx = i * 3;
		const dx = target.x - points[idx];
		const dy = target.y - points[idx + 1];
		const dz = target.z - points[idx + 2];
		const d2 = dx * dx + dy * dy + dz * dz;
		if (d2 < bestDistSq) {
			bestDistSq = d2;
			best = i;
		}
	}
	return best;
}

function getPoint(points: Float32Array, index: number): THREE.Vector3 {
	const idx = index * 3;
	return new THREE.Vector3(points[idx], points[idx + 1], points[idx + 2]);
}

function ringDistance(a: number, b: number, count: number): number {
	const diff = Math.abs(a - b);
	return Math.min(diff, count - diff);
}

function modIndex(index: number, count: number): number {
	const value = index % count;
	return value < 0 ? value + count : value;
}

function projectEdgeLength(points: Float32Array, a: number, b: number, targetLength: number): void {
	const ia = a * 3;
	const ib = b * 3;
	const dx = points[ib] - points[ia];
	const dy = points[ib + 1] - points[ia + 1];
	const dz = points[ib + 2] - points[ia + 2];
	const dist = Math.hypot(dx, dy, dz) || 1e-8;
	const correction = ((dist - targetLength) / dist) * 0.5;
	points[ia] += dx * correction;
	points[ia + 1] += dy * correction;
	points[ia + 2] += dz * correction;
	points[ib] -= dx * correction;
	points[ib + 1] -= dy * correction;
	points[ib + 2] -= dz * correction;
}

function averageEdgeLength(points: Float32Array): number {
	const count = points.length / 3;
	if (count < 2) return 0;
	let sum = 0;
	for (let i = 0; i < count; i += 1) {
		const j = (i + 1) % count;
		const ia = i * 3;
		const ja = j * 3;
		sum += Math.hypot(
			points[ia] - points[ja],
			points[ia + 1] - points[ja + 1],
			points[ia + 2] - points[ja + 2]
		);
	}
	return sum / count;
}

interface ComponentSegmentContact {
	s: number;
	t: number;
	distSq: number;
	dx: number;
	dy: number;
	dz: number;
}

function displaceComponentPoint(
	points: Float32Array,
	index: number,
	dx: number,
	dy: number,
	dz: number,
	pinnedIndex: number | null
): void {
	if (index === pinnedIndex) return;
	const base = index * 3;
	points[base] += dx;
	points[base + 1] += dy;
	points[base + 2] += dz;
}

function projectComponentEdgeLengths(
	points: Float32Array,
	targetLength: number,
	passes: number,
	pinnedIndex: number | null
): void {
	if (!Number.isFinite(targetLength) || targetLength <= 0) return;
	const count = points.length / 3;
	if (count < 2) return;
	for (let pass = 0; pass < passes; pass += 1) {
		for (let i = 0; i < count; i += 1) {
			const j = (i + 1) % count;
			const ia = i * 3;
			const ja = j * 3;
			const dx = points[ja] - points[ia];
			const dy = points[ja + 1] - points[ia + 1];
			const dz = points[ja + 2] - points[ia + 2];
			const dist = Math.hypot(dx, dy, dz) || 1e-8;
			const correction = (dist - targetLength) / dist;
			let leftWeight = 0.5;
			let rightWeight = 0.5;
			if (i === pinnedIndex) {
				leftWeight = 0;
				rightWeight = 1;
			} else if (j === pinnedIndex) {
				leftWeight = 1;
				rightWeight = 0;
			}
			if (leftWeight > 0) {
				points[ia] += dx * correction * leftWeight;
				points[ia + 1] += dy * correction * leftWeight;
				points[ia + 2] += dz * correction * leftWeight;
			}
			if (rightWeight > 0) {
				points[ja] -= dx * correction * rightWeight;
				points[ja + 1] -= dy * correction * rightWeight;
				points[ja + 2] -= dz * correction * rightWeight;
			}
		}
	}
}

function needsArclengthRedistribution(points: Float32Array): boolean {
	const count = points.length / 3;
	if (count < 8) return false;
	let total = 0;
	let minEdge = Number.POSITIVE_INFINITY;
	let maxEdge = 0;
	for (let i = 0; i < count; i += 1) {
		const j = (i + 1) % count;
		const ia = i * 3;
		const ja = j * 3;
		const edgeLength = Math.hypot(
			points[ia] - points[ja],
			points[ia + 1] - points[ja + 1],
			points[ia + 2] - points[ja + 2]
		);
		if (!Number.isFinite(edgeLength)) return false;
		total += edgeLength;
		if (edgeLength < minEdge) minEdge = edgeLength;
		if (edgeLength > maxEdge) maxEdge = edgeLength;
	}
	const meanEdge = total / count;
	if (!Number.isFinite(meanEdge) || meanEdge <= 1e-8) return false;
	if (maxEdge / Math.max(1e-8, minEdge) > 1.75) return true;
	if (maxEdge > meanEdge * 1.4) return true;
	if (minEdge < meanEdge * 0.65) return true;
	return false;
}

function redistributeClosedCurveArclength(points: Float32Array, anchorIndex: number | null): void {
	const count = points.length / 3;
	if (count < 4) return;

	const cumulative = new Float64Array(count + 1);
	let total = 0;
	for (let i = 0; i < count; i += 1) {
		const j = (i + 1) % count;
		const ia = i * 3;
		const ja = j * 3;
		total += Math.hypot(
			points[ia] - points[ja],
			points[ia + 1] - points[ja + 1],
			points[ia + 2] - points[ja + 2]
		);
		cumulative[i + 1] = total;
	}
	if (!Number.isFinite(total) || total < 1e-8) return;

	const step = total / count;
	let offset = 0;
	if (anchorIndex !== null) {
		const clamped = modIndex(anchorIndex, count);
		offset = cumulative[clamped] - clamped * step;
	}
	offset = ((offset % total) + total) % total;

	const out = new Float32Array(points.length);
	for (let i = 0; i < count; i += 1) {
		const d = (offset + i * step) % total;
		const seg = findSegmentIndex(cumulative, d);
		const segStart = cumulative[seg];
		const segEnd = cumulative[seg + 1];
		const span = Math.max(1e-8, segEnd - segStart);
		const t = (d - segStart) / span;
		const a = seg * 3;
		const b = ((seg + 1) % count) * 3;
		const base = i * 3;
		out[base] = points[a] + (points[b] - points[a]) * t;
		out[base + 1] = points[a + 1] + (points[b + 1] - points[a + 1]) * t;
		out[base + 2] = points[a + 2] + (points[b + 2] - points[a + 2]) * t;
	}
	points.set(out);
}

function findSegmentIndex(cumulative: Float64Array, distance: number): number {
	let low = 0;
	let high = cumulative.length - 2;
	while (low <= high) {
		const mid = (low + high) >>> 1;
		const a = cumulative[mid];
		const b = cumulative[mid + 1];
		if (distance < a) {
			high = mid - 1;
		} else if (distance >= b) {
			low = mid + 1;
		} else {
			return mid;
		}
	}
	return Math.max(0, Math.min(cumulative.length - 2, low));
}

function closestPointsBetweenComponentEdges(
	pointsA: Float32Array,
	a0: number,
	a1: number,
	pointsB: Float32Array,
	b0: number,
	b1: number
): ComponentSegmentContact {
	const p0x = pointsA[a0 * 3];
	const p0y = pointsA[a0 * 3 + 1];
	const p0z = pointsA[a0 * 3 + 2];
	const p1x = pointsA[a1 * 3];
	const p1y = pointsA[a1 * 3 + 1];
	const p1z = pointsA[a1 * 3 + 2];
	const q0x = pointsB[b0 * 3];
	const q0y = pointsB[b0 * 3 + 1];
	const q0z = pointsB[b0 * 3 + 2];
	const q1x = pointsB[b1 * 3];
	const q1y = pointsB[b1 * 3 + 1];
	const q1z = pointsB[b1 * 3 + 2];

	const ux = p1x - p0x;
	const uy = p1y - p0y;
	const uz = p1z - p0z;
	const vx = q1x - q0x;
	const vy = q1y - q0y;
	const vz = q1z - q0z;
	const wx = p0x - q0x;
	const wy = p0y - q0y;
	const wz = p0z - q0z;

	const a = dot3(ux, uy, uz, ux, uy, uz);
	const b = dot3(ux, uy, uz, vx, vy, vz);
	const c = dot3(vx, vy, vz, vx, vy, vz);
	const d = dot3(ux, uy, uz, wx, wy, wz);
	const e = dot3(vx, vy, vz, wx, wy, wz);
	const denom = a * c - b * b;
	const eps = 1e-6;

	let sN: number;
	let tN: number;
	let sD = denom;
	let tD = denom;

	if (denom < eps) {
		sN = 0;
		sD = 1;
		tN = e;
		tD = c;
	} else {
		sN = b * e - c * d;
		tN = a * e - b * d;
		if (sN < 0) {
			sN = 0;
			tN = e;
			tD = c;
		} else if (sN > sD) {
			sN = sD;
			tN = e + b;
			tD = c;
		}
	}

	if (tN < 0) {
		tN = 0;
		if (-d < 0) sN = 0;
		else if (-d > a) sN = sD;
		else {
			sN = -d;
			sD = a;
		}
	} else if (tN > tD) {
		tN = tD;
		if (-d + b < 0) sN = 0;
		else if (-d + b > a) sN = sD;
		else {
			sN = -d + b;
			sD = a;
		}
	}

	const s = Math.abs(sN) < eps ? 0 : sN / sD;
	const t = Math.abs(tN) < eps ? 0 : tN / tD;
	const cax = p0x + ux * s;
	const cay = p0y + uy * s;
	const caz = p0z + uz * s;
	const cbx = q0x + vx * t;
	const cby = q0y + vy * t;
	const cbz = q0z + vz * t;
	const dx = cax - cbx;
	const dy = cay - cby;
	const dz = caz - cbz;
	return { s, t, distSq: dx * dx + dy * dy + dz * dz, dx, dy, dz };
}

function dot3(ax: number, ay: number, az: number, bx: number, by: number, bz: number): number {
	return ax * bx + ay * by + az * bz;
}

function componentTubeColor(colorizeLinks: boolean, componentIndex: number): string {
	if (!colorizeLinks) return TUBE_COLOR;
	return COMPONENT_TUBE_COLORS[componentIndex % COMPONENT_TUBE_COLORS.length];
}

function componentLineColor(colorizeLinks: boolean, componentIndex: number): string {
	if (!colorizeLinks) return LINE_COLOR;
	return COMPONENT_LINE_COLORS[componentIndex % COMPONENT_LINE_COLORS.length];
}

function clampInteger(value: number, min: number, max: number): number {
	return Math.min(max, Math.max(min, Math.round(value)));
}

function clampNumber(value: number, min: number, max: number): number {
	return Math.min(max, Math.max(min, value));
}

export type { KnotPresetName };
