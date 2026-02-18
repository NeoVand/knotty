import type { WebGLRenderer } from 'three';
import * as THREE from 'three/webgpu';
import { WebGLRenderer as CoreWebGLRenderer } from 'three';
import WebGPU from 'three/examples/jsm/capabilities/WebGPU.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
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

export type KnotMaterialPreset = 'rope' | 'glass' | 'liquid_metal' | 'energy_field' | 'vector_field';

export interface KnotMaterialSettings {
	preset: KnotMaterialPreset;
	roughness: number;
	metalness: number;
	transmission: number;
	clearcoat: number;
	envMapIntensity: number;
	emissiveIntensity: number;
	animationSpeed: number;
	textureScale: number;
}

export interface KnotLightingSettings {
	exposure: number;
	ambientIntensity: number;
	keyIntensity: number;
	fillIntensity: number;
	rimIntensity: number;
}

const DEFAULT_MATERIAL_SETTINGS: KnotMaterialSettings = {
	preset: 'liquid_metal',
	roughness: 0.18,
	metalness: 1,
	transmission: 0,
	clearcoat: 0.88,
	envMapIntensity: 1.6,
	emissiveIntensity: 0.8,
	animationSpeed: 1.25,
	textureScale: 1.65
};

const DEFAULT_LIGHTING_SETTINGS: KnotLightingSettings = {
	exposure: 0.62,
	ambientIntensity: 0.12,
	keyIntensity: 1.3,
	fillIntensity: 0.06,
	rimIntensity: 0.42
};

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

interface AnimatedTextureState {
	texture: THREE.Texture;
	speed: number;
	axis: 'x' | 'y';
}

interface MaterialTextureLibrary {
	brushedMetal: THREE.CanvasTexture;
	energy: THREE.CanvasTexture;
	vectorField: THREE.CanvasTexture;
	frostedNormal: THREE.CanvasTexture;
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
	private readonly dragPlaneHit = new THREE.Vector3();
	private readonly dragOffset = new THREE.Vector3();
	private readonly cameraDirection = new THREE.Vector3();
	private readonly tempObject = new THREE.Object3D();

	private renderer: THREE.WebGPURenderer | WebGLRenderer | null = null;
	private scene: THREE.Scene | null = null;
	private camera: THREE.PerspectiveCamera | null = null;
	private controls: OrbitControls | null = null;
	private resizeObserver: ResizeObserver | null = null;
	private environmentTarget: THREE.RenderTarget | null = null;
	private ambientLight: THREE.HemisphereLight | null = null;
	private keyLight: THREE.DirectionalLight | null = null;
	private fillLight: THREE.DirectionalLight | null = null;
	private rimLight: THREE.PointLight | null = null;
	private groundPlane: THREE.Mesh<THREE.CircleGeometry, THREE.MeshPhysicalMaterial> | null = null;
	private groundPatternTexture: THREE.CanvasTexture | null = null;

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
	private colorizeLinks = true;
	private useArcGuideLayout = true;
	private repulsionStrength = 1;
	private relaxSmoothness = 0.62;
	private relaxIterations = 3;
	private lengthTargetScale = 0.88;
	private dragIndex: number | null = null;
	private dragComponentIndex: number | null = null;
	private readonly dragTarget = new THREE.Vector3();
	private geometryDirty = true;
	private disposed = false;
	private running = false;
	private lastFrameAt = 0;
	private lastMetricAt = 0;
	private lastBoundsAt = 0;
	private releaseStabilizeUntil = 0;
	private controlsInteracting = false;
	private relaxTick = 0;
	private maxRenderDimension2D = KnotEngine.DEFAULT_WEBGPU_TEXTURE_LIMIT;
	private materialSettings: KnotMaterialSettings = { ...DEFAULT_MATERIAL_SETTINGS };
	private lightingSettings: KnotLightingSettings = { ...DEFAULT_LIGHTING_SETTINGS };
	private materialTextures: MaterialTextureLibrary | null = null;
	private animatedTextures: AnimatedTextureState[] = [];

	constructor(options: KnotEngineOptions) {
		this.container = options.container;
		this.onMetrics = options.onMetrics;
		this.onStatus = options.onStatus;
	}

	async init(): Promise<void> {
		if (this.disposed) return;

		this.scene = new THREE.Scene();
		this.scene.background = new THREE.Color('#041410');
		this.scene.fog = null;

		this.camera = new THREE.PerspectiveCamera(43, 1, 0.1, 120);
		this.camera.position.set(0, 0, 7.8);

		this.renderer = await this.createRenderer();
		this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
		this.renderer.toneMappingExposure = this.lightingSettings.exposure;
		const rendererShadowMap = this.renderer as unknown as {
			shadowMap?: { enabled: boolean; type?: THREE.ShadowMapType };
		};
		if (rendererShadowMap.shadowMap) {
			rendererShadowMap.shadowMap.enabled = true;
			rendererShadowMap.shadowMap.type = THREE.PCFSoftShadowMap;
		}
		this.renderer.domElement.style.display = 'block';
		this.renderer.domElement.style.width = '100%';
		this.renderer.domElement.style.height = '100%';
		this.container.appendChild(this.renderer.domElement);
		this.buildEnvironmentMap();

		this.controls = new OrbitControls(this.camera, this.renderer.domElement);
		this.controls.enableDamping = true;
		this.controls.dampingFactor = 0.09;
		this.controls.minDistance = 4.2;
		this.controls.maxDistance = 42;
		this.controls.addEventListener('start', this.handleControlsStart);
		this.controls.addEventListener('end', this.handleControlsEnd);

		this.ambientLight = new THREE.HemisphereLight('#7effdd', '#071511', this.lightingSettings.ambientIntensity);
		this.keyLight = new THREE.DirectionalLight('#fff6cf', this.lightingSettings.keyIntensity);
		this.keyLight.position.set(5, 7, 6);
		this.keyLight.castShadow = true;
		this.keyLight.shadow.mapSize.set(4096, 4096);
		this.keyLight.shadow.bias = -0.00003;
		this.keyLight.shadow.normalBias = 0.01;
		this.keyLight.shadow.radius = 2;
		this.scene.add(this.keyLight.target);
		this.fillLight = new THREE.DirectionalLight('#7fbfff', this.lightingSettings.fillIntensity);
		this.fillLight.position.set(-5.5, -1.4, 4.8);
		this.rimLight = new THREE.PointLight('#ff9f4a', this.lightingSettings.rimIntensity * 15, 20, 2);
		this.rimLight.position.set(-3.6, 1.4, -4.2);
		this.scene.add(this.ambientLight, this.keyLight, this.fillLight, this.rimLight);
		this.createGroundPlane();
		this.applyLightingSettings();

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
		this.animatedTextures = [];
		this.detachInteractionHandlers();
		this.resizeObserver?.disconnect();
		if (this.controls) {
			this.controls.removeEventListener('start', this.handleControlsStart);
			this.controls.removeEventListener('end', this.handleControlsEnd);
			this.controls.dispose();
			this.controls = null;
		}
		this.disposeGeometry();
		this.disposeGroundPlane();
		this.disposeTextureLibrary();
		if (this.environmentTarget) {
			this.environmentTarget.dispose();
			this.environmentTarget = null;
		}
		if (this.scene) this.scene.environment = null;
		this.ambientLight = null;
		this.keyLight = null;
		this.fillLight = null;
		this.rimLight = null;
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

	setMaterialSettings(next: Partial<KnotMaterialSettings>): void {
		this.materialSettings = {
			preset: next.preset ?? this.materialSettings.preset,
			roughness: clampNumber(next.roughness ?? this.materialSettings.roughness, 0.02, 1),
			metalness: clampNumber(next.metalness ?? this.materialSettings.metalness, 0, 1),
			transmission: clampNumber(next.transmission ?? this.materialSettings.transmission, 0, 1),
			clearcoat: clampNumber(next.clearcoat ?? this.materialSettings.clearcoat, 0, 1),
			envMapIntensity: clampNumber(next.envMapIntensity ?? this.materialSettings.envMapIntensity, 0, 3),
			emissiveIntensity: clampNumber(next.emissiveIntensity ?? this.materialSettings.emissiveIntensity, 0, 4),
			animationSpeed: clampNumber(next.animationSpeed ?? this.materialSettings.animationSpeed, 0, 6),
			textureScale: clampNumber(next.textureScale ?? this.materialSettings.textureScale, 0.2, 8)
		};
		this.restyleTubeMaterials();
	}

	setLightingSettings(next: Partial<KnotLightingSettings>): void {
		this.lightingSettings = {
			exposure: clampNumber(next.exposure ?? this.lightingSettings.exposure, 0.3, 2.7),
			ambientIntensity: clampNumber(
				next.ambientIntensity ?? this.lightingSettings.ambientIntensity,
				0,
				2.4
			),
			keyIntensity: clampNumber(next.keyIntensity ?? this.lightingSettings.keyIntensity, 0, 4),
			fillIntensity: clampNumber(next.fillIntensity ?? this.lightingSettings.fillIntensity, 0, 3),
			rimIntensity: clampNumber(next.rimIntensity ?? this.lightingSettings.rimIntensity, 0, 3)
		};
		this.applyLightingSettings();
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

	setLengthTargetScale(value: number): void {
		this.lengthTargetScale = clampNumber(value, 0.6, 1.1);
		this.applyLengthTargetScale();
		this.applySolverTuning();
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
		const allowRecentering = now >= this.releaseStabilizeUntil;

		if (this.autoRelax && this.dragIndex === null && !this.controlsInteracting) {
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
		if (
			this.dragIndex === null &&
			!this.controlsInteracting &&
			allowRecentering &&
			now - this.lastBoundsAt > 220
		) {
			this.lastBoundsAt = now;
			this.updateCameraForCurrentBounds();
		}
		this.updateAnimatedMaterialTextures(dt);
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
		const activeRestLength = this.state.componentRestLengths[0] ?? this.state.restLength;
		this.solver = new KnotSolver(this.points.length / 3, activeRestLength, this.currentSolverOptions());
		this.solver.setKeepCentered(false);
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
		this.applyLengthTargetScale();
		this.dragIndex = null;
		this.dragComponentIndex = null;
		this.controlsInteracting = false;
		this.releaseStabilizeUntil = 0;
		this.geometryDirty = true;
		this.rebuildGeometry(true);
		this.fitViewToState();
		this.emitMetrics();
	}

	private currentSolverOptions(): Partial<SolverOptions> {
		const tension = this.relaxSmoothness;
		const surfaceBarrierWeight = 0.6 + this.repulsionStrength * 1.5;
		const surfacePenetrationWeight = 160 + this.repulsionStrength * 240;
		const lengthTensionBoost = Math.max(0, 1 - this.lengthTargetScale) * 1200;
		const stretchWeight = 220 + tension * 430 + lengthTensionBoost;
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
		const lengthTensionBoost = Math.max(0, 1 - this.lengthTargetScale) * 1200;
		const stretchWeight = 220 + tension * 430 + lengthTensionBoost;
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

	private applyLengthTargetScale(): void {
		if (this.solver) this.solver.setRestLengthScale(this.lengthTargetScale);
		for (const solver of this.passiveSolvers) {
			solver.setRestLengthScale(this.lengthTargetScale);
		}
	}

	private stepAllComponents(
		iterations: number,
		dt: number,
		dragSelection: DragSelection | null
	): void {
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
			this.forEachDynamicComponent((componentIndex, points) => {
				const pinnedIndex =
					dragSelection && dragSelection.componentIndex === componentIndex ? dragSelection.pointIndex : null;
				const targetLength = this.getComponentConstraintLength(componentIndex);
				projectComponentEdgeLengths(points, targetLength, dragSelection ? 1 : 2, pinnedIndex);
			});
			if (dragSelection === null && this.relaxTick % 3 === 0) {
				this.reparameterizeAllComponents(null);
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
			const tubeMaterial = this.createTubeMaterial(0);
			this.tubeMesh = new THREE.Mesh(tubeGeometry, tubeMaterial);
			this.tubeMesh.castShadow = true;
			this.tubeMesh.receiveShadow = true;
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
		this.rebuildAnimatedTextureList();
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
				const tubeMaterial = this.createTubeMaterial(i + 1);
				const tubeMesh = new THREE.Mesh(createTubeGeometry(points, state.thickness), tubeMaterial);
				tubeMesh.castShadow = true;
				tubeMesh.receiveShadow = true;
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
			this.rebuildAnimatedTextureList();
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
		this.rebuildAnimatedTextureList();
	}

	private clearPassiveGeometry(): void {
		for (const mesh of this.passiveTubeMeshes) {
			this.scene?.remove(mesh);
			mesh.geometry.dispose();
			this.disposeTubeMaterial(mesh.material as THREE.MeshPhysicalMaterial);
		}
		for (const mesh of this.passiveLineMeshes) {
			this.scene?.remove(mesh);
			mesh.geometry.dispose();
			(mesh.material as THREE.Material).dispose();
		}
		this.passiveTubeMeshes = [];
		this.passiveLineMeshes = [];
		this.rebuildAnimatedTextureList();
	}

	private applyComponentColors(): void {
		if (this.tubeMesh) {
			const material = this.tubeMesh.material as THREE.MeshPhysicalMaterial;
			material.color.set(this.resolveTubeColor(0));
		}
		if (this.lineMesh) {
			const material = this.lineMesh.material as THREE.LineBasicMaterial;
			material.color.set(this.resolveLineColor(0));
		}
		for (let i = 0; i < this.passiveTubeMeshes.length; i += 1) {
			const material = this.passiveTubeMeshes[i].material as THREE.MeshPhysicalMaterial;
			material.color.set(this.resolveTubeColor(i + 1));
		}
		for (let i = 0; i < this.passiveLineMeshes.length; i += 1) {
			const material = this.passiveLineMeshes[i].material as THREE.LineBasicMaterial;
			material.color.set(this.resolveLineColor(i + 1));
		}
	}

	private resolveTubeColor(componentIndex: number): string {
		if (this.colorizeLinks) return componentTubeColor(true, componentIndex);
		return presetTubeColor(this.materialSettings.preset);
	}

	private resolveLineColor(componentIndex: number): string {
		if (this.colorizeLinks) return componentLineColor(true, componentIndex);
		return presetLineColor(this.materialSettings.preset);
	}

	private restyleTubeMaterials(): void {
		if (this.tubeMesh) {
			this.configureTubeMaterial(this.tubeMesh.material as THREE.MeshPhysicalMaterial, 0);
		}
		for (let i = 0; i < this.passiveTubeMeshes.length; i += 1) {
			const material = this.passiveTubeMeshes[i].material as THREE.MeshPhysicalMaterial;
			this.configureTubeMaterial(material, i + 1);
		}
		this.rebuildAnimatedTextureList();
		this.applyComponentColors();
	}

	private createTubeMaterial(componentIndex: number): THREE.MeshPhysicalMaterial {
		const material = new THREE.MeshPhysicalMaterial();
		this.configureTubeMaterial(material, componentIndex);
		return material;
	}

	private configureTubeMaterial(material: THREE.MeshPhysicalMaterial, componentIndex: number): void {
		this.releaseOwnedMaterialTextures(material);
		const settings = this.materialSettings;
		const textures = this.getMaterialTextureLibrary();
		const textureScale = settings.textureScale;
		const flowSpeed = settings.animationSpeed;
		const ownedTextures: THREE.Texture[] = [];
		const animated: AnimatedTextureState[] = [];

		material.map = null;
		material.normalMap = null;
		material.normalScale.set(1, 1);
		material.emissiveMap = null;
		material.emissive.set('#000000');
		material.attenuationColor.set('#ffffff');
		material.attenuationDistance = Infinity;
		material.reflectivity = 0.5;
		material.specularIntensity = 1;
		material.side = THREE.FrontSide;
		material.depthWrite = true;
		material.depthTest = true;
		material.roughness = settings.roughness;
		material.metalness = settings.metalness;
		material.transmission = settings.transmission;
		material.thickness = 0.55;
		material.ior = 1.42;
		material.clearcoat = settings.clearcoat;
		material.clearcoatRoughness = clampNumber(settings.roughness * 0.92, 0.02, 1);
		material.envMapIntensity = settings.envMapIntensity;
		material.emissiveIntensity = settings.emissiveIntensity * 0.08;
		material.color.set(this.resolveTubeColor(componentIndex));
		material.opacity = 1;
		material.transparent = false;

		switch (settings.preset) {
			case 'rope': {
				material.roughness = clampNumber(settings.roughness * 1.12, 0.03, 1);
				material.metalness = clampNumber(settings.metalness * 0.2, 0, 0.35);
				material.clearcoat = clampNumber(settings.clearcoat * 0.55, 0, 0.8);
				material.envMapIntensity = settings.envMapIntensity * 0.7;
				break;
			}
			case 'glass': {
				material.roughness = clampNumber(settings.roughness * 0.22, 0.01, 0.2);
				material.metalness = 0;
				material.transmission = Math.max(settings.transmission, 0.94);
				material.thickness = 0.42;
				material.ior = 1.5;
				material.clearcoat = Math.max(settings.clearcoat, 0.96);
				material.clearcoatRoughness = clampNumber(material.roughness * 0.3, 0.01, 0.12);
				material.envMapIntensity = Math.max(settings.envMapIntensity, 1.12);
				material.attenuationDistance = 2.3;
				material.attenuationColor.set(this.resolveTubeColor(componentIndex));
				material.reflectivity = 0.92;
				material.transparent = true;
				material.opacity = 0.3;
				material.side = THREE.DoubleSide;
				material.depthWrite = false;
				const frosted = this.cloneTextureForMaterial(
					textures.frostedNormal,
					textureScale * 1.6,
					textureScale * 1.6,
					componentIndex * 0.12
				);
				material.normalMap = frosted;
				material.normalScale.set(0.04, 0.04);
				ownedTextures.push(frosted);
				break;
			}
			case 'liquid_metal': {
				material.roughness = clampNumber(settings.roughness * 0.55, 0.12, 0.32);
				material.metalness = Math.max(settings.metalness, 0.95);
				material.clearcoat = Math.max(settings.clearcoat, 0.8);
				material.clearcoatRoughness = clampNumber(settings.roughness * 0.28, 0.08, 0.22);
				material.envMapIntensity = clampNumber(Math.max(settings.envMapIntensity, 1.05), 0.3, 1.4);
				material.map = null;
				break;
			}
			case 'energy_field': {
				material.roughness = clampNumber(settings.roughness * 0.54, 0.02, 0.6);
				material.metalness = clampNumber(settings.metalness * 0.4, 0, 0.55);
				material.transmission = Math.max(settings.transmission * 0.5, 0.28);
				material.clearcoat = Math.max(settings.clearcoat, 0.68);
				material.envMapIntensity = settings.envMapIntensity * 0.86;
				material.emissive.set(this.colorizeLinks ? componentLineColor(true, componentIndex) : '#4fc7ff');
				material.emissiveIntensity = settings.emissiveIntensity * 1.85;
				material.transparent = true;
				material.depthWrite = false;
				const energy = this.cloneTextureForMaterial(
					textures.energy,
					textureScale * 2.8,
					textureScale * 1.15,
					componentIndex * 0.17
				);
				material.emissiveMap = energy;
				ownedTextures.push(energy);
				animated.push({
					texture: energy,
					speed: 0.08 + flowSpeed * 0.28 + componentIndex * 0.04,
					axis: 'x'
				});
				break;
			}
			case 'vector_field': {
				material.roughness = clampNumber(settings.roughness * 0.64, 0.03, 0.72);
				material.metalness = clampNumber(settings.metalness * 0.42, 0, 0.62);
				material.clearcoat = Math.max(settings.clearcoat, 0.64);
				material.envMapIntensity = settings.envMapIntensity * 0.82;
				material.emissive.set(this.colorizeLinks ? componentLineColor(true, componentIndex) : '#68ffe0');
				material.emissiveIntensity = settings.emissiveIntensity * 1.34;
				material.transparent = material.transmission > 0.02;
				if (material.transparent) material.depthWrite = false;
				const vectors = this.cloneTextureForMaterial(
					textures.vectorField,
					textureScale * 2.5,
					textureScale * 1.4,
					componentIndex * 0.21,
					componentIndex * 0.03
				);
				material.emissiveMap = vectors;
				ownedTextures.push(vectors);
				animated.push({
					texture: vectors,
					speed: 0.05 + flowSpeed * 0.22 + componentIndex * 0.03,
					axis: 'y'
				});
				break;
			}
		}

		material.userData.knottyOwnedTextures = ownedTextures;
		material.userData.knottyAnimatedTextures = animated;
		material.needsUpdate = true;
	}

	private cloneTextureForMaterial(
		base: THREE.Texture,
		repeatX: number,
		repeatY: number,
		offsetX = 0,
		offsetY = 0
	): THREE.Texture {
		const texture = base.clone();
		texture.wrapS = THREE.MirroredRepeatWrapping;
		texture.wrapT = THREE.MirroredRepeatWrapping;
		texture.repeat.set(Math.max(0.02, repeatX), Math.max(0.02, repeatY));
		texture.offset.set(modulo1(offsetX), modulo1(offsetY));
		texture.needsUpdate = true;
		return texture;
	}

	private releaseOwnedMaterialTextures(material: THREE.MeshPhysicalMaterial): void {
		const owned = material.userData.knottyOwnedTextures as THREE.Texture[] | undefined;
		if (owned) {
			for (const texture of owned) texture.dispose();
		}
		material.userData.knottyOwnedTextures = [];
		material.userData.knottyAnimatedTextures = [];
	}

	private disposeTubeMaterial(material: THREE.MeshPhysicalMaterial): void {
		this.releaseOwnedMaterialTextures(material);
		material.dispose();
	}

	private rebuildAnimatedTextureList(): void {
		const animated: AnimatedTextureState[] = [];
		const appendFrom = (mesh: THREE.Mesh | null): void => {
			if (!mesh) return;
			const material = mesh.material as THREE.MeshPhysicalMaterial;
			const entries = material.userData.knottyAnimatedTextures as AnimatedTextureState[] | undefined;
			if (!entries || entries.length === 0) return;
			for (const entry of entries) animated.push(entry);
		};
		appendFrom(this.tubeMesh);
		for (const mesh of this.passiveTubeMeshes) appendFrom(mesh);
		this.animatedTextures = animated;
	}

	private updateAnimatedMaterialTextures(dt: number): void {
		if (this.animatedTextures.length === 0) return;
		for (const entry of this.animatedTextures) {
			const speed = entry.speed * dt;
			if (entry.axis === 'x') {
				entry.texture.offset.x += speed;
				if (entry.texture.offset.x > 1024) entry.texture.offset.x -= 1024;
			} else {
				entry.texture.offset.y += speed;
				if (entry.texture.offset.y > 1024) entry.texture.offset.y -= 1024;
			}
		}
	}

	private applyLightingSettings(): void {
		if (this.renderer) this.renderer.toneMappingExposure = this.lightingSettings.exposure;
		if (this.ambientLight) this.ambientLight.intensity = this.lightingSettings.ambientIntensity;
		if (this.keyLight) this.keyLight.intensity = this.lightingSettings.keyIntensity;
		if (this.fillLight) this.fillLight.intensity = this.lightingSettings.fillIntensity;
		if (this.rimLight) this.rimLight.intensity = this.lightingSettings.rimIntensity * 10;
	}

	private createGroundPlane(): void {
		if (!this.scene || this.groundPlane) return;
		const geometry = new THREE.CircleGeometry(1, 160);
		const patternTexture = createPoincareDiskTexture(4096);
		patternTexture.wrapS = THREE.ClampToEdgeWrapping;
		patternTexture.wrapT = THREE.ClampToEdgeWrapping;
		patternTexture.repeat.set(1, 1);
		const maxAnisotropy =
			(this.renderer as unknown as { capabilities?: { getMaxAnisotropy?: () => number } } | null)?.capabilities?.getMaxAnisotropy?.() ??
			16;
		patternTexture.anisotropy = Math.max(8, Math.min(24, maxAnisotropy));
		patternTexture.minFilter = THREE.LinearMipmapLinearFilter;
		patternTexture.magFilter = THREE.LinearFilter;
		patternTexture.generateMipmaps = true;
		patternTexture.needsUpdate = true;
		this.groundPatternTexture = patternTexture;
		const material = new THREE.MeshPhysicalMaterial({
			color: '#ffffff',
			map: patternTexture,
			roughness: 0.44,
			metalness: 0.15,
			clearcoat: 0.92,
			clearcoatRoughness: 0.18,
			envMapIntensity: 1.1,
			emissive: new THREE.Color('#000000'),
			emissiveIntensity: 0
		});
		this.groundPlane = new THREE.Mesh(geometry, material);
		this.groundPlane.rotation.x = -Math.PI / 2;
		this.groundPlane.receiveShadow = true;
		this.groundPlane.castShadow = false;
		this.groundPlane.frustumCulled = false;
		this.scene.add(this.groundPlane);
	}

	private disposeGroundPlane(): void {
		if (!this.groundPlane) return;
		this.scene?.remove(this.groundPlane);
		this.groundPlane.geometry.dispose();
		this.groundPatternTexture?.dispose();
		this.groundPatternTexture = null;
		this.groundPlane.material.dispose();
		this.groundPlane = null;
	}

	private updateGroundPlane(center: THREE.Vector3, radius: number): void {
		if (!this.groundPlane) return;
		const safeRadius = Math.max(0.2, radius);
		const size = Math.max(14, safeRadius * 11);
		const drop = Math.max(1.4, safeRadius * 1.2);
		this.groundPlane.scale.set(size, size, 1);
		this.groundPlane.position.set(center.x, center.y - drop, center.z);
		if (this.keyLight) {
			this.keyLight.target.position.set(center.x, center.y - safeRadius * 0.22, center.z);
			this.keyLight.target.updateMatrixWorld();
			const shadowCamera = this.keyLight.shadow.camera as THREE.OrthographicCamera;
			const span = Math.max(6, safeRadius * 8.5);
			shadowCamera.left = -span;
			shadowCamera.right = span;
			shadowCamera.top = span;
			shadowCamera.bottom = -span;
			shadowCamera.near = 0.1;
			shadowCamera.far = Math.max(28, safeRadius * 36);
			shadowCamera.updateProjectionMatrix();
		}
	}

	private buildEnvironmentMap(): void {
		if (!this.renderer || !this.scene) return;
		if (this.environmentTarget) {
			this.environmentTarget.dispose();
			this.environmentTarget = null;
		}
		if (this.renderer instanceof CoreWebGLRenderer) {
			this.scene.environment = null;
			return;
		}
		const renderer = this.renderer as unknown as never;
		const pmremGenerator = new THREE.PMREMGenerator(renderer);
		const environmentScene = new THREE.Scene();
		const roomEnvironment = new RoomEnvironment();
		environmentScene.add(roomEnvironment);
		const reflectionPattern = createPoincareDiskTexture(1536);
		reflectionPattern.wrapS = THREE.ClampToEdgeWrapping;
		reflectionPattern.wrapT = THREE.ClampToEdgeWrapping;
		reflectionPattern.repeat.set(1, 1);
		const reflectionFloor = new THREE.Mesh(
			new THREE.CircleGeometry(8, 128),
			new THREE.MeshStandardMaterial({
				color: '#ffffff',
				map: reflectionPattern,
				roughness: 0.42,
				metalness: 0.08
			})
		);
		reflectionFloor.rotation.x = -Math.PI / 2;
		reflectionFloor.position.y = -1.9;
		environmentScene.add(reflectionFloor);
		this.environmentTarget = pmremGenerator.fromScene(environmentScene, 0.32);
		this.scene.environment = this.environmentTarget.texture;
		pmremGenerator.dispose();
		reflectionFloor.geometry.dispose();
		(reflectionFloor.material as THREE.MeshStandardMaterial).dispose();
		reflectionPattern.dispose();
		(roomEnvironment as unknown as { dispose?: () => void }).dispose?.();
	}

	private getMaterialTextureLibrary(): MaterialTextureLibrary {
		if (this.materialTextures) return this.materialTextures;
		this.materialTextures = {
			brushedMetal: createBrushedMetalTexture(),
			energy: createEnergyTexture(),
			vectorField: createVectorFieldTexture(),
			frostedNormal: createFrostedNormalTexture()
		};
		return this.materialTextures;
	}

	private disposeTextureLibrary(): void {
		if (!this.materialTextures) return;
		this.materialTextures.brushedMetal.dispose();
		this.materialTextures.energy.dispose();
		this.materialTextures.vectorField.dispose();
		this.materialTextures.frostedNormal.dispose();
		this.materialTextures = null;
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

	private getComponentConstraintLength(componentIndex: number): number {
		const solver = this.getComponentSolver(componentIndex);
		if (solver) {
			const current = solver.getCurrentRestLength();
			if (Number.isFinite(current) && current > 0) return current;
		}
		return this.getComponentRestLength(componentIndex);
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

	private handleControlsStart = (): void => {
		this.controlsInteracting = true;
	};

	private handleControlsEnd = (): void => {
		this.controlsInteracting = false;
		this.lastBoundsAt = performance.now();
	};

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
		if (this.raycaster.ray.intersectPlane(this.dragPlane, this.dragPlaneHit)) {
			this.dragOffset.copy(this.dragTarget).sub(this.dragPlaneHit);
		} else {
			this.dragOffset.set(0, 0, 0);
		}
		this.controlsInteracting = false;
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
		if (!this.raycaster.ray.intersectPlane(this.dragPlane, this.dragPlaneHit)) return;
		this.dragTarget.copy(this.dragPlaneHit).add(this.dragOffset);
	};

	private handlePointerUp = (): void => {
		if (this.dragIndex === null || this.dragComponentIndex === null || !this.controls) return;
		this.dragIndex = null;
		this.dragComponentIndex = null;
		this.dragOffset.set(0, 0, 0);
		const now = performance.now();
		this.releaseStabilizeUntil = now + 320;
		this.lastBoundsAt = now;
		this.controlsInteracting = false;
		this.controls.enabled = true;
		this.notifyStatus('Released segment. Keep relaxing or continue dragging.');
	};

	private disposeGeometry(): void {
		if (this.tubeMesh) {
			this.scene?.remove(this.tubeMesh);
			this.tubeMesh.geometry.dispose();
			this.disposeTubeMaterial(this.tubeMesh.material as THREE.MeshPhysicalMaterial);
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
		this.animatedTextures = [];
	}

	private applyDragTarget(): void {
		if (this.dragIndex === null || this.dragComponentIndex === null || !this.state) return;
		const points = this.getComponentPoints(this.dragComponentIndex);
		const solver = this.getComponentSolver(this.dragComponentIndex);
		const restLength = this.getComponentConstraintLength(this.dragComponentIndex);
		if (!points || !solver || !Number.isFinite(restLength) || restLength <= 0) return;
		const count = points.length / 3;
		const idx = this.dragIndex * 3;
		const current = new THREE.Vector3(points[idx], points[idx + 1], points[idx + 2]);
		const delta = this.dragTarget.clone().sub(current);
		const maxMove = restLength * 0.2;
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

				const normal = separationNormalFromComponentEdgePair(
					pointsA,
					edgeA,
					edgeANext,
					pointsB,
					edgeB,
					edgeBNext,
					contact
				);
				const nx = normal[0];
				const ny = normal[1];
				const nz = normal[2];

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
				const restLength = this.getComponentConstraintLength(componentIndex);
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
			const restLength = this.getComponentConstraintLength(componentIndex);
			projectComponentEdgeLengths(points, restLength, 3, pinnedIndex);
		}
	}

	private recenterAllComponents(blend = 0.03): void {
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
		const clampedBlend = clampNumber(blend, 0, 1);
		if (clampedBlend <= 0) return;

		for (const points of components) {
			const pointCount = points.length / 3;
			for (let i = 0; i < pointCount; i += 1) {
				const base = i * 3;
				points[base] -= cx * clampedBlend;
				points[base + 1] -= cy * clampedBlend;
				points[base + 2] -= cz * clampedBlend;
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
			const restLength = this.getComponentConstraintLength(componentIndex);
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
		this.updateGroundPlane(center, radius);
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

function separationNormalFromComponentEdgePair(
	pointsA: Float32Array,
	a0: number,
	a1: number,
	pointsB: Float32Array,
	b0: number,
	b1: number,
	contact: ComponentSegmentContact
): [number, number, number] {
	let nx = contact.dx;
	let ny = contact.dy;
	let nz = contact.dz;
	let len = Math.hypot(nx, ny, nz);
	if (len > 1e-7) return [nx / len, ny / len, nz / len];

	const a0b = a0 * 3;
	const a1b = a1 * 3;
	const b0b = b0 * 3;
	const b1b = b1 * 3;
	const ax = pointsA[a1b] - pointsA[a0b];
	const ay = pointsA[a1b + 1] - pointsA[a0b + 1];
	const az = pointsA[a1b + 2] - pointsA[a0b + 2];
	const bx = pointsB[b1b] - pointsB[b0b];
	const by = pointsB[b1b + 1] - pointsB[b0b + 1];
	const bz = pointsB[b1b + 2] - pointsB[b0b + 2];

	nx = ay * bz - az * by;
	ny = az * bx - ax * bz;
	nz = ax * by - ay * bx;
	len = Math.hypot(nx, ny, nz);
	if (len > 1e-7) return [nx / len, ny / len, nz / len];

	const amx = (pointsA[a0b] + pointsA[a1b]) * 0.5;
	const amy = (pointsA[a0b + 1] + pointsA[a1b + 1]) * 0.5;
	const amz = (pointsA[a0b + 2] + pointsA[a1b + 2]) * 0.5;
	const bmx = (pointsB[b0b] + pointsB[b1b]) * 0.5;
	const bmy = (pointsB[b0b + 1] + pointsB[b1b + 1]) * 0.5;
	const bmz = (pointsB[b0b + 2] + pointsB[b1b + 2]) * 0.5;
	nx = amx - bmx;
	ny = amy - bmy;
	nz = amz - bmz;
	len = Math.hypot(nx, ny, nz);
	if (len > 1e-7) return [nx / len, ny / len, nz / len];

	return orthogonalDirection(ax, ay, az);
}

function orthogonalDirection(x: number, y: number, z: number): [number, number, number] {
	let ox: number;
	let oy: number;
	let oz: number;
	if (Math.abs(x) <= Math.abs(y) && Math.abs(x) <= Math.abs(z)) {
		ox = 0;
		oy = -z;
		oz = y;
	} else if (Math.abs(y) <= Math.abs(x) && Math.abs(y) <= Math.abs(z)) {
		ox = -z;
		oy = 0;
		oz = x;
	} else {
		ox = -y;
		oy = x;
		oz = 0;
	}
	let len = Math.hypot(ox, oy, oz);
	if (len < 1e-8) {
		ox = 1;
		oy = 0;
		oz = 0;
		len = 1;
	}
	return [ox / len, oy / len, oz / len];
}

function dot3(ax: number, ay: number, az: number, bx: number, by: number, bz: number): number {
	return ax * bx + ay * by + az * bz;
}

function presetTubeColor(preset: KnotMaterialPreset): string {
	switch (preset) {
		case 'glass':
			return '#e8f6ff';
		case 'liquid_metal':
			return '#c8d2e2';
		case 'energy_field':
			return '#1f3367';
		case 'vector_field':
			return '#1f3b35';
		default:
			return TUBE_COLOR;
	}
}

function presetLineColor(preset: KnotMaterialPreset): string {
	switch (preset) {
		case 'glass':
			return '#d4f4ff';
		case 'liquid_metal':
			return '#d1e3ff';
		case 'energy_field':
			return '#79b6ff';
		case 'vector_field':
			return '#8df0d5';
		default:
			return LINE_COLOR;
	}
}

function componentTubeColor(colorizeLinks: boolean, componentIndex: number): string {
	if (!colorizeLinks) return TUBE_COLOR;
	return COMPONENT_TUBE_COLORS[componentIndex % COMPONENT_TUBE_COLORS.length];
}

function componentLineColor(colorizeLinks: boolean, componentIndex: number): string {
	if (!colorizeLinks) return LINE_COLOR;
	return COMPONENT_LINE_COLORS[componentIndex % COMPONENT_LINE_COLORS.length];
}

function modulo1(value: number): number {
	const wrapped = value % 1;
	return wrapped < 0 ? wrapped + 1 : wrapped;
}

function createCanvasTexture(
	size: number,
	draw: (context: CanvasRenderingContext2D, size: number) => void,
	colorSpace: THREE.ColorSpace = THREE.SRGBColorSpace
): THREE.CanvasTexture {
	const canvas = document.createElement('canvas');
	canvas.width = size;
	canvas.height = size;
	const context = canvas.getContext('2d');
	if (!context) throw new Error('Unable to allocate 2D canvas for texture.');
	draw(context, size);
	const texture = new THREE.CanvasTexture(canvas);
	texture.wrapS = THREE.RepeatWrapping;
	texture.wrapT = THREE.RepeatWrapping;
	texture.colorSpace = colorSpace;
	texture.needsUpdate = true;
	return texture;
}

function createPoincareDiskTexture(textureSize = 2048): THREE.CanvasTexture {
	return createCanvasTexture(textureSize, (context, size) => {
		const center = size * 0.5;
		const radius = size * 0.4992;
		const p = 7;
		const q = 3;
		const halfA = Math.PI / (2 * p);
		const b = Math.PI / q;
		const cosA = Math.cos(halfA);
		const sinA = Math.sin(halfA);
		const sinB = Math.sin(b);
		const cosB = Math.cos(b);
		const denominator = cosA * cosA - sinB * sinB;
		const safeDenominator = Math.max(1e-6, denominator);
		const circleCenterX = cosB / Math.sqrt(safeDenominator);
		const circleRadius = Math.sqrt(Math.max(1e-6, circleCenterX * circleCenterX - 1));
		const maxIterations = 72;
		const image = context.createImageData(size, size);
		const data = image.data;
		const dark = [8, 8, 8];
		const light = [244, 244, 244];

		const reflectAcrossLine = (x: number, y: number, angle: number): [number, number] => {
			const c = Math.cos(angle * 2);
			const s = Math.sin(angle * 2);
			return [c * x + s * y, s * x - c * y];
		};

		for (let y = 0; y < size; y += 1) {
			for (let x = 0; x < size; x += 1) {
				let nx = (x + 0.5 - center) / radius;
				let ny = (y + 0.5 - center) / radius;
				const r = Math.hypot(nx, ny);
				const i = (y * size + x) * 4;
				if (r >= 1) {
					const scale = 0.999999 / Math.max(1e-6, r);
					nx *= scale;
					ny *= scale;
				}

				let px = nx;
				let py = ny;
				let reflections = 0;

				for (let step = 0; step < maxIterations; step += 1) {
					const lower = sinA * px + cosA * py;
					const upper = -sinA * px + cosA * py;
					const dx = px - circleCenterX;
					const dy = py;
					const insideCircle = dx * dx + dy * dy < circleRadius * circleRadius;

					if (lower >= 0 && upper <= 0 && !insideCircle) break;

					if (lower < 0) {
						[px, py] = reflectAcrossLine(px, py, -halfA);
						reflections += 1;
						continue;
					}
					if (upper > 0) {
						[px, py] = reflectAcrossLine(px, py, halfA);
						reflections += 1;
						continue;
					}
					if (insideCircle) {
						const lenSq = Math.max(1e-8, dx * dx + dy * dy);
						const scale = (circleRadius * circleRadius) / lenSq;
						px = circleCenterX + dx * scale;
						py = dy * scale;
						reflections += 1;
						continue;
					}
				}

				const shade = reflections % 2 === 0 ? light : dark;
				data[i] = shade[0];
				data[i + 1] = shade[1];
				data[i + 2] = shade[2];
				data[i + 3] = 255;
			}
		}

		context.putImageData(image, 0, 0);
	});
}

function createBrushedMetalTexture(): THREE.CanvasTexture {
	return createCanvasTexture(256, (context, size) => {
		const gradient = context.createLinearGradient(0, 0, 0, size);
		gradient.addColorStop(0, '#8893a6');
		gradient.addColorStop(0.45, '#d8dee8');
		gradient.addColorStop(1, '#7f8a9a');
		context.fillStyle = gradient;
		context.fillRect(0, 0, size, size);
		for (let y = 0; y < size; y += 1) {
			const jitter = 36 + Math.floor(Math.random() * 48);
			context.fillStyle = `rgba(255, 255, 255, ${jitter / 1000})`;
			context.fillRect(0, y, size, 1);
		}
		for (let i = 0; i < 1700; i += 1) {
			const x = Math.floor(Math.random() * size);
			const y = Math.floor(Math.random() * size);
			const alpha = 0.03 + Math.random() * 0.04;
			context.fillStyle = `rgba(0, 0, 0, ${alpha})`;
			context.fillRect(x, y, 1, 2);
		}
	});
}

function createEnergyTexture(): THREE.CanvasTexture {
	return createCanvasTexture(256, (context, size) => {
		const gradient = context.createLinearGradient(0, 0, size, 0);
		gradient.addColorStop(0, '#081a2e');
		gradient.addColorStop(0.5, '#114470');
		gradient.addColorStop(1, '#07121f');
		context.fillStyle = gradient;
		context.fillRect(0, 0, size, size);
		context.globalCompositeOperation = 'lighter';
		for (let band = 0; band < 14; band += 1) {
			const y = ((band + 0.5) * size) / 14;
			context.strokeStyle = `rgba(99, 205, 255, ${0.14 + (band % 3) * 0.06})`;
			context.lineWidth = 3 + (band % 2) * 2;
			context.beginPath();
			for (let x = 0; x <= size; x += 8) {
				const wave = Math.sin((x / size) * Math.PI * 4 + band * 0.73) * (4 + (band % 4));
				if (x === 0) context.moveTo(x, y + wave);
				else context.lineTo(x, y + wave);
			}
			context.stroke();
		}
		context.globalCompositeOperation = 'source-over';
	});
}

function createVectorFieldTexture(): THREE.CanvasTexture {
	return createCanvasTexture(256, (context, size) => {
		context.fillStyle = '#102926';
		context.fillRect(0, 0, size, size);
		context.strokeStyle = 'rgba(132, 255, 222, 0.55)';
		context.fillStyle = 'rgba(176, 255, 232, 0.78)';
		context.lineWidth = 2;
		const spacing = 32;
		for (let y = spacing / 2; y < size; y += spacing) {
			for (let x = spacing / 2; x < size; x += spacing) {
				const angle = ((x + y) / size) * Math.PI * 1.35;
				const len = 12;
				const x2 = x + Math.cos(angle) * len;
				const y2 = y + Math.sin(angle) * len;
				context.beginPath();
				context.moveTo(x, y);
				context.lineTo(x2, y2);
				context.stroke();

				const head = 4;
				const left = angle + Math.PI * 0.82;
				const right = angle - Math.PI * 0.82;
				context.beginPath();
				context.moveTo(x2, y2);
				context.lineTo(x2 + Math.cos(left) * head, y2 + Math.sin(left) * head);
				context.lineTo(x2 + Math.cos(right) * head, y2 + Math.sin(right) * head);
				context.closePath();
				context.fill();
			}
		}
	});
}

function createFrostedNormalTexture(): THREE.CanvasTexture {
	return createCanvasTexture(
		256,
		(context, size) => {
			const image = context.createImageData(size, size);
			for (let i = 0; i < image.data.length; i += 4) {
				const nx = Math.floor(128 + (Math.random() - 0.5) * 44);
				const ny = Math.floor(128 + (Math.random() - 0.5) * 44);
				image.data[i] = nx;
				image.data[i + 1] = ny;
				image.data[i + 2] = 255;
				image.data[i + 3] = 255;
			}
			context.putImageData(image, 0, 0);
		},
		THREE.NoColorSpace
	);
}

function clampInteger(value: number, min: number, max: number): number {
	return Math.min(max, Math.max(min, Math.round(value)));
}

function clampNumber(value: number, min: number, max: number): number {
	return Math.min(max, Math.max(min, value));
}

export type { KnotPresetName };
