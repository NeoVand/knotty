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
	scrambleClosedCurve,
	type KnotPresetName,
	type KnotState
} from './presets';
import { KnotSolver } from './solver';

export interface KnotMetrics {
	preset: KnotPresetName;
	label: string;
	crossings: number;
	targetCrossings: number | null;
	energy: number;
	nodeCount: number;
}

export interface KnotEngineOptions {
	container: HTMLElement;
	onMetrics?: (metrics: KnotMetrics) => void;
	onStatus?: (message: string) => void;
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
	private solver: KnotSolver | null = null;
	private tubeMesh: THREE.Mesh | null = null;
	private controlMesh: THREE.InstancedMesh | null = null;
	private autoRelax = true;
	private showControlPoints = false;
	private dragIndex: number | null = null;
	private geometryDirty = true;
	private disposed = false;
	private running = false;
	private lastFrameAt = 0;
	private lastMetricAt = 0;
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
		this.controls.maxDistance = 12;

		const hemi = new THREE.HemisphereLight('#7effdd', '#071511', 0.85);
		const key = new THREE.DirectionalLight('#fff6cf', 1.35);
		key.position.set(5, 7, 6);
		const rim = new THREE.PointLight('#ff9f4a', 15, 20, 2);
		rim.position.set(-3.6, 1.4, -4.2);
		this.scene.add(hemi, key, rim);

		this.attachInteractionHandlers();
		this.createState('trefoil', 0.62);
		this.setShowControlPoints(false);
		this.running = true;
		this.lastFrameAt = performance.now();
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

	setPreset(name: KnotPresetName, scramble = 0.62): void {
		this.createState(name, scramble);
	}

	setAutoRelax(value: boolean): void {
		this.autoRelax = value;
	}

	setShowControlPoints(value: boolean): void {
		this.showControlPoints = value;
		if (this.controlMesh) this.controlMesh.visible = value;
	}

	scramble(amount: number): void {
		if (!this.points || !this.solver || !this.state) return;
		scrambleClosedCurve(this.points, amount);
		this.solver.reset();
		this.geometryDirty = true;
		this.notifyStatus(`Scrambled ${this.state.label}`);
	}

	stepRelax(iterations = 12): void {
		if (!this.points || !this.solver || !this.state) return;
		for (let i = 0; i < iterations; i += 1) {
			this.solver.step(this.points, 1 / 120, null, this.state.thickness);
		}
		this.geometryDirty = true;
	}

	showSolution(iterations = 220): void {
		if (!this.points || !this.solver || !this.state) return;
		for (let i = 0; i < iterations; i += 1) {
			this.solver.step(this.points, 1 / 100, null, this.state.thickness);
		}
		this.geometryDirty = true;
		this.notifyStatus('Relaxed toward a low-energy embedding.');
	}

	getAvailablePresets(): KnotPresetName[] {
		return PRESET_ORDER;
	}

	private frame = (now: number): void => {
		if (!this.running || this.disposed) return;
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
			solver.step(points, dt, null, state.thickness);
			this.geometryDirty = true;
		}

		if (this.dragIndex !== null) {
			solver.step(points, dt, this.dragIndex, state.thickness);
			this.geometryDirty = true;
		}

		if (this.geometryDirty) this.rebuildGeometry();
		controls.update();
		renderer.render(scene, camera);

		if (now - this.lastMetricAt > 220) {
			this.lastMetricAt = now;
			this.emitMetrics();
		}

		requestAnimationFrame(this.frame);
	};

	private createState(name: KnotPresetName, scramble: number): void {
		this.state = createKnotState(name, { scramble });
		this.points = copyPoints(this.state.points);
		this.solver = new KnotSolver(this.points.length / 3, this.state.restLength);
		this.dragIndex = null;
		this.geometryDirty = true;
		this.rebuildGeometry(true);
		this.emitMetrics();
		this.notifyStatus(`Loaded ${this.state.label}.`);
	}

	private rebuildGeometry(forceControlRebuild = false): void {
		const scene = this.scene;
		const state = this.state;
		const points = this.points;
		if (!scene || !state || !points) return;

		const tubeGeometry = createTubeGeometry(points, state.thickness);
		if (!this.tubeMesh) {
			const tubeMaterial = new THREE.MeshPhysicalMaterial({
				color: '#ffcf73',
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

	private emitMetrics(): void {
		if (!this.onMetrics || !this.points || !this.solver || !this.state) return;
		this.onMetrics({
			preset: this.state.name,
			label: this.state.label,
			crossings: estimateCrossings(this.points),
			targetCrossings: this.state.targetCrossings,
			energy: this.solver.measureEnergy(this.points, this.state.thickness),
			nodeCount: this.points.length / 3
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

		let picked: number | null = null;
		if (this.controlMesh && this.showControlPoints) {
			const pointHits = this.raycaster.intersectObject(this.controlMesh, false);
			const instanceId = pointHits[0]?.instanceId;
			if (typeof instanceId === 'number') picked = instanceId;
		}

		if (picked === null && this.tubeMesh) {
			const hits = this.raycaster.intersectObject(this.tubeMesh, false);
			const hit = hits[0];
			if (hit) picked = nearestPointIndex(this.points, hit.point);
		}

		if (picked === null) return;
		this.dragIndex = picked;
		if (this.solver) this.solver.zeroVelocity(picked);
		this.dragPoint.copy(getPoint(this.points, picked));
		this.camera.getWorldDirection(this.cameraDirection);
		this.dragPlane.setFromNormalAndCoplanarPoint(this.cameraDirection, this.dragPoint);
		this.controls.enabled = false;
		dom.setPointerCapture(event.pointerId);
		this.notifyStatus('Dragging a knot segment.');
	};

	private handlePointerMove = (event: PointerEvent): void => {
		if (this.dragIndex === null || !this.camera || !this.renderer || !this.points || !this.solver || !this.state)
			return;
		const rect = this.renderer.domElement.getBoundingClientRect();
		const x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
		const y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
		this.pointer.set(x, y);
		this.raycaster.setFromCamera(this.pointer, this.camera);
		if (!this.raycaster.ray.intersectPlane(this.dragPlane, this.dragPoint)) return;
		setPoint(this.points, this.dragIndex, this.dragPoint);
		this.solver.zeroVelocity(this.dragIndex);
		this.geometryDirty = true;
	};

	private handlePointerUp = (): void => {
		if (this.dragIndex === null || !this.controls) return;
		this.dragIndex = null;
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
		if (this.controlMesh) {
			this.scene?.remove(this.controlMesh);
			this.controlMesh.geometry.dispose();
			(this.controlMesh.material as THREE.Material).dispose();
			this.controlMesh = null;
		}
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
	const path = new THREE.CatmullRomCurve3(vectors, true, 'catmullrom', 0.45);
	const tubularSegments = Math.max(200, count * 2);
	return new THREE.TubeGeometry(path, tubularSegments, thickness, 18, true);
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

function setPoint(points: Float32Array, index: number, point: THREE.Vector3): void {
	const idx = index * 3;
	points[idx] = point.x;
	points[idx + 1] = point.y;
	points[idx + 2] = point.z;
}

export type { KnotPresetName };
