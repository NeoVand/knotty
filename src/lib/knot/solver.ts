export interface SolverOptions {
	alpha: number;
	beta: number;
	armijo: number;
	minStep: number;
	maxStep: number;
	maxLineSearchSteps: number;
	cgIterations: number;
	cgTolerance: number;
	regularization: number;
	highOrderWeight: number;
	lowOrderWeight: number;
	repulsionWeight: number;
	bendingWeight: number;
	smoothingWeight: number;
	stretchWeight: number;
	constraintPasses: number;
	edgeCollisionPasses: number;
	surfaceClearanceFactor: number;
	surfaceNearFieldFactor: number;
	surfaceBarrierWeight: number;
	surfacePenetrationWeight: number;
}

export interface SolverDiagnostics {
	lastStepAccepted: boolean;
	lastAcceptedStepSize: number;
	acceptedSteps: number;
	rejectedSteps: number;
	lastMinEdgeDistance: number;
	lastClearanceRatio: number;
	lastEnergyBefore: number;
	lastEnergyAfter: number;
}

const DEFAULT_OPTIONS: SolverOptions = {
	alpha: 3,
	beta: 6,
	armijo: 0.01,
	minStep: 0.0005,
	maxStep: 0.07,
	maxLineSearchSteps: 9,
	cgIterations: 54,
	cgTolerance: 1e-4,
	regularization: 2e-3,
	highOrderWeight: 0.38,
	lowOrderWeight: 1.0,
	repulsionWeight: 1.0,
	bendingWeight: 26,
	smoothingWeight: 0.1,
	stretchWeight: 220,
	constraintPasses: 8,
	edgeCollisionPasses: 2,
	surfaceClearanceFactor: 2.06,
	surfaceNearFieldFactor: 1.6,
	surfaceBarrierWeight: 1.5,
	surfacePenetrationWeight: 240
};

interface SegmentContact {
	s: number;
	t: number;
	distSq: number;
	dx: number;
	dy: number;
	dz: number;
}

interface EdgeHashData {
	buckets: Map<string, number[]>;
	ix: Int32Array;
	iy: Int32Array;
	iz: Int32Array;
}

export class KnotSolver {
	private readonly nodeCount: number;
	private readonly restLength: number;
	private options: SolverOptions;
	private lastStepSize = 0.018;
	private keepCentered = true;

	private readonly candidate: Float32Array;
	private readonly tangents: Float32Array;
	private readonly dualLengths: Float32Array;
	private readonly edgeLengths: Float32Array;
	private readonly l2Gradient: Float32Array;
	private readonly direction: Float32Array;
	private readonly pairWeights: Float32Array;
	private readonly rowSums: Float32Array;
	private readonly diag: Float32Array;

	private readonly cgX: Float32Array;
	private readonly cgR: Float32Array;
	private readonly cgP: Float32Array;
	private readonly cgAp: Float32Array;
	private readonly cgZ: Float32Array;
	private readonly cgOut: Float32Array;
	private readonly smoothBuffer: Float32Array;
	private readonly edgeCellX: Int32Array;
	private readonly edgeCellY: Int32Array;
	private readonly edgeCellZ: Int32Array;
	private readonly diagnostics: SolverDiagnostics = {
		lastStepAccepted: false,
		lastAcceptedStepSize: 0,
		acceptedSteps: 0,
		rejectedSteps: 0,
		lastMinEdgeDistance: Number.POSITIVE_INFINITY,
		lastClearanceRatio: Number.POSITIVE_INFINITY,
		lastEnergyBefore: 0,
		lastEnergyAfter: 0
	};

	constructor(nodeCount: number, restLength: number, options: Partial<SolverOptions> = {}) {
		this.nodeCount = nodeCount;
		this.restLength = restLength;
		this.options = { ...DEFAULT_OPTIONS, ...options };

		const vectorLength = nodeCount * 3;
		this.candidate = new Float32Array(vectorLength);
		this.tangents = new Float32Array(vectorLength);
		this.dualLengths = new Float32Array(nodeCount);
		this.edgeLengths = new Float32Array(nodeCount);
		this.l2Gradient = new Float32Array(vectorLength);
		this.direction = new Float32Array(vectorLength);
		this.pairWeights = new Float32Array(nodeCount * nodeCount);
		this.rowSums = new Float32Array(nodeCount);
		this.diag = new Float32Array(nodeCount);

		this.cgX = new Float32Array(nodeCount);
		this.cgR = new Float32Array(nodeCount);
		this.cgP = new Float32Array(nodeCount);
		this.cgAp = new Float32Array(nodeCount);
		this.cgZ = new Float32Array(nodeCount);
		this.cgOut = new Float32Array(nodeCount);
		this.smoothBuffer = new Float32Array(vectorLength);
		this.edgeCellX = new Int32Array(nodeCount);
		this.edgeCellY = new Int32Array(nodeCount);
		this.edgeCellZ = new Int32Array(nodeCount);
	}

	reset(): void {
		this.lastStepSize = 0.018;
		this.diagnostics.lastStepAccepted = false;
		this.diagnostics.lastAcceptedStepSize = 0;
		this.diagnostics.acceptedSteps = 0;
		this.diagnostics.rejectedSteps = 0;
		this.diagnostics.lastMinEdgeDistance = Number.POSITIVE_INFINITY;
		this.diagnostics.lastClearanceRatio = Number.POSITIVE_INFINITY;
		this.diagnostics.lastEnergyBefore = 0;
		this.diagnostics.lastEnergyAfter = 0;
	}

	setTuning(
		next: Partial<
			Pick<
				SolverOptions,
				| 'repulsionWeight'
				| 'smoothingWeight'
				| 'highOrderWeight'
				| 'bendingWeight'
				| 'stretchWeight'
				| 'surfaceBarrierWeight'
				| 'surfacePenetrationWeight'
				| 'surfaceNearFieldFactor'
				| 'surfaceClearanceFactor'
				| 'constraintPasses'
				| 'edgeCollisionPasses'
			>
		>
	): void {
		if (next.repulsionWeight !== undefined) {
			this.options.repulsionWeight = clamp(next.repulsionWeight, 0.1, 5);
		}
		if (next.smoothingWeight !== undefined) {
			this.options.smoothingWeight = clamp(next.smoothingWeight, 0, 0.35);
		}
		if (next.highOrderWeight !== undefined) {
			this.options.highOrderWeight = clamp(next.highOrderWeight, 0.08, 1.2);
		}
		if (next.bendingWeight !== undefined) {
			this.options.bendingWeight = clamp(next.bendingWeight, 0, 220);
		}
		if (next.stretchWeight !== undefined) {
			this.options.stretchWeight = clamp(next.stretchWeight, 40, 900);
		}
		if (next.surfaceBarrierWeight !== undefined) {
			this.options.surfaceBarrierWeight = clamp(next.surfaceBarrierWeight, 0, 8);
		}
		if (next.surfacePenetrationWeight !== undefined) {
			this.options.surfacePenetrationWeight = clamp(next.surfacePenetrationWeight, 20, 1200);
		}
		if (next.surfaceNearFieldFactor !== undefined) {
			this.options.surfaceNearFieldFactor = clamp(next.surfaceNearFieldFactor, 1.05, 3.2);
		}
		if (next.surfaceClearanceFactor !== undefined) {
			this.options.surfaceClearanceFactor = clamp(next.surfaceClearanceFactor, 1.8, 3.2);
		}
		if (next.constraintPasses !== undefined) {
			this.options.constraintPasses = clamp(Math.round(next.constraintPasses), 2, 28);
		}
		if (next.edgeCollisionPasses !== undefined) {
			this.options.edgeCollisionPasses = clamp(Math.round(next.edgeCollisionPasses), 1, 8);
		}
	}

	setKeepCentered(value: boolean): void {
		this.keepCentered = value;
	}

	getDiagnostics(): SolverDiagnostics {
		return { ...this.diagnostics };
	}

	zeroVelocity(_index: number): void {
		// No-op: this solver integrates a constrained descent direction directly.
	}

	enforceInextensibility(points: Float32Array, pinnedIndex: number | null, thickness: number, passes = 1): void {
		const totalPasses = Math.max(1, Math.round(passes));
		for (let i = 0; i < totalPasses; i += 1) {
			this.applyDistanceConstraints(points, pinnedIndex);
			this.applyEdgeCollisions(points, pinnedIndex, thickness);
			this.applyDistanceConstraints(points, pinnedIndex);
		}
	}

	step(points: Float32Array, dt: number, pinnedIndex: number | null, thickness: number): void {
		const baseEnergy = this.measureEnergy(points, thickness);
		this.diagnostics.lastEnergyBefore = baseEnergy;
		this.fillL2Gradient(points, thickness);
		zeroPinnedVector(this.l2Gradient, pinnedIndex);
		if (pinnedIndex === null) removeMean(this.l2Gradient);

		this.buildPreconditioner(points, thickness);
		this.solveSobolevDirection(this.l2Gradient, this.direction, pinnedIndex);
		zeroPinnedVector(this.direction, pinnedIndex);
		if (pinnedIndex === null) removeMean(this.direction);

		const directionNorm = Math.sqrt(dot(this.direction, this.direction));
		if (!Number.isFinite(directionNorm) || directionNorm < 1e-7) return;
		scale(this.direction, 1 / directionNorm);

		const gradDotDirection = dot(this.l2Gradient, this.direction);
		const stepScale = Math.sqrt(clamp(dt / (1 / 60), 0.2, 1.8));
		let stepSize = clamp(this.lastStepSize * 1.2 * stepScale, this.options.minStep, this.options.maxStep);
		let accepted = false;

		for (let attempt = 0; attempt < this.options.maxLineSearchSteps; attempt += 1) {
			for (let i = 0; i < points.length; i += 1) {
				this.candidate[i] = points[i] - stepSize * this.direction[i];
			}

			this.applyDistanceConstraints(this.candidate, pinnedIndex);
			this.applyEdgeCollisions(this.candidate, pinnedIndex, thickness);
			this.applyCurveSmoothing(this.candidate, pinnedIndex);
			this.applyDistanceConstraints(this.candidate, pinnedIndex);
			if (pinnedIndex === null && this.keepCentered) recenter(this.candidate, this.nodeCount);

			const nextEnergy = this.measureEnergy(this.candidate, thickness);
			const armijoTarget = baseEnergy - this.options.armijo * stepSize * Math.max(1e-8, gradDotDirection);

			if (nextEnergy < baseEnergy || nextEnergy <= armijoTarget) {
				points.set(this.candidate);
				this.lastStepSize = stepSize;
				this.diagnostics.lastEnergyAfter = nextEnergy;
				this.diagnostics.lastStepAccepted = true;
				this.diagnostics.lastAcceptedStepSize = stepSize;
				this.diagnostics.acceptedSteps += 1;
				accepted = true;
				break;
			}

			stepSize *= 0.5;
			if (stepSize < this.options.minStep) break;
		}

		if (!accepted) {
			this.lastStepSize = Math.max(this.options.minStep, this.lastStepSize * 0.6);
			this.diagnostics.lastEnergyAfter = baseEnergy;
			this.diagnostics.lastStepAccepted = false;
			this.diagnostics.rejectedSteps += 1;
		}

		const minEdgeDistance = this.measureMinEdgeDistance(points);
		const clearance = thickness * this.options.surfaceClearanceFactor;
		this.diagnostics.lastMinEdgeDistance = minEdgeDistance;
		this.diagnostics.lastClearanceRatio = minEdgeDistance / Math.max(1e-6, clearance);
	}

	measureEnergy(points: Float32Array, thickness: number): number {
		this.updateCurveFrames(points);
		const { alpha, beta, repulsionWeight } = this.options;
		let energy = 0;
		let stretchPenalty = 0;

		for (let i = 0; i < this.nodeCount; i += 1) {
			const lenDiff = this.edgeLengths[i] - this.restLength;
			stretchPenalty += lenDiff * lenDiff;
		}

		for (let i = 0; i < this.nodeCount; i += 1) {
			const pi = i * 3;
			const ti = pi;
			for (let j = i + 1; j < this.nodeCount; j += 1) {
				if (ringDistance(i, j, this.nodeCount) <= 1) continue;

				const pj = j * 3;
				const dx = points[pi] - points[pj];
				const dy = points[pi + 1] - points[pj + 1];
				const dz = points[pi + 2] - points[pj + 2];
				const distSq = dx * dx + dy * dy + dz * dz;
				const dist = Math.sqrt(distSq) + 1e-8;

				const dotI = dx * this.tangents[ti] + dy * this.tangents[ti + 1] + dz * this.tangents[ti + 2];
				const npxI = dx - dotI * this.tangents[ti];
				const npyI = dy - dotI * this.tangents[ti + 1];
				const npzI = dz - dotI * this.tangents[ti + 2];
				const normI = Math.sqrt(npxI * npxI + npyI * npyI + npzI * npzI) + 1e-8;

				const tj = pj;
				const dotJ = dx * this.tangents[tj] + dy * this.tangents[tj + 1] + dz * this.tangents[tj + 2];
				const npxJ = dx - dotJ * this.tangents[tj];
				const npyJ = dy - dotJ * this.tangents[tj + 1];
				const npzJ = dz - dotJ * this.tangents[tj + 2];
				const normJ = Math.sqrt(npxJ * npxJ + npyJ * npyJ + npzJ * npzJ) + 1e-8;

				const weight = this.dualLengths[i] * this.dualLengths[j];
				const invDistBeta = 1 / Math.pow(dist, beta);
				energy += repulsionWeight * weight * (Math.pow(normI, alpha) + Math.pow(normJ, alpha)) * invDistBeta;
			}
		}

		const surfaceBarrierPenalty = this.measureSurfaceBarrierEnergy(points, thickness);
		const bendingPenalty = this.measureBendingEnergy(points);
		return (
			energy +
			this.options.stretchWeight * stretchPenalty +
			this.options.bendingWeight * bendingPenalty +
			surfaceBarrierPenalty
		);
	}

	private fillL2Gradient(points: Float32Array, thickness: number): void {
		this.updateCurveFrames(points);
		this.l2Gradient.fill(0);
		const { alpha, beta, repulsionWeight } = this.options;

		for (let i = 0; i < this.nodeCount; i += 1) {
			const pi = i * 3;
			const ti = pi;
			for (let j = i + 1; j < this.nodeCount; j += 1) {
				if (ringDistance(i, j, this.nodeCount) <= 1) continue;
				const pj = j * 3;

				const dx = points[pi] - points[pj];
				const dy = points[pi + 1] - points[pj + 1];
				const dz = points[pi + 2] - points[pj + 2];
				const distSq = dx * dx + dy * dy + dz * dz + 1e-10;
				const dist = Math.sqrt(distSq);
				const invDistBeta = 1 / Math.pow(dist, beta);
				const invDistBetaPlus2 = invDistBeta / distSq;

				const gradIx = gradientKernelComponent(
					dx,
					dy,
					dz,
					this.tangents[ti],
					this.tangents[ti + 1],
					this.tangents[ti + 2],
					alpha,
					beta,
					invDistBeta,
					invDistBetaPlus2
				);
				const tj = pj;
				const gradJx = gradientKernelComponent(
					dx,
					dy,
					dz,
					this.tangents[tj],
					this.tangents[tj + 1],
					this.tangents[tj + 2],
					alpha,
					beta,
					invDistBeta,
					invDistBetaPlus2
				);

				const weight = this.dualLengths[i] * this.dualLengths[j];
				const gx = repulsionWeight * weight * (gradIx[0] + gradJx[0]);
				const gy = repulsionWeight * weight * (gradIx[1] + gradJx[1]);
				const gz = repulsionWeight * weight * (gradIx[2] + gradJx[2]);

				this.l2Gradient[pi] += gx;
				this.l2Gradient[pi + 1] += gy;
				this.l2Gradient[pi + 2] += gz;
				this.l2Gradient[pj] -= gx;
				this.l2Gradient[pj + 1] -= gy;
				this.l2Gradient[pj + 2] -= gz;
			}
		}

		this.addStretchGradient(points);
		this.addBendingGradient(points);
		this.addSurfaceBarrierGradient(points, thickness);
	}

	private buildPreconditioner(points: Float32Array, thickness: number): void {
		this.updateCurveFrames(points);
		const { alpha, beta, lowOrderWeight, regularization, highOrderWeight, repulsionWeight } = this.options;
		const sigma = (beta - 1) / alpha - 1;
		const n = this.nodeCount;
		const minDist = Math.max(thickness * 2.0, 1e-3);

		this.pairWeights.fill(0);
		this.rowSums.fill(0);

		for (let i = 0; i < n; i += 1) {
			const pi = i * 3;
			const ti = pi;
			for (let j = i + 1; j < n; j += 1) {
				if (ringDistance(i, j, n) <= 1) continue;
				const pj = j * 3;
				const tj = pj;

				const dx = points[pi] - points[pj];
				const dy = points[pi + 1] - points[pj + 1];
				const dz = points[pi + 2] - points[pj + 2];
				const distSq = dx * dx + dy * dy + dz * dz + 1e-10;
				const dist = Math.sqrt(distSq);
				const safeDist = Math.max(minDist, dist);

				const kI = kernel24(dx, dy, dz, this.tangents[ti], this.tangents[ti + 1], this.tangents[ti + 2], safeDist);
				const kJ = kernel24(dx, dy, dz, this.tangents[tj], this.tangents[tj + 1], this.tangents[tj + 2], safeDist);
				const kSym = 0.5 * (kI + kJ);
				const weight =
					repulsionWeight *
					lowOrderWeight *
					this.dualLengths[i] *
					this.dualLengths[j] *
					kSym /
					Math.pow(safeDist, 2 * sigma + 1);

				if (!Number.isFinite(weight) || weight <= 0) continue;
				const indexIJ = i * n + j;
				const indexJI = j * n + i;
				this.pairWeights[indexIJ] = weight;
				this.pairWeights[indexJI] = weight;
				this.rowSums[i] += weight;
				this.rowSums[j] += weight;
			}
		}

		for (let i = 0; i < n; i += 1) {
			this.diag[i] = regularization + 6 * highOrderWeight + this.rowSums[i];
		}
	}

	private solveSobolevDirection(
		rhs: Float32Array,
		output: Float32Array,
		pinnedIndex: number | null
	): void {
		solveAxis(this, rhs, output, 0, pinnedIndex);
		solveAxis(this, rhs, output, 1, pinnedIndex);
		solveAxis(this, rhs, output, 2, pinnedIndex);
	}

	private applyOperator(axisInput: Float32Array, axisOutput: Float32Array, pinnedIndex: number | null): void {
		const n = this.nodeCount;
		const { regularization, highOrderWeight } = this.options;

		for (let i = 0; i < n; i += 1) {
			if (i === pinnedIndex) {
				axisOutput[i] = axisInput[i];
				continue;
			}

			const im1 = (i - 1 + n) % n;
			const ip1 = (i + 1) % n;
			const im2 = (i - 2 + n) % n;
			const ip2 = (i + 2) % n;
			const biLap =
				6 * axisInput[i] -
				4 * (axisInput[im1] + axisInput[ip1]) +
				(axisInput[im2] + axisInput[ip2]);

			let lowOrder = this.rowSums[i] * axisInput[i];
			const rowOffset = i * n;
			for (let j = 0; j < n; j += 1) {
				if (j === i) continue;
				const w = this.pairWeights[rowOffset + j];
				if (w === 0) continue;
				lowOrder -= w * axisInput[j];
			}

			axisOutput[i] = regularization * axisInput[i] + highOrderWeight * biLap + lowOrder;
		}
	}

	private applyDistanceConstraints(points: Float32Array, pinnedIndex: number | null): void {
		for (let pass = 0; pass < this.options.constraintPasses; pass += 1) {
			for (let i = 0; i < this.nodeCount; i += 1) {
				const j = (i + 1) % this.nodeCount;
				const ai = i * 3;
				const bj = j * 3;
				const dx = points[bj] - points[ai];
				const dy = points[bj + 1] - points[ai + 1];
				const dz = points[bj + 2] - points[ai + 2];
				const dist = Math.hypot(dx, dy, dz) || 1e-6;
				const correction = (dist - this.restLength) / dist;
				let aWeight = 0.5;
				let bWeight = 0.5;
				if (i === pinnedIndex) {
					aWeight = 0;
					bWeight = 1;
				} else if (j === pinnedIndex) {
					aWeight = 1;
					bWeight = 0;
				}

				if (aWeight > 0) {
					points[ai] += dx * correction * aWeight;
					points[ai + 1] += dy * correction * aWeight;
					points[ai + 2] += dz * correction * aWeight;
				}
				if (bWeight > 0) {
					points[bj] -= dx * correction * bWeight;
					points[bj + 1] -= dy * correction * bWeight;
					points[bj + 2] -= dz * correction * bWeight;
				}
			}
		}
	}

	private applyEdgeCollisions(points: Float32Array, pinnedIndex: number | null, tubeRadius: number): void {
		const passes = this.options.edgeCollisionPasses;
		const clearance = tubeRadius * this.options.surfaceClearanceFactor;
		const nearField = clearance * this.options.surfaceNearFieldFactor;
		const softness = Math.max(1e-4, nearField - clearance);

		for (let pass = 0; pass < passes; pass += 1) {
			this.forEachPotentialEdgePair(points, nearField, (edgeA, edgeB) => {
				const edgeANext = (edgeA + 1) % this.nodeCount;
				const edgeBNext = (edgeB + 1) % this.nodeCount;
				const contact = closestPointsBetweenEdges(points, edgeA, edgeANext, edgeB, edgeBNext);
				const dist = Math.sqrt(contact.distSq);
				if (!Number.isFinite(dist) || dist >= nearField) return;

				let nx = contact.dx;
				let ny = contact.dy;
				let nz = contact.dz;
				let normalLength = Math.hypot(nx, ny, nz);
				if (normalLength < 1e-7) {
					nx = points[edgeANext * 3] - points[edgeA * 3];
					ny = points[edgeANext * 3 + 1] - points[edgeA * 3 + 1];
					nz = points[edgeANext * 3 + 2] - points[edgeA * 3 + 2];
					normalLength = Math.hypot(nx, ny, nz) || 1;
				}
				nx /= normalLength;
				ny /= normalLength;
				nz /= normalLength;

				let push = 0;
				if (dist < clearance) {
					push = (clearance - dist) * 0.52;
				} else {
					const blend = (nearField - dist) / softness;
					push = clearance * blend * blend * 0.025;
				}
				if (push <= 0) return;

				const a0 = 1 - contact.s;
				const a1 = contact.s;
				const b0 = 1 - contact.t;
				const b1 = contact.t;

				applyPointDisplacement(points, edgeA, nx * push * a0, ny * push * a0, nz * push * a0, pinnedIndex);
				applyPointDisplacement(points, edgeANext, nx * push * a1, ny * push * a1, nz * push * a1, pinnedIndex);
				applyPointDisplacement(points, edgeB, -nx * push * b0, -ny * push * b0, -nz * push * b0, pinnedIndex);
				applyPointDisplacement(
					points,
					edgeBNext,
					-nx * push * b1,
					-ny * push * b1,
					-nz * push * b1,
					pinnedIndex
				);
			});
		}
	}

	private measureSurfaceBarrierEnergy(points: Float32Array, tubeRadius: number): number {
		const clearance = tubeRadius * this.options.surfaceClearanceFactor;
		const nearField = clearance * this.options.surfaceNearFieldFactor;
		const softness = Math.max(1e-4, nearField - clearance);
		let energy = 0;

		this.forEachPotentialEdgePair(points, nearField, (edgeA, edgeB) => {
			const edgeANext = (edgeA + 1) % this.nodeCount;
			const edgeBNext = (edgeB + 1) % this.nodeCount;
			const contact = closestPointsBetweenEdges(points, edgeA, edgeANext, edgeB, edgeBNext);
			const dist = Math.sqrt(contact.distSq);
			if (!Number.isFinite(dist) || dist >= nearField) return;

			if (dist < clearance) {
				const penetration = clearance - dist;
				energy += this.options.surfacePenetrationWeight * penetration * penetration;
				return;
			}

			const blend = (nearField - dist) / softness;
			energy += this.options.surfaceBarrierWeight * blend * blend * blend;
		});

		return energy;
	}

	private addSurfaceBarrierGradient(points: Float32Array, tubeRadius: number): void {
		const clearance = tubeRadius * this.options.surfaceClearanceFactor;
		const nearField = clearance * this.options.surfaceNearFieldFactor;
		const softness = Math.max(1e-4, nearField - clearance);

		this.forEachPotentialEdgePair(points, nearField, (edgeA, edgeB) => {
			const edgeANext = (edgeA + 1) % this.nodeCount;
			const edgeBNext = (edgeB + 1) % this.nodeCount;
			const contact = closestPointsBetweenEdges(points, edgeA, edgeANext, edgeB, edgeBNext);
			const dist = Math.sqrt(contact.distSq);
			if (!Number.isFinite(dist) || dist >= nearField) return;

			let nx = contact.dx;
			let ny = contact.dy;
			let nz = contact.dz;
			let normalLength = Math.hypot(nx, ny, nz);
			if (normalLength < 1e-7) {
				nx = points[edgeANext * 3] - points[edgeA * 3];
				ny = points[edgeANext * 3 + 1] - points[edgeA * 3 + 1];
				nz = points[edgeANext * 3 + 2] - points[edgeA * 3 + 2];
				normalLength = Math.hypot(nx, ny, nz) || 1;
			}
			nx /= normalLength;
			ny /= normalLength;
			nz /= normalLength;

			let magnitude = 0;
			if (dist < clearance) {
				magnitude = this.options.surfacePenetrationWeight * 2 * (clearance - dist);
			} else {
				const blend = (nearField - dist) / softness;
				magnitude = (this.options.surfaceBarrierWeight * 3 * blend * blend) / softness;
			}
			if (!Number.isFinite(magnitude) || magnitude <= 0) return;

			const a0 = 1 - contact.s;
			const a1 = contact.s;
			const b0 = 1 - contact.t;
			const b1 = contact.t;

			applyGradientContribution(
				this.l2Gradient,
				edgeA,
				-nx * magnitude * a0,
				-ny * magnitude * a0,
				-nz * magnitude * a0
			);
			applyGradientContribution(
				this.l2Gradient,
				edgeANext,
				-nx * magnitude * a1,
				-ny * magnitude * a1,
				-nz * magnitude * a1
			);
			applyGradientContribution(
				this.l2Gradient,
				edgeB,
				nx * magnitude * b0,
				ny * magnitude * b0,
				nz * magnitude * b0
			);
			applyGradientContribution(
				this.l2Gradient,
				edgeBNext,
				nx * magnitude * b1,
				ny * magnitude * b1,
				nz * magnitude * b1
			);
		});
	}

	private measureMinEdgeDistance(points: Float32Array): number {
		let minDistance = Number.POSITIVE_INFINITY;
		for (let edgeA = 0; edgeA < this.nodeCount; edgeA += 1) {
			const edgeANext = (edgeA + 1) % this.nodeCount;
			for (let edgeB = edgeA + 1; edgeB < this.nodeCount; edgeB += 1) {
				if (areAdjacentEdges(edgeA, edgeB, this.nodeCount)) continue;
				const edgeBNext = (edgeB + 1) % this.nodeCount;
				const contact = closestPointsBetweenEdges(points, edgeA, edgeANext, edgeB, edgeBNext);
				const dist = Math.sqrt(contact.distSq);
				if (!Number.isFinite(dist)) continue;
				if (dist < minDistance) minDistance = dist;
			}
		}
		return minDistance;
	}

	private forEachPotentialEdgePair(points: Float32Array, searchRadius: number, visitor: (a: number, b: number) => void): void {
		if (this.nodeCount < 4) return;
		const hash = this.buildEdgeHash(points, searchRadius);
		for (let edgeA = 0; edgeA < this.nodeCount; edgeA += 1) {
			const cellX = hash.ix[edgeA];
			const cellY = hash.iy[edgeA];
			const cellZ = hash.iz[edgeA];
			for (let dx = -1; dx <= 1; dx += 1) {
				for (let dy = -1; dy <= 1; dy += 1) {
					for (let dz = -1; dz <= 1; dz += 1) {
						const bucket = hash.buckets.get(edgeCellKey(cellX + dx, cellY + dy, cellZ + dz));
						if (!bucket) continue;
						for (let k = 0; k < bucket.length; k += 1) {
							const edgeB = bucket[k];
							if (edgeB <= edgeA) continue;
							if (areAdjacentEdges(edgeA, edgeB, this.nodeCount)) continue;
							visitor(edgeA, edgeB);
						}
					}
				}
			}
		}
	}

	private buildEdgeHash(points: Float32Array, searchRadius: number): EdgeHashData {
		const cellSize = Math.max(1e-4, searchRadius);
		const invCellSize = 1 / cellSize;
		const buckets = new Map<string, number[]>();
		for (let edge = 0; edge < this.nodeCount; edge += 1) {
			const edgeNext = (edge + 1) % this.nodeCount;
			const a = edge * 3;
			const b = edgeNext * 3;
			const mx = (points[a] + points[b]) * 0.5;
			const my = (points[a + 1] + points[b + 1]) * 0.5;
			const mz = (points[a + 2] + points[b + 2]) * 0.5;
			const ix = Math.floor(mx * invCellSize);
			const iy = Math.floor(my * invCellSize);
			const iz = Math.floor(mz * invCellSize);
			this.edgeCellX[edge] = ix;
			this.edgeCellY[edge] = iy;
			this.edgeCellZ[edge] = iz;
			const key = edgeCellKey(ix, iy, iz);
			const bucket = buckets.get(key);
			if (bucket) bucket.push(edge);
			else buckets.set(key, [edge]);
		}
		return {
			buckets,
			ix: this.edgeCellX,
			iy: this.edgeCellY,
			iz: this.edgeCellZ
		};
	}

	private applyCurveSmoothing(points: Float32Array, pinnedIndex: number | null): void {
		const weight = this.options.smoothingWeight;
		if (weight <= 1e-6) return;
		const n = this.nodeCount;
		this.smoothBuffer.set(points);
		for (let i = 0; i < n; i += 1) {
			if (i === pinnedIndex) continue;
			const im1 = (i - 1 + n) % n;
			const ip1 = (i + 1) % n;
			const base = i * 3;
			const pBase = im1 * 3;
			const nBase = ip1 * 3;
			this.smoothBuffer[base] =
				points[base] + weight * (0.5 * (points[pBase] + points[nBase]) - points[base]);
			this.smoothBuffer[base + 1] =
				points[base + 1] +
				weight * (0.5 * (points[pBase + 1] + points[nBase + 1]) - points[base + 1]);
			this.smoothBuffer[base + 2] =
				points[base + 2] +
				weight * (0.5 * (points[pBase + 2] + points[nBase + 2]) - points[base + 2]);
		}
		points.set(this.smoothBuffer);
	}

	private addStretchGradient(points: Float32Array): void {
		const weight = this.options.stretchWeight * 2;
		for (let i = 0; i < this.nodeCount; i += 1) {
			const j = (i + 1) % this.nodeCount;
			const ia = i * 3;
			const ja = j * 3;
			const dx = points[ia] - points[ja];
			const dy = points[ia + 1] - points[ja + 1];
			const dz = points[ia + 2] - points[ja + 2];
			const dist = Math.hypot(dx, dy, dz) + 1e-8;
			const diff = dist - this.restLength;
			const coeff = (weight * diff) / dist;
			const gx = coeff * dx;
			const gy = coeff * dy;
			const gz = coeff * dz;
			this.l2Gradient[ia] += gx;
			this.l2Gradient[ia + 1] += gy;
			this.l2Gradient[ia + 2] += gz;
			this.l2Gradient[ja] -= gx;
			this.l2Gradient[ja + 1] -= gy;
			this.l2Gradient[ja + 2] -= gz;
		}
	}

	private measureBendingEnergy(points: Float32Array): number {
		if (this.options.bendingWeight <= 1e-8) return 0;
		let energy = 0;
		const n = this.nodeCount;
		for (let i = 0; i < n; i += 1) {
			const im1 = (i - 1 + n) % n;
			const ip1 = (i + 1) % n;
			const iBase = i * 3;
			const pBase = im1 * 3;
			const nBase = ip1 * 3;
			const ddx = points[pBase] - 2 * points[iBase] + points[nBase];
			const ddy = points[pBase + 1] - 2 * points[iBase + 1] + points[nBase + 1];
			const ddz = points[pBase + 2] - 2 * points[iBase + 2] + points[nBase + 2];
			energy += ddx * ddx + ddy * ddy + ddz * ddz;
		}
		return energy;
	}

	private addBendingGradient(points: Float32Array): void {
		const bendWeight = this.options.bendingWeight;
		if (bendWeight <= 1e-8) return;
		const coeff = bendWeight * 2;
		const n = this.nodeCount;
		for (let i = 0; i < n; i += 1) {
			const im1 = (i - 1 + n) % n;
			const ip1 = (i + 1) % n;
			const iBase = i * 3;
			const pBase = im1 * 3;
			const nBase = ip1 * 3;
			const ddx = points[pBase] - 2 * points[iBase] + points[nBase];
			const ddy = points[pBase + 1] - 2 * points[iBase + 1] + points[nBase + 1];
			const ddz = points[pBase + 2] - 2 * points[iBase + 2] + points[nBase + 2];

			this.l2Gradient[pBase] += coeff * ddx;
			this.l2Gradient[pBase + 1] += coeff * ddy;
			this.l2Gradient[pBase + 2] += coeff * ddz;

			this.l2Gradient[iBase] -= coeff * 2 * ddx;
			this.l2Gradient[iBase + 1] -= coeff * 2 * ddy;
			this.l2Gradient[iBase + 2] -= coeff * 2 * ddz;

			this.l2Gradient[nBase] += coeff * ddx;
			this.l2Gradient[nBase + 1] += coeff * ddy;
			this.l2Gradient[nBase + 2] += coeff * ddz;
		}
	}

	private updateCurveFrames(points: Float32Array): void {
		const n = this.nodeCount;
		for (let i = 0; i < n; i += 1) {
			const im1 = (i - 1 + n) % n;
			const ip1 = (i + 1) % n;
			const iBase = i * 3;
			const pBase = im1 * 3;
			const nBase = ip1 * 3;

			const fx = points[nBase] - points[iBase];
			const fy = points[nBase + 1] - points[iBase + 1];
			const fz = points[nBase + 2] - points[iBase + 2];
			const bx = points[iBase] - points[pBase];
			const by = points[iBase + 1] - points[pBase + 1];
			const bz = points[iBase + 2] - points[pBase + 2];

			const forwardLen = Math.hypot(fx, fy, fz) || 1e-8;
			const backLen = Math.hypot(bx, by, bz) || 1e-8;
			this.edgeLengths[i] = forwardLen;
			this.dualLengths[i] = 0.5 * (forwardLen + backLen);

			let tx = fx / forwardLen + bx / backLen;
			let ty = fy / forwardLen + by / backLen;
			let tz = fz / forwardLen + bz / backLen;
			const tLen = Math.hypot(tx, ty, tz) || 1e-8;
			tx /= tLen;
			ty /= tLen;
			tz /= tLen;
			this.tangents[iBase] = tx;
			this.tangents[iBase + 1] = ty;
			this.tangents[iBase + 2] = tz;
		}
	}
}

function solveAxis(
	solver: KnotSolver,
	rhs: Float32Array,
	out: Float32Array,
	axis: number,
	pinnedIndex: number | null
): void {
	const n = solver['nodeCount'];
	const x = solver['cgX'];
	const r = solver['cgR'];
	const p = solver['cgP'];
	const ap = solver['cgAp'];
	const z = solver['cgZ'];
	const solution = solver['cgOut'];
	x.fill(0);
	r.fill(0);
	p.fill(0);
	ap.fill(0);
	z.fill(0);
	solution.fill(0);

	let normB = 0;
	for (let i = 0; i < n; i += 1) {
		const b = rhs[i * 3 + axis];
		if (i === pinnedIndex) {
			r[i] = 0;
			continue;
		}
		r[i] = b;
		normB += b * b;
	}
	normB = Math.sqrt(normB) + 1e-12;

	let rzOld = 0;
	for (let i = 0; i < n; i += 1) {
		if (i === pinnedIndex) continue;
		z[i] = r[i] / solver['diag'][i];
		p[i] = z[i];
		rzOld += r[i] * z[i];
	}

	for (let iter = 0; iter < solver['options'].cgIterations; iter += 1) {
		solver['applyOperator'](p, ap, pinnedIndex);
		let pAp = 0;
		for (let i = 0; i < n; i += 1) {
			if (i === pinnedIndex) continue;
			pAp += p[i] * ap[i];
		}
		if (!Number.isFinite(pAp) || Math.abs(pAp) < 1e-14) break;

		const alpha = rzOld / pAp;
		for (let i = 0; i < n; i += 1) {
			if (i === pinnedIndex) continue;
			solution[i] += alpha * p[i];
			r[i] -= alpha * ap[i];
		}

		let rNorm = 0;
		for (let i = 0; i < n; i += 1) {
			if (i === pinnedIndex) continue;
			rNorm += r[i] * r[i];
		}
		rNorm = Math.sqrt(rNorm);
		if (rNorm <= solver['options'].cgTolerance * normB) break;

		let rzNew = 0;
		for (let i = 0; i < n; i += 1) {
			if (i === pinnedIndex) continue;
			z[i] = r[i] / solver['diag'][i];
			rzNew += r[i] * z[i];
		}
		if (Math.abs(rzOld) < 1e-20) break;
		const beta = rzNew / rzOld;
		for (let i = 0; i < n; i += 1) {
			if (i === pinnedIndex) continue;
			p[i] = z[i] + beta * p[i];
		}
		rzOld = rzNew;
	}

	for (let i = 0; i < n; i += 1) out[i * 3 + axis] = i === pinnedIndex ? 0 : solution[i];
}

function kernel24(
	dx: number,
	dy: number,
	dz: number,
	tx: number,
	ty: number,
	tz: number,
	dist: number
): number {
	const cx = ty * dz - tz * dy;
	const cy = tz * dx - tx * dz;
	const cz = tx * dy - ty * dx;
	const crossSq = cx * cx + cy * cy + cz * cz;
	return crossSq / Math.pow(dist + 1e-8, 4);
}

function gradientKernelComponent(
	dx: number,
	dy: number,
	dz: number,
	tx: number,
	ty: number,
	tz: number,
	alpha: number,
	beta: number,
	invDistBeta: number,
	invDistBetaPlus2: number
): [number, number, number] {
	const dotT = dx * tx + dy * ty + dz * tz;
	const npx = dx - dotT * tx;
	const npy = dy - dotT * ty;
	const npz = dz - dotT * tz;
	const nNormSq = npx * npx + npy * npy + npz * npz + 1e-12;
	const nNorm = Math.sqrt(nNormSq);
	const nPowAlpha = Math.pow(nNorm, alpha);
	let term1x = 0;
	let term1y = 0;
	let term1z = 0;
	if (nNorm > 1e-8) {
		const coeff = alpha * Math.pow(nNorm, alpha - 2) * invDistBeta;
		term1x = coeff * npx;
		term1y = coeff * npy;
		term1z = coeff * npz;
	}
	const coeff2 = beta * nPowAlpha * invDistBetaPlus2;
	return [term1x - coeff2 * dx, term1y - coeff2 * dy, term1z - coeff2 * dz];
}

function removeMean(vector3: Float32Array): void {
	let mx = 0;
	let my = 0;
	let mz = 0;
	const count = vector3.length / 3;
	for (let i = 0; i < vector3.length; i += 3) {
		mx += vector3[i];
		my += vector3[i + 1];
		mz += vector3[i + 2];
	}
	mx /= count;
	my /= count;
	mz /= count;
	for (let i = 0; i < vector3.length; i += 3) {
		vector3[i] -= mx;
		vector3[i + 1] -= my;
		vector3[i + 2] -= mz;
	}
}

function zeroPinnedVector(vector3: Float32Array, pinnedIndex: number | null): void {
	if (pinnedIndex === null) return;
	const idx = pinnedIndex * 3;
	vector3[idx] = 0;
	vector3[idx + 1] = 0;
	vector3[idx + 2] = 0;
}

function dot(a: Float32Array, b: Float32Array): number {
	let total = 0;
	for (let i = 0; i < a.length; i += 1) total += a[i] * b[i];
	return total;
}

function scale(v: Float32Array, factor: number): void {
	for (let i = 0; i < v.length; i += 1) v[i] *= factor;
}

function recenter(points: Float32Array, count: number): void {
	let cx = 0;
	let cy = 0;
	let cz = 0;
	for (let i = 0; i < count; i += 1) {
		const idx = i * 3;
		cx += points[idx];
		cy += points[idx + 1];
		cz += points[idx + 2];
	}
	cx /= count;
	cy /= count;
	cz /= count;
	for (let i = 0; i < count; i += 1) {
		const idx = i * 3;
		points[idx] -= cx;
		points[idx + 1] -= cy;
		points[idx + 2] -= cz;
	}
}

function ringDistance(a: number, b: number, count: number): number {
	const diff = Math.abs(a - b);
	return Math.min(diff, count - diff);
}

function clamp(value: number, min: number, max: number): number {
	return Math.min(max, Math.max(min, value));
}

function applyPointDisplacement(
	points: Float32Array,
	index: number,
	dx: number,
	dy: number,
	dz: number,
	pinnedIndex: number | null
): void {
	if (index === pinnedIndex) return;
	const idx = index * 3;
	points[idx] += dx;
	points[idx + 1] += dy;
	points[idx + 2] += dz;
}

function applyGradientContribution(
	gradient: Float32Array,
	index: number,
	dx: number,
	dy: number,
	dz: number
): void {
	const idx = index * 3;
	gradient[idx] += dx;
	gradient[idx + 1] += dy;
	gradient[idx + 2] += dz;
}

function areAdjacentEdges(a: number, b: number, count: number): boolean {
	if (a === b) return true;
	const aNext = (a + 1) % count;
	const bNext = (b + 1) % count;
	return a === bNext || b === aNext;
}

function edgeCellKey(ix: number, iy: number, iz: number): string {
	return `${ix},${iy},${iz}`;
}

function closestPointsBetweenEdges(
	points: Float32Array,
	a0: number,
	a1: number,
	b0: number,
	b1: number
): SegmentContact {
	const p0x = points[a0 * 3];
	const p0y = points[a0 * 3 + 1];
	const p0z = points[a0 * 3 + 2];
	const p1x = points[a1 * 3];
	const p1y = points[a1 * 3 + 1];
	const p1z = points[a1 * 3 + 2];
	const q0x = points[b0 * 3];
	const q0y = points[b0 * 3 + 1];
	const q0z = points[b0 * 3 + 2];
	const q1x = points[b1 * 3];
	const q1y = points[b1 * 3 + 1];
	const q1z = points[b1 * 3 + 2];

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

	const cx = p0x + s * ux;
	const cy = p0y + s * uy;
	const cz = p0z + s * uz;
	const dx = cx - (q0x + t * vx);
	const dy = cy - (q0y + t * vy);
	const dz = cz - (q0z + t * vz);

	return { s, t, distSq: dx * dx + dy * dy + dz * dz, dx, dy, dz };
}

function dot3(ax: number, ay: number, az: number, bx: number, by: number, bz: number): number {
	return ax * bx + ay * by + az * bz;
}
