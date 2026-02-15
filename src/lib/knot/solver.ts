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
	constraintPasses: number;
	edgeCollisionPasses: number;
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
	constraintPasses: 5,
	edgeCollisionPasses: 2
};

interface SegmentContact {
	s: number;
	t: number;
	distSq: number;
	dx: number;
	dy: number;
	dz: number;
}

export class KnotSolver {
	private readonly nodeCount: number;
	private readonly restLength: number;
	private readonly options: SolverOptions;
	private lastStepSize = 0.018;

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
	}

	reset(): void {
		this.lastStepSize = 0.018;
	}

	zeroVelocity(_index: number): void {
		// No-op: this solver integrates a constrained descent direction directly.
	}

	step(points: Float32Array, dt: number, pinnedIndex: number | null, thickness: number): void {
		const baseEnergy = this.measureEnergy(points, thickness);
		this.fillL2Gradient(points);
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
			this.applyEdgeCollisions(this.candidate, pinnedIndex, thickness * 1.08);
			if (pinnedIndex === null) recenter(this.candidate, this.nodeCount);

			const nextEnergy = this.measureEnergy(this.candidate, thickness);
			const armijoTarget = baseEnergy - this.options.armijo * stepSize * Math.max(1e-8, gradDotDirection);

			if (nextEnergy < baseEnergy || nextEnergy <= armijoTarget) {
				points.set(this.candidate);
				this.lastStepSize = stepSize;
				accepted = true;
				break;
			}

			stepSize *= 0.5;
			if (stepSize < this.options.minStep) break;
		}

		if (!accepted) this.lastStepSize = Math.max(this.options.minStep, this.lastStepSize * 0.6);
	}

	measureEnergy(points: Float32Array, thickness: number): number {
		this.updateCurveFrames(points);
		const { alpha, beta } = this.options;
		let energy = 0;
		let stretchPenalty = 0;
		let collisionPenalty = 0;
		const minDistance = Math.max(1e-4, thickness * 1.22);

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
				energy += weight * (Math.pow(normI, alpha) + Math.pow(normJ, alpha)) * invDistBeta;

				if (dist < minDistance) {
					const penetration = minDistance - dist;
					collisionPenalty += penetration * penetration;
				}
			}
		}

		return energy + 28 * stretchPenalty + 120 * collisionPenalty;
	}

	private fillL2Gradient(points: Float32Array): void {
		this.updateCurveFrames(points);
		this.l2Gradient.fill(0);
		const { alpha, beta } = this.options;

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
				const gx = weight * (gradIx[0] + gradJx[0]);
				const gy = weight * (gradIx[1] + gradJx[1]);
				const gz = weight * (gradIx[2] + gradJx[2]);

				this.l2Gradient[pi] += gx;
				this.l2Gradient[pi + 1] += gy;
				this.l2Gradient[pi + 2] += gz;
				this.l2Gradient[pj] -= gx;
				this.l2Gradient[pj + 1] -= gy;
				this.l2Gradient[pj + 2] -= gz;
			}
		}
	}

	private buildPreconditioner(points: Float32Array, thickness: number): void {
		this.updateCurveFrames(points);
		const { alpha, beta, lowOrderWeight, regularization, highOrderWeight } = this.options;
		const sigma = (beta - 1) / alpha - 1;
		const n = this.nodeCount;
		const minDist = Math.max(thickness * 0.85, 1e-3);

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

	private applyEdgeCollisions(points: Float32Array, pinnedIndex: number | null, thickness: number): void {
		const passes = this.options.edgeCollisionPasses;
		for (let pass = 0; pass < passes; pass += 1) {
			for (let edgeA = 0; edgeA < this.nodeCount; edgeA += 1) {
				const edgeANext = (edgeA + 1) % this.nodeCount;
				for (let edgeB = edgeA + 1; edgeB < this.nodeCount; edgeB += 1) {
					const edgeBNext = (edgeB + 1) % this.nodeCount;
					if (areAdjacentEdges(edgeA, edgeB, this.nodeCount)) continue;

					const contact = closestPointsBetweenEdges(points, edgeA, edgeANext, edgeB, edgeBNext);
					const dist = Math.sqrt(contact.distSq);
					if (!Number.isFinite(dist) || dist >= thickness) continue;

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

					const push = (thickness - dist) * 0.5;
					const a0 = 1 - contact.s;
					const a1 = contact.s;
					const b0 = 1 - contact.t;
					const b1 = contact.t;

					applyPointDisplacement(points, edgeA, nx * push * a0, ny * push * a0, nz * push * a0, pinnedIndex);
					applyPointDisplacement(
						points,
						edgeANext,
						nx * push * a1,
						ny * push * a1,
						nz * push * a1,
						pinnedIndex
					);
					applyPointDisplacement(
						points,
						edgeB,
						-nx * push * b0,
						-ny * push * b0,
						-nz * push * b0,
						pinnedIndex
					);
					applyPointDisplacement(
						points,
						edgeBNext,
						-nx * push * b1,
						-ny * push * b1,
						-nz * push * b1,
						pinnedIndex
					);
				}
			}
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

function areAdjacentEdges(a: number, b: number, count: number): boolean {
	if (a === b) return true;
	const aNext = (a + 1) % count;
	const bNext = (b + 1) % count;
	return a === bNext || b === aNext;
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
