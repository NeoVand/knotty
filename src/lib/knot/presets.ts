export type KnotPresetName = 'trefoil' | 'cinquefoil' | 'septfoil' | 'random_torus';

export interface KnotState {
	name: KnotPresetName;
	label: string;
	points: Float32Array;
	restLength: number;
	thickness: number;
	targetCrossings: number | null;
	p: number;
	q: number;
}

interface CreateOptions {
	nodeCount?: number;
	scramble?: number;
}

interface TorusPreset {
	label: string;
	p: number;
	q: number;
}

const PRESET_CONFIG: Record<Exclude<KnotPresetName, 'random_torus'>, TorusPreset> = {
	trefoil: { label: 'Trefoil T(2,3)', p: 2, q: 3 },
	cinquefoil: { label: 'Cinquefoil T(2,5)', p: 2, q: 5 },
	septfoil: { label: 'Septfoil T(2,7)', p: 2, q: 7 }
};

const RANDOM_PAIRS: Array<[number, number]> = [
	[3, 4],
	[3, 5],
	[4, 5],
	[5, 6],
	[5, 7],
	[6, 7]
];

export const PRESET_ORDER: KnotPresetName[] = ['trefoil', 'cinquefoil', 'septfoil', 'random_torus'];

export function createKnotState(name: KnotPresetName, options: CreateOptions = {}): KnotState {
	const nodeCount = Math.max(80, options.nodeCount ?? 160);
	const scramble = clamp(options.scramble ?? 0.62, 0, 1.4);
	const preset =
		name === 'random_torus'
			? buildRandomPreset()
			: PRESET_CONFIG[name as Exclude<KnotPresetName, 'random_torus'>];

	const base = createTorusKnotPoints(preset.p, preset.q, nodeCount);
	normalizeToRadius(base, 2.1);
	if (scramble > 0) scrambleClosedCurve(base, scramble);

	return {
		name,
		label: name === 'random_torus' ? `Random Torus T(${preset.p},${preset.q})` : preset.label,
		points: base,
		restLength: averageEdgeLength(base),
		thickness: 0.16,
		targetCrossings: torusCrossingNumber(preset.p, preset.q),
		p: preset.p,
		q: preset.q
	};
}

export function copyPoints(points: Float32Array): Float32Array {
	return new Float32Array(points);
}

export function scrambleClosedCurve(points: Float32Array, amount: number): void {
	const count = points.length / 3;
	const amplitude = 0.38 * amount;
	const warpA = 2 + Math.floor(Math.random() * 4);
	const warpB = 3 + Math.floor(Math.random() * 5);
	const phaseA = Math.random() * Math.PI * 2;
	const phaseB = Math.random() * Math.PI * 2;

	for (let i = 0; i < count; i += 1) {
		const i0 = (i - 1 + count) % count;
		const i1 = (i + 1) % count;

		const tx = points[i1 * 3] - points[i0 * 3];
		const ty = points[i1 * 3 + 1] - points[i0 * 3 + 1];
		const tz = points[i1 * 3 + 2] - points[i0 * 3 + 2];
		const tLen = Math.hypot(tx, ty, tz) || 1;
		const nxRaw = Math.abs(tx / tLen) < 0.85 ? 1 : 0;
		const nyRaw = Math.abs(ty / tLen) < 0.85 ? 1 : 0;
		const nzRaw = Math.abs(tz / tLen) < 0.85 ? 1 : 0;

		let ux = ty * nzRaw - tz * nyRaw;
		let uy = tz * nxRaw - tx * nzRaw;
		let uz = tx * nyRaw - ty * nxRaw;
		const uLen = Math.hypot(ux, uy, uz) || 1;
		ux /= uLen;
		uy /= uLen;
		uz /= uLen;

		let vx = ty * uz - tz * uy;
		let vy = tz * ux - tx * uz;
		let vz = tx * uy - ty * ux;
		const vLen = Math.hypot(vx, vy, vz) || 1;
		vx /= vLen;
		vy /= vLen;
		vz /= vLen;

		const a = Math.sin(((i / count) * Math.PI * 2 * warpA) + phaseA);
		const b = Math.cos(((i / count) * Math.PI * 2 * warpB) + phaseB);
		const jitterA = (Math.random() - 0.5) * 0.35;
		const jitterB = (Math.random() - 0.5) * 0.35;
		const offsetA = amplitude * (a + jitterA);
		const offsetB = amplitude * (b + jitterB);
		const base = i * 3;
		points[base] += ux * offsetA + vx * offsetB;
		points[base + 1] += uy * offsetA + vy * offsetB;
		points[base + 2] += uz * offsetA + vz * offsetB;
	}

	normalizeToRadius(points, 2.1);
}

function buildRandomPreset(): TorusPreset {
	const [p, q] = RANDOM_PAIRS[Math.floor(Math.random() * RANDOM_PAIRS.length)];
	return { label: `Random Torus T(${p},${q})`, p, q };
}

function createTorusKnotPoints(p: number, q: number, count: number): Float32Array {
	const points = new Float32Array(count * 3);
	const majorRadius = 1.35;
	const minorRadius = 0.55;

	for (let i = 0; i < count; i += 1) {
		const t = (i / count) * Math.PI * 2;
		const ct = Math.cos(t * p);
		const st = Math.sin(t * p);
		const cq = Math.cos(t * q);
		const sq = Math.sin(t * q);
		const r = majorRadius + minorRadius * cq;
		const idx = i * 3;
		points[idx] = r * ct;
		points[idx + 1] = r * st;
		points[idx + 2] = minorRadius * sq;
	}

	return points;
}

function torusCrossingNumber(p: number, q: number): number {
	return Math.min((p - 1) * q, (q - 1) * p);
}

function averageEdgeLength(points: Float32Array): number {
	const count = points.length / 3;
	let sum = 0;

	for (let i = 0; i < count; i += 1) {
		const j = (i + 1) % count;
		sum += distance(points, i, j);
	}

	return sum / count;
}

function normalizeToRadius(points: Float32Array, radius: number): void {
	const count = points.length / 3;
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

	let maxNorm = 1e-6;
	for (let i = 0; i < count; i += 1) {
		const idx = i * 3;
		points[idx] -= cx;
		points[idx + 1] -= cy;
		points[idx + 2] -= cz;
		maxNorm = Math.max(maxNorm, Math.hypot(points[idx], points[idx + 1], points[idx + 2]));
	}

	const scale = radius / maxNorm;
	for (let i = 0; i < points.length; i += 1) points[i] *= scale;
}

function distance(points: Float32Array, i: number, j: number): number {
	const ia = i * 3;
	const ja = j * 3;
	return Math.hypot(
		points[ia] - points[ja],
		points[ia + 1] - points[ja + 1],
		points[ia + 2] - points[ja + 2]
	);
}

function clamp(value: number, min: number, max: number): number {
	return Math.min(max, Math.max(min, value));
}
