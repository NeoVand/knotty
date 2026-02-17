import { generateShadowKnot } from './shadow-knot-generator';

export type KnotPresetName = 'trefoil' | 'cinquefoil' | 'septfoil' | 'random_shadow';

export interface KnotState {
	name: KnotPresetName;
	label: string;
	points: Float32Array;
	componentsPoints: Float32Array[];
	componentRestLengths: number[];
	restLength: number;
	thickness: number;
	targetCrossings: number | null;
	componentCount: number;
	p: number;
	q: number;
}

interface CreateOptions {
	nodeCount?: number;
	useArcGuideLayout?: boolean;
	seed?: number;
	crossings?: number;
}

interface ShadowPreset {
	label: string;
	crossings: number;
	seed: number;
}

const PRESET_CONFIG: Record<Exclude<KnotPresetName, 'random_shadow'>, ShadowPreset> = {
	trefoil: { label: '3-Crossing Shadow', crossings: 3, seed: 0x1a2b3c4d },
	cinquefoil: { label: '5-Crossing Shadow', crossings: 5, seed: 0x5e7d3f1a },
	septfoil: { label: '7-Crossing Shadow', crossings: 7, seed: 0x92d3b4c1 }
};

const RANDOM_CROSSINGS = [5, 6, 7, 8, 9, 10, 11, 12];
export const PRESET_ORDER: KnotPresetName[] = ['trefoil', 'cinquefoil', 'septfoil', 'random_shadow'];

export function createKnotState(name: KnotPresetName, options: CreateOptions = {}): KnotState {
	const nodeCount = Math.max(80, options.nodeCount ?? 180);
	const thickness = 0.16;
	const useArcGuideLayout = options.useArcGuideLayout ?? true;
	const preset =
		name === 'random_shadow'
			? buildRandomPreset(options.seed, options.crossings)
			: PRESET_CONFIG[name];
	const generated = generateShadowKnot({
		crossings: preset.crossings,
		nodeCount,
		seed: options.seed ?? preset.seed,
		tubeRadius: thickness,
		useArcGuideLayout
	});
	const points = generated.points;
	const componentsPoints = generated.componentsPoints;
	const generatedCrossings = generated.crossings;
	const label =
		name === 'random_shadow'
			? options.crossings === undefined
				? `Random Shadow (${generated.crossings} crossings)`
				: `Shadow (${generated.crossings} crossings)`
			: preset.label;

	return {
		name,
		label,
		points,
		componentsPoints,
		componentRestLengths: componentsPoints.map((component) => averageEdgeLength(component)),
		restLength: averageEdgeLength(points),
		thickness,
		targetCrossings: generatedCrossings,
		componentCount: generated.components,
		p: 0,
		q: generatedCrossings
	};
}

export function copyPoints(points: Float32Array): Float32Array {
	return new Float32Array(points);
}

function buildRandomPreset(seed?: number, crossingsOverride?: number): ShadowPreset {
	const crossings =
		crossingsOverride === undefined
			? RANDOM_CROSSINGS[Math.floor(Math.random() * RANDOM_CROSSINGS.length)]
			: Math.max(3, Math.round(crossingsOverride));
	return {
		label: `Random Shadow (${crossings} crossings)`,
		crossings,
		seed: normalizeSeed(seed ?? Math.floor(Math.random() * 0xffffffff))
	};
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

function distance(points: Float32Array, i: number, j: number): number {
	const ia = i * 3;
	const ja = j * 3;
	return Math.hypot(
		points[ia] - points[ja],
		points[ia + 1] - points[ja + 1],
		points[ia + 2] - points[ja + 2]
	);
}

function normalizeSeed(seed: number): number {
	const normalized = seed >>> 0;
	return normalized === 0 ? 1 : normalized;
}
