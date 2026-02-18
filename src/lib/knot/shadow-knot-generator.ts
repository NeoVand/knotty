import { Delaunay } from 'd3-delaunay';

interface Point {
	x: number;
	y: number;
}

interface Point3 {
	x: number;
	y: number;
	z: number;
}

interface PrimalEdge {
	u: number;
	v: number;
}

interface Face {
	vertices: number[];
	edges: number[];
}

interface MedialEdge {
	a: number;
	b: number;
	control: Point;
}

interface CubicCurve {
	p0: Point;
	p1: Point;
	p2: Point;
	p3: Point;
}

interface IncidentStub {
	halfId: number;
	control: Point;
	outer: boolean;
}

interface DecoratedStub extends IncidentStub {
	dir: Point;
	angle: number;
	point: Point;
}

interface RenderCurve {
	curve: CubicCurve;
	startHalfId: number;
	endHalfId: number;
}

interface ArcGuideSample {
	u: number;
	v: number;
	t: number;
}

interface Segment3 {
	to: number;
	curve: CubicCurve;
	z0: number;
	z1: number;
	kind: 'edge' | 'crossing';
	sign: number;
}

interface ComponentDescriptor {
	startHalf: number;
	size: number;
}

interface PointRange {
	start: number;
	count: number;
}

interface Candidate {
	edgeByHalf: Array<Segment3 | null>;
	crossingByHalf: Array<Segment3 | null>;
	componentDescriptors: ComponentDescriptor[];
	components: number;
	crossings: number;
	quality: number;
	seed: number;
	minCrossingAngleDeg: number;
}

interface BuildCandidateOptions {
	cornerInset: number;
	strokeWidth: number;
	tubeRadius: number;
	padding: number;
	width: number;
	height: number;
	rng: () => number;
	seed: number;
}

export interface ShadowKnotOptions {
	crossings: number;
	nodeCount: number;
	seed: number;
	tubeRadius?: number;
	useArcGuideLayout?: boolean;
	maxAttempts?: number;
	cornerInset?: number;
	strokeWidth?: number;
}

export interface ShadowKnotResult {
	points: Float32Array;
	componentsPoints: Float32Array[];
	crossings: number;
	components: number;
	quality: number;
	seed: number;
	minCrossingAngleDeg: number;
}

const TAU = Math.PI * 2;
const GOLDEN_GAMMA = 0x9e3779b9;

export function generateShadowKnot(options: ShadowKnotOptions): ShadowKnotResult {
	const normalized = normalizeRequestedCrossings(options.crossings);
	const nodeCount = Math.max(80, options.nodeCount);
	const tubeRadius = Math.max(0.02, options.tubeRadius ?? 0.16);
	const cornerInset = clamp(options.cornerInset ?? 0.21, 0.04, 0.9);
	const strokeWidth = clamp(options.strokeWidth ?? 3.6, 1.1, 20);
	const useArcGuideLayout = options.useArcGuideLayout ?? false;
	const width = 960;
	const height = 960;
	const padding = 56;
	const maxAttempts =
		options.maxAttempts ?? Math.round(42 + Math.sqrt(normalized.target) * 5.2 + Math.max(0, strokeWidth - 6) * 5);
	const baseSeed = normalizeSeed(options.seed);
	const { primalVertices, hullVertices } = choosePrimalSize(normalized.target);

	let best: Candidate | null = null;
	for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
		const attemptSeed = (baseSeed + Math.imul(attempt + 1, GOLDEN_GAMMA)) >>> 0;
		const rng = mulberry32(attemptSeed);
		const pointsRaw = generatePoints(primalVertices, hullVertices, width, height, padding, rng);
		let points = pointsRaw;
		let primal = buildPrimalGraph(points, normalized.target, hullVertices);
		if (!primal) continue;

		if (useArcGuideLayout) {
			const relaxed = relaxWithArcGuideNodes(points, primal.edges, hullVertices, width, height, padding);
			const relaxedPrimal = buildPrimalGraph(relaxed, normalized.target, hullVertices);
			if (relaxedPrimal) {
				points = relaxed;
				primal = relaxedPrimal;
			}
		}

		const candidate = buildCandidate(points, primal, {
			cornerInset,
			strokeWidth,
			tubeRadius,
			padding,
			width,
			height,
			rng,
			seed: attemptSeed
		});
		if (!candidate) continue;

		if (!best || candidate.quality < best.quality) {
			best = candidate;
		}

		const qualityThreshold = 0.58 + Math.max(0, strokeWidth - 10) * 0.005;
		const angleThreshold = 49 + Math.max(0, strokeWidth - 8) * 0.2;
		if (attempt >= 5 && candidate.quality <= qualityThreshold && candidate.minCrossingAngleDeg >= angleThreshold) {
			break;
		}
	}

	if (!best && useArcGuideLayout) {
		return generateShadowKnot({
			...options,
			useArcGuideLayout: false,
			maxAttempts
		});
	}

	if (!best) {
		throw new Error('Failed to generate a valid 4-regular planar shadow.');
	}

	const sampledComponents = sampleCandidateComponents(best, nodeCount);
	if (!sampledComponents) {
		throw new Error('Failed to sample generated knot candidate.');
	}
	const packed = packComponents(sampledComponents);
	normalizeToRadius3D(packed.points, 2.1);
	ensureInitialClearance(packed.points, tubeRadius * 2.15, packed.ranges);
	const normalizedComponents = unpackComponents(packed.points, packed.ranges);

	return {
		points: normalizedComponents[0],
		componentsPoints: normalizedComponents,
		crossings: best.crossings,
		components: best.components,
		quality: best.quality,
		seed: best.seed,
		minCrossingAngleDeg: best.minCrossingAngleDeg
	};
}

function buildCandidate(
	points: Point[],
	primal: { edges: PrimalEdge[]; faces: Face[] },
	options: BuildCandidateOptions
): Candidate | null {
	const { cornerInset, strokeWidth, tubeRadius, padding, width, height, rng, seed } = options;
	const viewportCenter = { x: width * 0.5, y: height * 0.5 };
	const crossingPositions = primal.edges.map((edge) => ({
		x: (points[edge.u].x + points[edge.v].x) * 0.5,
		y: (points[edge.u].y + points[edge.v].y) * 0.5
	}));
	const medialEdges: MedialEdge[] = [];
	const incidentByCrossing: IncidentStub[][] = Array.from({ length: crossingPositions.length }, () => []);

	for (const face of primal.faces) {
		const centroid = face.vertices.reduce(
			(acc, vi) => {
				acc.x += points[vi].x;
				acc.y += points[vi].y;
				return acc;
			},
			{ x: 0, y: 0 }
		);
		centroid.x /= face.vertices.length;
		centroid.y /= face.vertices.length;

		const k = face.edges.length;
		const isOuterFace = k > 3;

		for (let i = 0; i < k; i += 1) {
			const a = face.edges[i];
			const b = face.edges[(i + 1) % k];
			const cornerVertex = face.vertices[(i + 1) % k];
			const control = isOuterFace
				? outerFaceControl(
						points[face.vertices[i]],
						points[cornerVertex],
						points[face.vertices[(i + 2) % k]],
						viewportCenter,
						cornerInset,
						padding
					)
				: lerp(points[cornerVertex], centroid, cornerInset);

			const edgeIndex = medialEdges.length;
			medialEdges.push({ a, b, control });
			incidentByCrossing[a].push({ halfId: edgeIndex * 2, control, outer: isOuterFace });
			incidentByCrossing[b].push({ halfId: edgeIndex * 2 + 1, control, outer: isOuterFace });
		}
	}

	if (incidentByCrossing.some((stubs) => stubs.length !== 4)) return null;

	let armLengthSum = 0;
	const minArmByCrossing = new Float64Array(incidentByCrossing.length);
	for (let i = 0; i < incidentByCrossing.length; i += 1) {
		const center = crossingPositions[i];
		const stubs = incidentByCrossing[i];
		let minArm = Number.POSITIVE_INFINITY;
		for (let j = 0; j < stubs.length; j += 1) {
			const arm = length(sub(stubs[j].control, center));
			armLengthSum += arm;
			if (arm < minArm) minArm = arm;
		}
		minArmByCrossing[i] = Number.isFinite(minArm) ? minArm : 1;
	}

	const averageArmLength = armLengthSum / (incidentByCrossing.length * 4);
	const densityScale = Math.sqrt((width * height) / crossingPositions.length);
	const desiredTrim = strokeWidth * (0.58 + cornerInset * 0.16) + 0.7;
	const geometricRadius = Math.min(averageArmLength * 0.34, densityScale * 0.18);
	const radiusTarget = Math.min(geometricRadius, desiredTrim + strokeWidth * 0.42 + 1.9);
	const radiusCapByStroke = desiredTrim + strokeWidth * 0.78 + 4.5;
	const radiusCapByGeometry = Math.min(averageArmLength * 0.52, densityScale * 0.3);
	const radiusCap = Math.max(desiredTrim + 0.45, Math.min(radiusCapByGeometry, radiusCapByStroke));
	const nominalRadius = clamp(radiusTarget, desiredTrim, radiusCap);

	const sortedStubsByCrossing: DecoratedStub[][] = Array.from({ length: crossingPositions.length }, () => []);
	const halfEdgePoints: Point[] = Array.from({ length: medialEdges.length * 2 }, () => ({ x: 0, y: 0 }));
	const halfEdgeOutDirs: Point[] = Array.from({ length: medialEdges.length * 2 }, () => ({ x: 1, y: 0 }));
	let trimSum = 0;
	let radiusDeficitPenalty = 0;
	let severeRadiusDeficits = 0;

	for (let i = 0; i < crossingPositions.length; i += 1) {
		const center = crossingPositions[i];
		const safeRadius = Math.min(nominalRadius, Math.max(0.55, minArmByCrossing[i] * 0.8));
		const sorted = incidentByCrossing[i]
			.map((stub) => {
				const rawDir = normalize(sub(stub.control, center));
				return {
					...stub,
					rawDir,
					controlAngle: Math.atan2(stub.control.y - center.y, stub.control.x - center.x)
				};
			})
			.sort((left, right) => left.controlAngle - right.controlAngle);

		if (sorted.length !== 4) return null;

		let sinSum = 0;
		let cosSum = 0;
		for (let j = 0; j < 4; j += 1) {
			const phase = sorted[j].controlAngle - j * (Math.PI * 0.5);
			sinSum += Math.sin(phase);
			cosSum += Math.cos(phase);
		}

		const baseRotation = Math.atan2(sinSum, cosSum);
		let maxBlendedTrim = Number.POSITIVE_INFINITY;
		const interiorBlend = clamp(0.16 + cornerInset * 0.5, 0.14, 0.45);
		const blendedDirs: Point[] = [];
		for (let j = 0; j < 4; j += 1) {
			const theta = baseRotation + j * (Math.PI * 0.5);
			let idealDir = { x: Math.cos(theta), y: Math.sin(theta) };
			const rawDir = sorted[j].rawDir;
			if (dot(rawDir, idealDir) < 0) {
				idealDir = scale(idealDir, -1);
			}
			const outerBlend = clamp(0.06 + (1 - cornerInset) * 0.22, 0.05, 0.28);
			const blend = sorted[j].outer ? outerBlend : interiorBlend;
			const dir = normalize(add(scale(rawDir, 1 - blend), scale(idealDir, blend)));
			blendedDirs.push(dir);
			const support = dot(sub(sorted[j].control, center), dir);
			maxBlendedTrim = Math.min(maxBlendedTrim, support * 0.8);
		}

		const blendedRadius = Math.max(0.8, Math.min(safeRadius, maxBlendedTrim));
		if (blendedRadius < desiredTrim) {
			const deficit = (desiredTrim - blendedRadius) / desiredTrim;
			radiusDeficitPenalty += deficit * deficit * 4.6;
		}
		if (blendedRadius < desiredTrim * 0.94) {
			severeRadiusDeficits += 1;
		}

		const decorated: DecoratedStub[] = [];
		for (let j = 0; j < 4; j += 1) {
			const dir = blendedDirs[j];
			const point = add(center, scale(dir, blendedRadius));
			const halfId = sorted[j].halfId;
			halfEdgePoints[halfId] = point;
			halfEdgeOutDirs[halfId] = dir;
			trimSum += blendedRadius;
			decorated.push({
				halfId,
				control: sorted[j].control,
				outer: sorted[j].outer,
				dir,
				angle: Math.atan2(dir.y, dir.x),
				point
			});
		}

		decorated.sort((left, right) => left.angle - right.angle);
		sortedStubsByCrossing[i] = decorated;
	}

	if (severeRadiusDeficits > 0) return null;

	const crossingRadius = trimSum > 0 ? trimSum / (medialEdges.length * 2) : nominalRadius;
	const desiredAdjacentGap = Math.max(strokeWidth * 0.9 + 3, tubeRadius * 10.2);
	let anglePenalty = 0;
	let gapPenalty = 0;
	let minCrossingAngle = Math.PI * 0.5;

	for (let i = 0; i < crossingPositions.length; i += 1) {
		const sorted = sortedStubsByCrossing[i];
		if (!sorted || sorted.length !== 4) continue;

		const strandA = normalize(sub(sorted[2].point, sorted[0].point));
		const strandB = normalize(sub(sorted[3].point, sorted[1].point));
		const strandDot = Math.abs(dot(strandA, strandB));
		const angle = Math.acos(clamp(strandDot, 0, 1));
		minCrossingAngle = Math.min(minCrossingAngle, angle);
		anglePenalty += strandDot * strandDot;

		for (let j = 0; j < 4; j += 1) {
			const current = sorted[j];
			const next = sorted[(j + 1) % 4];
			let gap = next.angle - current.angle;
			if (gap <= 0) gap += TAU;
			const normalizedGap = gap / (Math.PI * 0.5);
			anglePenalty += Math.abs(normalizedGap - 1) * 0.34;
			const adjacentDistance = length(sub(next.point, current.point));
			if (adjacentDistance < desiredAdjacentGap) {
				const gapDeficit = (desiredAdjacentGap - adjacentDistance) / desiredAdjacentGap;
				gapPenalty += gapDeficit * gapDeficit * (0.95 + strokeWidth * 0.04);
			}
		}
	}

	const strokePressure = 1 + Math.max(0, strokeWidth - 4) * 0.14;
	const quality =
		(anglePenalty + gapPenalty * (1.8 * strokePressure) + radiusDeficitPenalty * (1.35 * strokePressure)) /
		Math.max(1, crossingPositions.length);
	const minCrossingAngleDeg = (minCrossingAngle * 180) / Math.PI;

	const halfCount = medialEdges.length * 2;
	const edgeByHalf: Array<Segment3 | null> = Array.from({ length: halfCount }, () => null);
	const crossingByHalf: Array<Segment3 | null> = Array.from({ length: halfCount }, () => null);
	const halfSign = new Float32Array(halfCount);
	const baseCurves: RenderCurve[] = [];
	const overCurves: RenderCurve[] = [];

	for (let i = 0; i < medialEdges.length; i += 1) {
		const edge = medialEdges[i];
		const halfA = i * 2;
		const halfB = i * 2 + 1;
		const start = halfEdgePoints[halfA];
		const end = halfEdgePoints[halfB];
		const startOut = halfEdgeOutDirs[halfA];
		const endOut = halfEdgeOutDirs[halfB];
		const curve = smoothEdgeCurve(start, end, edge.control, startOut, endOut, crossingRadius);
		baseCurves.push({ curve, startHalfId: halfA, endHalfId: halfB });
		edgeByHalf[halfA] = { to: halfB, curve, z0: 0, z1: 0, kind: 'edge', sign: 0 };
		edgeByHalf[halfB] = { to: halfA, curve: reverseCurve(curve), z0: 0, z1: 0, kind: 'edge', sign: 0 };
	}

	const zLift = Math.max(8, crossingRadius * 2.25, tubeRadius * 22);
	for (let i = 0; i < crossingPositions.length; i += 1) {
		const center = crossingPositions[i];
		const sorted = sortedStubsByCrossing[i];
		if (!sorted || sorted.length !== 4) return null;

		const overPairBase = rng() < 0.5 ? 0 : 1;
		const underPairBase = overPairBase === 0 ? 1 : 0;
		connectPair(sorted, overPairBase, 1, center, crossingRadius, zLift, crossingByHalf, halfSign, overCurves, true);
		connectPair(
			sorted,
			underPairBase,
			-1,
			center,
			crossingRadius,
			zLift,
			crossingByHalf,
			halfSign,
			overCurves,
			false
		);
	}

	for (let i = 0; i < halfCount; i += 1) {
		if (Math.abs(halfSign[i]) < 0.5) return null;
		const edge = edgeByHalf[i];
		if (!edge) return null;
		const z0 = halfSign[i] * zLift * 0.88;
		const z1 = halfSign[edge.to] * zLift * 0.88;
		edge.z0 = z0;
		edge.z1 = z1;
		edge.sign = Math.sign(z0 + z1);
	}

	if (hasUnwantedIntersections([...baseCurves, ...overCurves], strokeWidth)) return null;

	const componentDescriptors = analyzeComponents(edgeByHalf, crossingByHalf);
	if (componentDescriptors.length < 1) return null;

	return {
		edgeByHalf,
		crossingByHalf,
		componentDescriptors,
		components: componentDescriptors.length,
		crossings: crossingPositions.length,
		quality,
		seed,
		minCrossingAngleDeg
	};
}

function connectPair(
	sorted: DecoratedStub[],
	baseIndex: number,
	sign: number,
	center: Point,
	crossingRadius: number,
	zLift: number,
	crossingByHalf: Array<Segment3 | null>,
	halfSign: Float32Array,
	overCurves: RenderCurve[],
	recordOverCurve: boolean
): void {
	const start = sorted[baseIndex];
	const end = sorted[baseIndex + 2];
	halfSign[start.halfId] = sign;
	halfSign[end.halfId] = sign;
	const startTangent = scale(start.dir, -1);
	const endTangent = end.dir;
	const curve = smoothOverpassCurve(start.point, end.point, center, startTangent, endTangent, crossingRadius);
	const z = sign * zLift;
	crossingByHalf[start.halfId] = {
		to: end.halfId,
		curve,
		z0: z,
		z1: z,
		kind: 'crossing',
		sign
	};
	crossingByHalf[end.halfId] = {
		to: start.halfId,
		curve: reverseCurve(curve),
		z0: z,
		z1: z,
		kind: 'crossing',
		sign
	};
	if (recordOverCurve) {
		overCurves.push({ curve, startHalfId: start.halfId, endHalfId: end.halfId });
	}
}

function analyzeComponents(
	edgeByHalf: Array<Segment3 | null>,
	crossingByHalf: Array<Segment3 | null>
): ComponentDescriptor[] {
	const halfCount = edgeByHalf.length;
	const visited = new Uint8Array(halfCount);
	const descriptors: ComponentDescriptor[] = [];

	for (let start = 0; start < halfCount; start += 1) {
		if (visited[start]) continue;
		if (!edgeByHalf[start]) continue;

		let current = start;
		let guard = 0;
		let componentSize = 0;
		const guardLimit = Math.max(8, halfCount * 6);
		while (!visited[current] && guard < guardLimit) {
			visited[current] = 1;
			const edge = edgeByHalf[current];
			if (!edge) return [];
			const crossing = crossingByHalf[edge.to];
			if (!crossing) return [];
			current = crossing.to;
			guard += 1;
			componentSize += 1;
		}

		if (guard >= guardLimit) return [];
		descriptors.push({ startHalf: start, size: componentSize });
	}

	descriptors.sort((left, right) => right.size - left.size);
	return descriptors;
}

function sampleCandidateComponents(candidate: Candidate, targetCount: number): Float32Array[] | null {
	const descriptors = candidate.componentDescriptors;
	if (descriptors.length === 0) return null;

	const minPerComponent = 24;
	const requestedTotal = Math.max(targetCount, minPerComponent * descriptors.length);
	const totalSize = descriptors.reduce((sum, descriptor) => sum + descriptor.size, 0);
	const counts = descriptors.map((descriptor) =>
		Math.max(minPerComponent, Math.round((requestedTotal * descriptor.size) / Math.max(1, totalSize)))
	);

	let assigned = counts.reduce((sum, count) => sum + count, 0);
	while (assigned < requestedTotal) {
		counts[0] += 1;
		assigned += 1;
	}
	while (assigned > requestedTotal) {
		let changed = false;
		for (let i = 0; i < counts.length && assigned > requestedTotal; i += 1) {
			if (counts[i] > minPerComponent) {
				counts[i] -= 1;
				assigned -= 1;
				changed = true;
			}
		}
		if (!changed) break;
	}

	const components: Float32Array[] = [];
	for (let i = 0; i < descriptors.length; i += 1) {
		const sampled = sampleComponentCycle(candidate, descriptors[i].startHalf, counts[i]);
		if (!sampled) return null;
		components.push(sampled);
	}

	return components;
}

function sampleComponentCycle(candidate: Candidate, startHalf: number, targetCount: number): Float32Array | null {
	const samples: Point3[] = [];
	let current = startHalf;
	let guard = 0;
	const guardLimit = Math.max(16, candidate.edgeByHalf.length * 8);

	while (guard < guardLimit) {
		const edge = candidate.edgeByHalf[current];
		if (!edge) return null;
		appendSegmentSamples(samples, edge);

		const crossing = candidate.crossingByHalf[edge.to];
		if (!crossing) return null;
		appendSegmentSamples(samples, crossing);

		current = crossing.to;
		guard += 1;
		if (current === startHalf) break;
	}

	if (guard >= guardLimit) return null;
	if (samples.length < 12) return null;

	const resampled = resampleClosedPolyline(samples, targetCount);
	if (!resampled) return null;
	smoothClosedCurve(resampled, 2, 0.16);
	return points3ToFloat32Array(resampled);
}

function appendSegmentSamples(target: Point3[], segment: Segment3): void {
	const lengthEstimate = estimateCubicLength(segment.curve);
	const steps = clamp(Math.ceil(lengthEstimate / 12), 6, 38);
	const startIndex = target.length === 0 ? 0 : 1;
	for (let i = startIndex; i <= steps; i += 1) {
		const t = i / steps;
		const point = cubicPoint(segment.curve, t);
		const ease = t * t * (3 - 2 * t);
		const z = segment.z0 + (segment.z1 - segment.z0) * ease;
		target.push({ x: point.x, y: point.y, z });
	}
}

function resampleClosedPolyline(points: Point3[], count: number): Point3[] | null {
	const filtered = dedupePoints(points);
	if (filtered.length < 4) return null;

	const n = filtered.length;
	const cumulative = new Float64Array(n + 1);
	let total = 0;
	for (let i = 0; i < n; i += 1) {
		const next = (i + 1) % n;
		total += distance3(filtered[i], filtered[next]);
		cumulative[i + 1] = total;
	}
	if (!Number.isFinite(total) || total < 1e-6) return null;

	const target = Math.max(24, count);
	const out: Point3[] = [];
	let seg = 0;
	for (let i = 0; i < target; i += 1) {
		const d = (i / target) * total;
		while (seg < n - 1 && cumulative[seg + 1] < d) {
			seg += 1;
		}
		const segStart = cumulative[seg];
		const segEnd = cumulative[seg + 1];
		const span = Math.max(1e-9, segEnd - segStart);
		const t = clamp((d - segStart) / span, 0, 1);
		const a = filtered[seg];
		const b = filtered[(seg + 1) % n];
		out.push({
			x: a.x + (b.x - a.x) * t,
			y: a.y + (b.y - a.y) * t,
			z: a.z + (b.z - a.z) * t
		});
	}

	return out;
}

function dedupePoints(points: Point3[]): Point3[] {
	const out: Point3[] = [];
	for (let i = 0; i < points.length; i += 1) {
		const point = points[i];
		if (out.length === 0) {
			out.push(point);
			continue;
		}
		const prev = out[out.length - 1];
		if (distance3(point, prev) > 1e-4) out.push(point);
	}
	if (out.length > 1 && distance3(out[0], out[out.length - 1]) < 1e-4) out.pop();
	return out;
}

function smoothClosedCurve(points: Point3[], iterations: number, alpha: number): void {
	const n = points.length;
	const copy = Array.from({ length: n }, () => ({ x: 0, y: 0, z: 0 }));
	for (let iter = 0; iter < iterations; iter += 1) {
		for (let i = 0; i < n; i += 1) {
			const prev = points[(i - 1 + n) % n];
			const next = points[(i + 1) % n];
			copy[i].x = points[i].x + alpha * (0.5 * (prev.x + next.x) - points[i].x);
			copy[i].y = points[i].y + alpha * (0.5 * (prev.y + next.y) - points[i].y);
			copy[i].z = points[i].z + alpha * (0.5 * (prev.z + next.z) - points[i].z);
		}
		for (let i = 0; i < n; i += 1) {
			points[i].x = copy[i].x;
			points[i].y = copy[i].y;
			points[i].z = copy[i].z;
		}
	}
}

function points3ToFloat32Array(points: Point3[]): Float32Array {
	const out = new Float32Array(points.length * 3);
	for (let i = 0; i < points.length; i += 1) {
		const base = i * 3;
		out[base] = points[i].x;
		out[base + 1] = points[i].y;
		out[base + 2] = points[i].z;
	}
	return out;
}

function packComponents(components: Float32Array[]): { points: Float32Array; ranges: PointRange[] } {
	const totalPoints = components.reduce((sum, component) => sum + component.length / 3, 0);
	const points = new Float32Array(totalPoints * 3);
	const ranges: PointRange[] = [];
	let cursor = 0;
	for (const component of components) {
		const count = component.length / 3;
		ranges.push({ start: cursor, count });
		points.set(component, cursor * 3);
		cursor += count;
	}
	return { points, ranges };
}

function unpackComponents(points: Float32Array, ranges: PointRange[]): Float32Array[] {
	return ranges.map((range) => points.slice(range.start * 3, (range.start + range.count) * 3));
}

function normalizeToRadius3D(points: Float32Array, radius: number): void {
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

	const scaleFactor = radius / maxNorm;
	for (let i = 0; i < points.length; i += 1) points[i] *= scaleFactor;
}

function ensureInitialClearance(points: Float32Array, minEdgeDistance: number, ranges: PointRange[]): void {
	const activeRanges = ranges.filter((range) => range.count > 1);
	const totalPoints = activeRanges.reduce((sum, range) => sum + range.count, 0);
	if (totalPoints < 8) return;
	const edges = buildEdgeRefs(activeRanges);
	if (edges.length < 4) return;
	const baseEdgeLengths = averageEdgeLengths3(points, activeRanges);

	// Guarantee over/under separation at crossings survives final normalization.
	const halfGap = minEdgeDistance * 0.5;
	let maxAbsZ = 0;
	for (let i = 2; i < points.length; i += 3) {
		maxAbsZ = Math.max(maxAbsZ, Math.abs(points[i]));
	}
	if (maxAbsZ > 1e-6 && maxAbsZ < halfGap) {
		const zScale = halfGap / maxAbsZ;
		for (let i = 2; i < points.length; i += 3) points[i] *= zScale;
	}

	// Resolve residual non-adjacent edge collisions in the generated state.
	for (let pass = 0; pass < 6; pass += 1) {
		let moved = false;
		for (let edgeAIndex = 0; edgeAIndex < edges.length; edgeAIndex += 1) {
			const edgeA = edges[edgeAIndex];
			for (let edgeBIndex = edgeAIndex + 1; edgeBIndex < edges.length; edgeBIndex += 1) {
				const edgeB = edges[edgeBIndex];
				if (
					edgeA.rangeIndex === edgeB.rangeIndex &&
					areAdjacentEdges(edgeA.localIndex, edgeB.localIndex, activeRanges[edgeA.rangeIndex].count)
				) {
					continue;
				}
				const contact = closestPointsBetweenEdges3(points, edgeA.start, edgeA.end, edgeB.start, edgeB.end);
				const dist = Math.sqrt(contact.distSq);
				if (!Number.isFinite(dist) || dist >= minEdgeDistance) continue;

				let nx = contact.dx;
				let ny = contact.dy;
				let nz = contact.dz;
				let normalLength = Math.hypot(nx, ny, nz);
				if (normalLength < 1e-7) {
					nx = points[edgeA.end * 3] - points[edgeA.start * 3];
					ny = points[edgeA.end * 3 + 1] - points[edgeA.start * 3 + 1];
					nz = points[edgeA.end * 3 + 2] - points[edgeA.start * 3 + 2];
					normalLength = Math.hypot(nx, ny, nz) || 1;
				}
				nx /= normalLength;
				ny /= normalLength;
				nz /= normalLength;

				const push = (minEdgeDistance - dist) * 0.48;
				const a0 = 1 - contact.s;
				const a1 = contact.s;
				const b0 = 1 - contact.t;
				const b1 = contact.t;

				applyPointDisplacement3(points, edgeA.start, nx * push * a0, ny * push * a0, nz * push * a0);
				applyPointDisplacement3(points, edgeA.end, nx * push * a1, ny * push * a1, nz * push * a1);
				applyPointDisplacement3(points, edgeB.start, -nx * push * b0, -ny * push * b0, -nz * push * b0);
				applyPointDisplacement3(points, edgeB.end, -nx * push * b1, -ny * push * b1, -nz * push * b1);
				moved = true;
			}
		}
		if (!moved) break;
		// Preserve baseline rope length while removing overlaps.
		projectEdgeLengths(points, baseEdgeLengths, activeRanges, 2);
	}

	recenter3(points);
}

interface EdgeRef {
	start: number;
	end: number;
	rangeIndex: number;
	localIndex: number;
}

function buildEdgeRefs(ranges: PointRange[]): EdgeRef[] {
	const edges: EdgeRef[] = [];
	for (let rangeIndex = 0; rangeIndex < ranges.length; rangeIndex += 1) {
		const range = ranges[rangeIndex];
		for (let localIndex = 0; localIndex < range.count; localIndex += 1) {
			edges.push({
				start: range.start + localIndex,
				end: range.start + ((localIndex + 1) % range.count),
				rangeIndex,
				localIndex
			});
		}
	}
	return edges;
}

interface SegmentContact3 {
	s: number;
	t: number;
	distSq: number;
	dx: number;
	dy: number;
	dz: number;
}

function closestPointsBetweenEdges3(
	points: Float32Array,
	a0: number,
	a1: number,
	b0: number,
	b1: number
): SegmentContact3 {
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
		if (-d < 0) {
			sN = 0;
		} else if (-d > a) {
			sN = sD;
		} else {
			sN = -d;
			sD = a;
		}
	} else if (tN > tD) {
		tN = tD;
		if (-d + b < 0) {
			sN = 0;
		} else if (-d + b > a) {
			sN = sD;
		} else {
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

function applyPointDisplacement3(points: Float32Array, index: number, dx: number, dy: number, dz: number): void {
	const idx = index * 3;
	points[idx] += dx;
	points[idx + 1] += dy;
	points[idx + 2] += dz;
}

function projectEdgeLengths(
	points: Float32Array,
	targetLengths: Float64Array,
	ranges: PointRange[],
	passes: number
): void {
	for (let pass = 0; pass < passes; pass += 1) {
		for (let rangeIndex = 0; rangeIndex < ranges.length; rangeIndex += 1) {
			const range = ranges[rangeIndex];
			const targetLength = targetLengths[rangeIndex];
			if (range.count < 2 || !Number.isFinite(targetLength) || targetLength <= 0) continue;
			for (let localIndex = 0; localIndex < range.count; localIndex += 1) {
				const i = range.start + localIndex;
				const j = range.start + ((localIndex + 1) % range.count);
				const ia = i * 3;
				const ja = j * 3;
				const dx = points[ja] - points[ia];
				const dy = points[ja + 1] - points[ia + 1];
				const dz = points[ja + 2] - points[ia + 2];
				const dist = Math.hypot(dx, dy, dz) || 1e-8;
				const correction = (dist - targetLength) / dist;
				points[ia] += dx * correction * 0.5;
				points[ia + 1] += dy * correction * 0.5;
				points[ia + 2] += dz * correction * 0.5;
				points[ja] -= dx * correction * 0.5;
				points[ja + 1] -= dy * correction * 0.5;
				points[ja + 2] -= dz * correction * 0.5;
			}
		}
	}
}

function averageEdgeLengths3(points: Float32Array, ranges: PointRange[]): Float64Array {
	const out = new Float64Array(ranges.length);
	for (let rangeIndex = 0; rangeIndex < ranges.length; rangeIndex += 1) {
		const range = ranges[rangeIndex];
		if (range.count < 2) {
			out[rangeIndex] = 0;
			continue;
		}
		let total = 0;
		for (let localIndex = 0; localIndex < range.count; localIndex += 1) {
			const i = range.start + localIndex;
			const j = range.start + ((localIndex + 1) % range.count);
			const ia = i * 3;
			const ja = j * 3;
			total += Math.hypot(
				points[ia] - points[ja],
				points[ia + 1] - points[ja + 1],
				points[ia + 2] - points[ja + 2]
			);
		}
		out[rangeIndex] = total / range.count;
	}
	return out;
}

function recenter3(points: Float32Array): void {
	const n = points.length / 3;
	let cx = 0;
	let cy = 0;
	let cz = 0;
	for (let i = 0; i < n; i += 1) {
		const idx = i * 3;
		cx += points[idx];
		cy += points[idx + 1];
		cz += points[idx + 2];
	}
	cx /= n;
	cy /= n;
	cz /= n;
	for (let i = 0; i < n; i += 1) {
		const idx = i * 3;
		points[idx] -= cx;
		points[idx + 1] -= cy;
		points[idx + 2] -= cz;
	}
}

function areAdjacentEdges(a: number, b: number, count: number): boolean {
	if (a === b) return true;
	const aNext = (a + 1) % count;
	const bNext = (b + 1) % count;
	return a === bNext || b === aNext;
}

function dot3(ax: number, ay: number, az: number, bx: number, by: number, bz: number): number {
	return ax * bx + ay * by + az * bz;
}

function normalizeRequestedCrossings(raw: number): { target: number } {
	const rounded = Math.max(1, Math.round(raw));
	if (rounded <= 3) return { target: 3 };
	if (rounded === 4) return { target: 5 };
	return { target: rounded };
}

function choosePrimalSize(targetCrossings: number): { primalVertices: number; hullVertices: number } {
	const minVertices = Math.ceil((targetCrossings + 6) / 3);
	const maxVertices = Math.floor((targetCrossings + 3) / 2);
	if (minVertices > maxVertices) {
		throw new Error('Unable to choose valid primal graph size.');
	}

	const desiredHull = clamp(Math.round(Math.sqrt(targetCrossings) + 4), 3, maxVertices);
	const desiredInterior = Math.max(1, Math.round(Math.sqrt(targetCrossings) * 0.45));
	let best: { primalVertices: number; hullVertices: number; score: number } | null = null;

	for (let primalVertices = minVertices; primalVertices <= maxVertices; primalVertices += 1) {
		const hullVertices = 3 * primalVertices - 3 - targetCrossings;
		if (hullVertices < 3 || hullVertices > primalVertices) continue;
		const interiorVertices = primalVertices - hullVertices;
		const hullPenalty = Math.abs(hullVertices - desiredHull);
		const interiorPenalty = interiorVertices < desiredInterior ? (desiredInterior - interiorVertices) * 1.35 : 0;
		const score = hullPenalty + interiorPenalty;
		if (
			!best ||
			score < best.score ||
			(score === best.score && hullVertices > best.hullVertices) ||
			(score === best.score && hullVertices === best.hullVertices && primalVertices < best.primalVertices)
		) {
			best = { primalVertices, hullVertices, score };
		}
	}
	if (!best) throw new Error('Failed to solve primal constraints.');
	return { primalVertices: best.primalVertices, hullVertices: best.hullVertices };
}

function generatePoints(
	primalVertices: number,
	hullVertices: number,
	width: number,
	height: number,
	padding: number,
	rng: () => number
): Point[] {
	const cx = width / 2;
	const cy = height / 2;
	const rx = Math.max(8, width / 2 - padding);
	const ry = Math.max(8, height / 2 - padding);
	const points: Point[] = [];
	const phase = rng() * TAU;
	const wobblePhase = rng() * TAU;

	for (let i = 0; i < hullVertices; i += 1) {
		const theta = phase + (TAU * i) / hullVertices;
		const wobble = 1 + 0.035 * Math.sin(theta * 3 + wobblePhase);
		points.push({
			x: cx + rx * wobble * Math.cos(theta),
			y: cy + ry * wobble * Math.sin(theta)
		});
	}

	const interiorCount = primalVertices - hullVertices;
	const hull = points.slice(0, hullVertices);
	const fanAreas: number[] = [];
	let totalFanArea = 0;

	for (let i = 1; i < hullVertices - 1; i += 1) {
		const a = hull[0];
		const b = hull[i];
		const c = hull[i + 1];
		const doubleArea = Math.abs((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x));
		fanAreas.push(doubleArea);
		totalFanArea += doubleArea;
	}

	for (let i = 0; i < interiorCount; i += 1) {
		let tri = 0;
		let pick = rng() * totalFanArea;
		while (tri < fanAreas.length - 1 && pick > fanAreas[tri]) {
			pick -= fanAreas[tri];
			tri += 1;
		}

		const a = hull[0];
		const b = hull[tri + 1];
		const c = hull[tri + 2];
		let u = rng();
		let v = rng();
		if (u + v > 1) {
			u = 1 - u;
			v = 1 - v;
		}
		const sample = {
			x: a.x + u * (b.x - a.x) + v * (c.x - a.x),
			y: a.y + u * (b.y - a.y) + v * (c.y - a.y)
		};
		const inwardPull = 0.04 + rng() * 0.06;
		points.push(lerp(sample, { x: cx, y: cy }, inwardPull));
	}

	return points;
}

function buildPrimalGraph(
	points: Point[],
	targetCrossings: number,
	expectedHull: number
): { edges: PrimalEdge[]; faces: Face[] } | null {
	const delaunay = Delaunay.from(
		points,
		(point) => point.x,
		(point) => point.y
	);
	const hull = Array.from(delaunay.hull);
	if (hull.length !== expectedHull) return null;

	const edges: PrimalEdge[] = [];
	const edgeMap = new Map<string, number>();
	const getEdgeId = (u: number, v: number): number => {
		const key = edgeKey(u, v);
		const existing = edgeMap.get(key);
		if (existing !== undefined) return existing;
		const id = edges.length;
		edges.push({ u: Math.min(u, v), v: Math.max(u, v) });
		edgeMap.set(key, id);
		return id;
	};

	const faces: Face[] = [];
	const tris = delaunay.triangles;
	for (let i = 0; i < tris.length; i += 3) {
		const a = tris[i];
		const b = tris[i + 1];
		const c = tris[i + 2];
		faces.push({
			vertices: [a, b, c],
			edges: [getEdgeId(a, b), getEdgeId(b, c), getEdgeId(c, a)]
		});
	}

	const outerEdges: number[] = [];
	for (let i = 0; i < hull.length; i += 1) {
		const a = hull[i];
		const b = hull[(i + 1) % hull.length];
		outerEdges.push(getEdgeId(a, b));
	}
	faces.push({ vertices: hull, edges: outerEdges });

	if (edges.length !== targetCrossings) return null;
	return { edges, faces };
}

function relaxWithArcGuideNodes(
	pointsInput: Point[],
	edges: PrimalEdge[],
	hullVertices: number,
	width: number,
	height: number,
	padding: number
): Point[] {
	if (pointsInput.length <= hullVertices + 1 || edges.length === 0) return pointsInput;
	const points = pointsInput.map((point) => ({ ...point }));
	const count = points.length;
	const cx = width * 0.5;
	const cy = height * 0.5;
	const rx = Math.max(14, width * 0.5 - padding - 4);
	const ry = Math.max(14, height * 0.5 - padding - 4);

	const samplesPerEdge = 2;
	const sampleEdgeBudget = count <= 260 ? edges.length : 220;
	const edgeStride = Math.max(1, Math.ceil(edges.length / sampleEdgeBudget));
	const samples: ArcGuideSample[] = [];
	for (let edgeIndex = 0; edgeIndex < edges.length; edgeIndex += edgeStride) {
		const edge = edges[edgeIndex];
		for (let i = 1; i <= samplesPerEdge; i += 1) {
			samples.push({ u: edge.u, v: edge.v, t: i / (samplesPerEdge + 1) });
		}
	}

	const ghostPoints: Point[] = Array.from({ length: samples.length }, () => ({ x: 0, y: 0 }));
	const forces: Point[] = Array.from({ length: count }, () => ({ x: 0, y: 0 }));
	const velocities: Point[] = Array.from({ length: count }, () => ({ x: 0, y: 0 }));

	const meanEdgeLength =
		edges.reduce((sum, edge) => sum + length(sub(points[edge.u], points[edge.v])), 0) / edges.length;
	const targetEdgeLength = Math.max(6, meanEdgeLength);
	const drawableArea = Math.max(1, (width - padding * 2) * (height - padding * 2));
	const spacing = Math.sqrt(drawableArea / Math.max(1, count + samples.length * 0.85));
	const springK = 0.0065;
	const repulseVV = spacing * spacing * 0.018;
	const repulseVG = spacing * spacing * 0.046;
	const repulseGG = spacing * spacing * 0.07;
	const centerK = 0.0012;
	const boundaryK = 0.11;
	const ghostReaction = 0.72;
	const softness = 36;
	const runVertexRepulsion = count <= 420;
	const runGhostGhost = samples.length <= 560;
	const iterations = count <= 220 ? 24 : count <= 600 ? 14 : 9;

	for (let iteration = 0; iteration < iterations; iteration += 1) {
		for (let i = 0; i < count; i += 1) {
			forces[i].x = 0;
			forces[i].y = 0;
		}

		for (const edge of edges) {
			const delta = sub(points[edge.v], points[edge.u]);
			const dist = Math.max(1e-6, length(delta));
			const dir = scale(delta, 1 / dist);
			const mag = (dist - targetEdgeLength) * springK;
			const fx = dir.x * mag;
			const fy = dir.y * mag;
			forces[edge.u].x += fx;
			forces[edge.u].y += fy;
			forces[edge.v].x -= fx;
			forces[edge.v].y -= fy;
		}

		if (runVertexRepulsion) {
			for (let i = 0; i < count; i += 1) {
				for (let j = i + 1; j < count; j += 1) {
					const dx = points[i].x - points[j].x;
					const dy = points[i].y - points[j].y;
					const distSq = dx * dx + dy * dy + softness;
					const invDist = 1 / Math.sqrt(distSq);
					const mag = repulseVV / distSq;
					const fx = dx * invDist * mag;
					const fy = dy * invDist * mag;
					forces[i].x += fx;
					forces[i].y += fy;
					forces[j].x -= fx;
					forces[j].y -= fy;
				}
			}
		}

		for (let i = 0; i < samples.length; i += 1) {
			const sample = samples[i];
			ghostPoints[i] = lerp(points[sample.u], points[sample.v], sample.t);
		}

		for (let vertex = 0; vertex < count; vertex += 1) {
			for (let i = 0; i < samples.length; i += 1) {
				const sample = samples[i];
				if (vertex === sample.u || vertex === sample.v) continue;
				const dx = points[vertex].x - ghostPoints[i].x;
				const dy = points[vertex].y - ghostPoints[i].y;
				const distSq = dx * dx + dy * dy + softness;
				const invDist = 1 / Math.sqrt(distSq);
				const mag = repulseVG / distSq;
				const fx = dx * invDist * mag;
				const fy = dy * invDist * mag;
				forces[vertex].x += fx;
				forces[vertex].y += fy;

				const backFx = -fx * ghostReaction;
				const backFy = -fy * ghostReaction;
				const uWeight = 1 - sample.t;
				const vWeight = sample.t;
				forces[sample.u].x += backFx * uWeight;
				forces[sample.u].y += backFy * uWeight;
				forces[sample.v].x += backFx * vWeight;
				forces[sample.v].y += backFy * vWeight;
			}
		}

		if (runGhostGhost) {
			for (let i = 0; i < samples.length; i += 1) {
				const left = samples[i];
				for (let j = i + 1; j < samples.length; j += 1) {
					const right = samples[j];
					if ((left.u === right.u && left.v === right.v) || (left.u === right.v && left.v === right.u)) {
						continue;
					}
					const dx = ghostPoints[i].x - ghostPoints[j].x;
					const dy = ghostPoints[i].y - ghostPoints[j].y;
					const distSq = dx * dx + dy * dy + softness;
					const invDist = 1 / Math.sqrt(distSq);
					const mag = repulseGG / distSq;
					const fx = dx * invDist * mag;
					const fy = dy * invDist * mag;
					forces[left.u].x += fx * (1 - left.t);
					forces[left.u].y += fy * (1 - left.t);
					forces[left.v].x += fx * left.t;
					forces[left.v].y += fy * left.t;
					forces[right.u].x -= fx * (1 - right.t);
					forces[right.u].y -= fy * (1 - right.t);
					forces[right.v].x -= fx * right.t;
					forces[right.v].y -= fy * right.t;
				}
			}
		}

		for (let i = hullVertices; i < count; i += 1) {
			const towardCenter = sub({ x: cx, y: cy }, points[i]);
			forces[i].x += towardCenter.x * centerK;
			forces[i].y += towardCenter.y * centerK;

			const dx = points[i].x - cx;
			const dy = points[i].y - cy;
			const radial = Math.sqrt((dx * dx) / (rx * rx) + (dy * dy) / (ry * ry));
			if (radial > 0.95) {
				const inward = (radial - 0.95) * boundaryK;
				forces[i].x -= dx * inward;
				forces[i].y -= dy * inward;
			}
		}

		for (let i = hullVertices; i < count; i += 1) {
			velocities[i].x = velocities[i].x * 0.78 + forces[i].x * 0.68;
			velocities[i].y = velocities[i].y * 0.78 + forces[i].y * 0.68;
			const maxStep = Math.max(2.3, spacing * 0.065);
			const speed = Math.hypot(velocities[i].x, velocities[i].y);
			if (speed > maxStep) {
				const stepScale = maxStep / speed;
				velocities[i].x *= stepScale;
				velocities[i].y *= stepScale;
			}
			const moved = { x: points[i].x + velocities[i].x, y: points[i].y + velocities[i].y };
			points[i] = clampInsideEllipse(moved, cx, cy, rx, ry);
		}
	}

	return points;
}

function outerFaceControl(
	prev: Point,
	current: Point,
	next: Point,
	viewportCenter: Point,
	cornerInset: number,
	padding: number
): Point {
	const towardPrev = normalize(sub(prev, current));
	const towardNext = normalize(sub(next, current));
	let inwardBisector = add(towardPrev, towardNext);
	if (length(inwardBisector) < 1e-9) {
		inwardBisector = normalize(sub(viewportCenter, current));
	} else {
		inwardBisector = normalize(inwardBisector);
	}
	const outwardBisector = scale(inwardBisector, -1);
	const localScale = Math.min(length(sub(prev, current)), length(sub(next, current)));
	const inset01 = clamp((cornerInset - 0.04) / 0.86, 0, 1);
	const easedInset = inset01 * inset01 * (3 - 2 * inset01);
	const minOffset = Math.max(5, localScale * 0.11);
	const softCap = Math.max(minOffset + 1, padding * (0.95 + easedInset * 2.65));
	const desiredOffset = localScale * (0.16 + easedInset * 0.92);
	const offset = clamp(desiredOffset, minOffset, softCap);
	return add(current, scale(outwardBisector, offset));
}

function smoothEdgeCurve(
	start: Point,
	end: Point,
	guide: Point,
	startOutDir: Point,
	endOutDir: Point,
	crossingRadius: number
): CubicCurve {
	const chord = sub(end, start);
	const chordLength = Math.max(1e-6, length(chord));
	const startDir = normalize(startOutDir);
	const endDir = normalize(endOutDir);
	const minHandle = Math.max(1.2, crossingRadius * 0.38);
	const maxHandle = Math.max(minHandle, Math.min(chordLength * 0.49, crossingRadius * 3.1));
	const guideProjectionStart = Math.max(0.05, dot(sub(guide, start), startDir));
	const guideProjectionEnd = Math.max(0.05, dot(sub(guide, end), endDir));
	const guideDistanceFromChord =
		Math.abs(chord.x * (guide.y - start.y) - chord.y * (guide.x - start.x)) / chordLength;
	const chordBlend = chordLength * 0.32 + guideDistanceFromChord * 0.24;
	const startReach = Math.max(0, dot(chord, startDir));
	const endReach = Math.max(0, dot(scale(chord, -1), endDir));
	const startHandleLength = clamp(
		Math.max(chordBlend * 0.58, guideProjectionStart * 0.7, startReach * 0.28),
		minHandle,
		maxHandle
	);
	const endHandleLength = clamp(
		Math.max(chordBlend * 0.58, guideProjectionEnd * 0.7, endReach * 0.28),
		minHandle,
		maxHandle
	);
	return {
		p0: start,
		p1: add(start, scale(startDir, startHandleLength)),
		p2: add(end, scale(endDir, endHandleLength)),
		p3: end
	};
}

function smoothOverpassCurve(
	start: Point,
	end: Point,
	center: Point,
	startTangent: Point,
	endTangent: Point,
	crossingRadius: number
): CubicCurve {
	const chord = sub(end, start);
	const chordLength = Math.max(1e-6, length(chord));
	const startDir = normalize(startTangent);
	const endDir = normalize(endTangent);
	const minHandle = Math.max(1.15, crossingRadius * 0.42);
	const maxHandle = Math.max(minHandle, Math.min(chordLength * 0.46, crossingRadius * 2.2));
	const centerReach =
		Math.abs(dot(sub(center, start), startDir)) + Math.abs(dot(sub(center, end), endDir));
	const chordReach = chordLength * 0.34;
	const targetHandle = Math.max(crossingRadius * 0.56, Math.min(chordReach, centerReach * 0.31));
	const startHandleLength = clamp(targetHandle, minHandle, maxHandle);
	const endHandleLength = clamp(targetHandle, minHandle, maxHandle);
	return {
		p0: start,
		p1: add(start, scale(startDir, startHandleLength)),
		p2: sub(end, scale(endDir, endHandleLength)),
		p3: end
	};
}

function hasUnwantedIntersections(curves: RenderCurve[], strokeWidth: number): boolean {
	if (curves.length < 2) return false;

	interface SegmentSample {
		curveIndex: number;
		t0: number;
		t1: number;
		a: Point;
		b: Point;
		minX: number;
		maxX: number;
		minY: number;
		maxY: number;
	}

	const fullClearance = Math.max(0.9, strokeWidth * 0.09 + 0.65);
	const endpointClearance = Math.max(0.65, fullClearance * 0.55);
	const endpointSlackT = clamp(0.095 + strokeWidth * 0.0032, 0.1, 0.2);
	const endpointBandT = clamp(0.14 + strokeWidth * 0.0026, 0.14, 0.22);
	const cellSize = Math.max(14, fullClearance * 1.42);
	const segments: SegmentSample[] = [];

	for (let i = 0; i < curves.length; i += 1) {
		const curve = curves[i].curve;
		const approxLength =
			length(sub(curve.p1, curve.p0)) + length(sub(curve.p2, curve.p1)) + length(sub(curve.p3, curve.p2));
		const steps = clamp(Math.ceil(approxLength / Math.max(8, fullClearance * 0.62)), 10, 44);
		let prevPoint = curve.p0;
		let prevT = 0;
		for (let k = 1; k <= steps; k += 1) {
			const t = k / steps;
			const point = k === steps ? curve.p3 : cubicPoint(curve, t);
			if (distanceSquared(prevPoint, point) > 1e-8) {
				segments.push({
					curveIndex: i,
					t0: prevT,
					t1: t,
					a: prevPoint,
					b: point,
					minX: Math.min(prevPoint.x, point.x),
					maxX: Math.max(prevPoint.x, point.x),
					minY: Math.min(prevPoint.y, point.y),
					maxY: Math.max(prevPoint.y, point.y)
				});
			}
			prevPoint = point;
			prevT = t;
		}
	}

	const bucket = new Map<string, number[]>();
	const checkedPairs = new Set<string>();

	const nearSharedEndpoint = (segment: SegmentSample, curve: RenderCurve, sharedHalfId: number): boolean => {
		if (curve.startHalfId === sharedHalfId && segment.t1 <= endpointSlackT) return true;
		if (curve.endHalfId === sharedHalfId && segment.t0 >= 1 - endpointSlackT) return true;
		return false;
	};

	const bboxDistanceSq = (left: SegmentSample, right: SegmentSample): number => {
		const dx = Math.max(0, Math.max(left.minX - right.maxX, right.minX - left.maxX));
		const dy = Math.max(0, Math.max(left.minY - right.maxY, right.minY - left.maxY));
		return dx * dx + dy * dy;
	};

	for (let i = 0; i < segments.length; i += 1) {
		const segment = segments[i];
		const minX = segment.minX - fullClearance;
		const maxX = segment.maxX + fullClearance;
		const minY = segment.minY - fullClearance;
		const maxY = segment.maxY + fullClearance;
		const gx0 = Math.floor(minX / cellSize);
		const gx1 = Math.floor(maxX / cellSize);
		const gy0 = Math.floor(minY / cellSize);
		const gy1 = Math.floor(maxY / cellSize);

		for (let gx = gx0; gx <= gx1; gx += 1) {
			for (let gy = gy0; gy <= gy1; gy += 1) {
				const key = `${gx},${gy}`;
				const seen = bucket.get(key);
				if (!seen) continue;

				for (let idx = 0; idx < seen.length; idx += 1) {
					const otherIndex = seen[idx];
					const pairKey = `${otherIndex}:${i}`;
					if (checkedPairs.has(pairKey)) continue;
					checkedPairs.add(pairKey);
					const other = segments[otherIndex];
					if (segment.curveIndex === other.curveIndex) continue;
					const curve = curves[segment.curveIndex];
					const otherCurve = curves[other.curveIndex];
					const sharedStart = curve.startHalfId;
					const sharedEnd = curve.endHalfId;

					let sharedHalfId = -1;
					if (sharedStart === otherCurve.startHalfId || sharedStart === otherCurve.endHalfId) {
						sharedHalfId = sharedStart;
					} else if (sharedEnd === otherCurve.startHalfId || sharedEnd === otherCurve.endHalfId) {
						sharedHalfId = sharedEnd;
					}

					if (
						sharedHalfId >= 0 &&
						nearSharedEndpoint(segment, curve, sharedHalfId) &&
						nearSharedEndpoint(other, otherCurve, sharedHalfId)
					) {
						continue;
					}

					const nearEndpoint =
						segment.t0 <= endpointBandT ||
						segment.t1 >= 1 - endpointBandT ||
						other.t0 <= endpointBandT ||
						other.t1 >= 1 - endpointBandT;
					const clearance = nearEndpoint ? endpointClearance : fullClearance;
					const clearanceSq = clearance * clearance;
					if (bboxDistanceSq(segment, other) >= clearanceSq) continue;
					if (segmentDistanceSquared(segment.a, segment.b, other.a, other.b) < clearanceSq) return true;
				}
			}
		}

		for (let gx = gx0; gx <= gx1; gx += 1) {
			for (let gy = gy0; gy <= gy1; gy += 1) {
				const key = `${gx},${gy}`;
				const list = bucket.get(key);
				if (list) list.push(i);
				else bucket.set(key, [i]);
			}
		}
	}

	return false;
}

function segmentDistanceSquared(a0: Point, a1: Point, b0: Point, b1: Point): number {
	const eps = 1e-9;
	const u = sub(a1, a0);
	const v = sub(b1, b0);
	const w = sub(a0, b0);
	const aa = dot(u, u);
	const b = dot(u, v);
	const c = dot(v, v);
	const d = dot(u, w);
	const e = dot(v, w);
	const denom = aa * c - b * b;
	let sNumerator = 0;
	let sDenominator = denom;
	let tNumerator = 0;
	let tDenominator = denom;

	if (denom < eps) {
		sNumerator = 0;
		sDenominator = 1;
		tNumerator = e;
		tDenominator = c;
	} else {
		sNumerator = b * e - c * d;
		tNumerator = aa * e - b * d;
		if (sNumerator < 0) {
			sNumerator = 0;
			tNumerator = e;
			tDenominator = c;
		} else if (sNumerator > sDenominator) {
			sNumerator = sDenominator;
			tNumerator = e + b;
			tDenominator = c;
		}
	}

	if (tNumerator < 0) {
		tNumerator = 0;
		if (-d < 0) sNumerator = 0;
		else if (-d > aa) sNumerator = sDenominator;
		else {
			sNumerator = -d;
			sDenominator = aa;
		}
	} else if (tNumerator > tDenominator) {
		tNumerator = tDenominator;
		if (-d + b < 0) sNumerator = 0;
		else if (-d + b > aa) sNumerator = sDenominator;
		else {
			sNumerator = -d + b;
			sDenominator = aa;
		}
	}

	const s = Math.abs(sNumerator) < eps ? 0 : sNumerator / sDenominator;
	const t = Math.abs(tNumerator) < eps ? 0 : tNumerator / tDenominator;
	const delta = sub(add(w, scale(u, s)), scale(v, t));
	return dot(delta, delta);
}

function clampInsideEllipse(point: Point, cx: number, cy: number, rx: number, ry: number): Point {
	const dx = point.x - cx;
	const dy = point.y - cy;
	const norm = (dx * dx) / (rx * rx) + (dy * dy) / (ry * ry);
	if (norm <= 1) return point;
	const factor = 1 / Math.sqrt(norm);
	return { x: cx + dx * factor, y: cy + dy * factor };
}

function reverseCurve(curve: CubicCurve): CubicCurve {
	return {
		p0: curve.p3,
		p1: curve.p2,
		p2: curve.p1,
		p3: curve.p0
	};
}

function estimateCubicLength(curve: CubicCurve): number {
	return length(sub(curve.p1, curve.p0)) + length(sub(curve.p2, curve.p1)) + length(sub(curve.p3, curve.p2));
}

function cubicPoint(curve: CubicCurve, t: number): Point {
	const u = 1 - t;
	const uu = u * u;
	const tt = t * t;
	const uuu = uu * u;
	const ttt = tt * t;
	return {
		x: uuu * curve.p0.x + 3 * uu * t * curve.p1.x + 3 * u * tt * curve.p2.x + ttt * curve.p3.x,
		y: uuu * curve.p0.y + 3 * uu * t * curve.p1.y + 3 * u * tt * curve.p2.y + ttt * curve.p3.y
	};
}

function distanceSquared(a: Point, b: Point): number {
	const dx = a.x - b.x;
	const dy = a.y - b.y;
	return dx * dx + dy * dy;
}

function distance3(a: Point3, b: Point3): number {
	const dx = a.x - b.x;
	const dy = a.y - b.y;
	const dz = a.z - b.z;
	return Math.hypot(dx, dy, dz);
}

function normalizeSeed(seed: number): number {
	const normalized = seed >>> 0;
	return normalized === 0 ? 1 : normalized;
}

function mulberry32(seed: number): () => number {
	let t = seed >>> 0;
	return () => {
		t = (t + 0x6d2b79f5) >>> 0;
		let x = Math.imul(t ^ (t >>> 15), 1 | t);
		x ^= x + Math.imul(x ^ (x >>> 7), 61 | x);
		return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
	};
}

function edgeKey(u: number, v: number): string {
	return u < v ? `${u}:${v}` : `${v}:${u}`;
}

function clamp(value: number, min: number, max: number): number {
	return Math.min(max, Math.max(min, value));
}

function lerp(a: Point, b: Point, t: number): Point {
	return {
		x: a.x + (b.x - a.x) * t,
		y: a.y + (b.y - a.y) * t
	};
}

function sub(a: Point, b: Point): Point {
	return { x: a.x - b.x, y: a.y - b.y };
}

function add(a: Point, b: Point): Point {
	return { x: a.x + b.x, y: a.y + b.y };
}

function scale(a: Point, k: number): Point {
	return { x: a.x * k, y: a.y * k };
}

function dot(a: Point, b: Point): number {
	return a.x * b.x + a.y * b.y;
}

function length(a: Point): number {
	return Math.hypot(a.x, a.y);
}

function normalize(a: Point): Point {
	const len = length(a);
	if (len < 1e-9) return { x: 1, y: 0 };
	return { x: a.x / len, y: a.y / len };
}
