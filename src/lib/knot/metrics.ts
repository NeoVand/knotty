const VIEW_ANGLES: Array<[number, number]> = [
	[0.2, 0.1],
	[0.6, 0.4],
	[1.1, 0.7],
	[1.7, 0.5],
	[2.2, 1.0],
	[2.8, 0.8],
	[3.4, 0.35],
	[4.1, 1.2]
];

export function estimateCrossings(points: Float32Array): number {
	const count = points.length / 3;
	const projected = new Float32Array(count * 2);
	let best = Number.POSITIVE_INFINITY;

	for (const [yaw, pitch] of VIEW_ANGLES) {
		projectPoints(points, projected, yaw, pitch);
		const candidate = countCrossings(projected);
		if (candidate < best) best = candidate;
	}

	return Number.isFinite(best) ? best : 0;
}

function projectPoints(
	source: Float32Array,
	target: Float32Array,
	yaw: number,
	pitch: number
): void {
	const cy = Math.cos(yaw);
	const sy = Math.sin(yaw);
	const cx = Math.cos(pitch);
	const sx = Math.sin(pitch);
	const count = source.length / 3;

	for (let i = 0; i < count; i += 1) {
		const idx = i * 3;
		const x = source[idx];
		const y = source[idx + 1];
		const z = source[idx + 2];
		const x1 = cy * x + sy * z;
		const z1 = -sy * x + cy * z;
		const y1 = cx * y - sx * z1;
		const out = i * 2;
		target[out] = x1;
		target[out + 1] = y1;
	}
}

function countCrossings(projected: Float32Array): number {
	const count = projected.length / 2;
	let intersections = 0;

	for (let i = 0; i < count; i += 1) {
		const iNext = (i + 1) % count;
		const i0 = i * 2;
		const i1 = iNext * 2;
		for (let j = i + 1; j < count; j += 1) {
			const jNext = (j + 1) % count;
			if (areAdjacentEdges(i, j, count)) continue;
			const j0 = j * 2;
			const j1 = jNext * 2;
			if (
				segmentsIntersect(
					projected[i0],
					projected[i0 + 1],
					projected[i1],
					projected[i1 + 1],
					projected[j0],
					projected[j0 + 1],
					projected[j1],
					projected[j1 + 1]
				)
			) {
				intersections += 1;
			}
		}
	}

	return intersections;
}

function segmentsIntersect(
	ax: number,
	ay: number,
	bx: number,
	by: number,
	cx: number,
	cy: number,
	dx: number,
	dy: number
): boolean {
	const eps = 1e-6;
	const o1 = orientation(ax, ay, bx, by, cx, cy);
	const o2 = orientation(ax, ay, bx, by, dx, dy);
	const o3 = orientation(cx, cy, dx, dy, ax, ay);
	const o4 = orientation(cx, cy, dx, dy, bx, by);

	if (Math.abs(o1) < eps || Math.abs(o2) < eps || Math.abs(o3) < eps || Math.abs(o4) < eps) return false;
	return (o1 > 0) !== (o2 > 0) && (o3 > 0) !== (o4 > 0);
}

function orientation(ax: number, ay: number, bx: number, by: number, cx: number, cy: number): number {
	return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

function areAdjacentEdges(a: number, b: number, count: number): boolean {
	if (a === b) return true;
	const aNext = (a + 1) % count;
	const bNext = (b + 1) % count;
	return a === bNext || b === aNext;
}
