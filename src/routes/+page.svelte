<script lang="ts">
	import { onMount } from 'svelte';
	import {
		KnotEngine,
		PRESET_ORDER,
		type KnotMetrics,
		type KnotPresetName
	} from '$lib/knot';

	const PRESET_LABELS: Record<KnotPresetName, string> = {
		trefoil: 'Trefoil T(2,3)',
		cinquefoil: 'Cinquefoil T(2,5)',
		septfoil: 'Septfoil T(2,7)',
		random_torus: 'Random Torus'
	};

	let viewport: HTMLDivElement | undefined;
	let engine = $state<KnotEngine | null>(null);

	let preset = $state<KnotPresetName>('trefoil');
	let scrambleAmount = $state(0.62);
	let autoRelax = $state(true);
	let showControlPoints = $state(false);
	let status = $state('Preparing WebGPU renderer...');
	let bootError = $state<string | null>(null);
	let metrics = $state<KnotMetrics>({
		preset: 'trefoil',
		label: 'Trefoil T(2,3)',
		crossings: 0,
		targetCrossings: 3,
		energy: 0,
		nodeCount: 0
	});

	function loadPreset(): void {
		engine?.setPreset(preset, scrambleAmount);
	}

	function scramble(): void {
		engine?.scramble(scrambleAmount);
	}

	function relaxStep(): void {
		engine?.stepRelax(16);
	}

	function showSolution(): void {
		engine?.showSolution();
	}

	function randomChallenge(): void {
		preset = 'random_torus';
		engine?.setPreset('random_torus', scrambleAmount);
	}

	$effect(() => {
		if (engine) engine.setAutoRelax(autoRelax);
	});

	$effect(() => {
		if (engine) engine.setShowControlPoints(showControlPoints);
	});

	onMount(() => {
		const container = viewport;
		if (!container) return;
		let cancelled = false;

		const bootstrap = async () => {
			try {
				engine = new KnotEngine({
					container,
					onMetrics: (next) => {
						if (!cancelled) metrics = next;
					},
					onStatus: (next) => {
						if (!cancelled) status = next;
					}
				});
				await engine.init();
				if (cancelled) return;
				engine.setAutoRelax(autoRelax);
				engine.setShowControlPoints(showControlPoints);
				engine.setPreset(preset, scrambleAmount);
				status = 'Drag the knot and reduce crossings toward the target.';
			} catch (error) {
				bootError =
					error instanceof Error
						? error.message
						: 'Unable to initialize WebGPU. Use a browser with WebGPU support.';
			}
		};

		bootstrap();

		return () => {
			cancelled = true;
			engine?.dispose();
			engine = null;
		};
	});
</script>

<main class="page">
	<section class="panel">
		<h1>Knotty Lab</h1>
		<p class="subtitle">
			Build intuition for knot moves: drag a segment, keep the loop self-avoiding, and chase the smallest
			crossing diagram.
		</p>

		<div class="controls">
			<label>
				Challenge knot
				<select bind:value={preset}>
					{#each PRESET_ORDER as option}
						<option value={option}>{PRESET_LABELS[option]}</option>
					{/each}
				</select>
			</label>

			<label>
				Scramble intensity: {scrambleAmount.toFixed(2)}
				<input type="range" min="0" max="1.2" step="0.02" bind:value={scrambleAmount} />
			</label>

			<div class="toggles">
				<label><input type="checkbox" bind:checked={autoRelax} /> Auto relax</label>
				<label><input type="checkbox" bind:checked={showControlPoints} /> Show control points</label>
			</div>

			<div class="actions">
				<button onclick={loadPreset} disabled={!engine}>Load challenge</button>
				<button onclick={scramble} disabled={!engine}>Scramble</button>
				<button onclick={relaxStep} disabled={!engine}>Relax step</button>
				<button onclick={showSolution} disabled={!engine}>Show solution</button>
				<button class="accent" onclick={randomChallenge} disabled={!engine}>Random challenge</button>
			</div>
		</div>

		<div class="stats">
			<div>
				<span>Current crossings</span>
				<strong>{metrics.crossings}</strong>
			</div>
			<div>
				<span>Target crossings</span>
				<strong>{metrics.targetCrossings ?? 'unknown'}</strong>
			</div>
			<div>
				<span>Energy</span>
				<strong>{metrics.energy.toFixed(1)}</strong>
			</div>
			<div>
				<span>Nodes</span>
				<strong>{metrics.nodeCount}</strong>
			</div>
		</div>

		<p class="status">{status}</p>
		{#if bootError}
			<p class="error">{bootError}</p>
		{/if}
	</section>

	<section class="viewport-shell">
		<div class="viewport" bind:this={viewport}></div>
		<div class="overlay">
			<h2>{metrics.label}</h2>
			<p>Drag a segment, then orbit camera to inspect crossings.</p>
		</div>
	</section>
</main>

<style>
	:global(body) {
		margin: 0;
		font-family: 'Trebuchet MS', 'Segoe UI', sans-serif;
		color: #f3f7ec;
		background:
			radial-gradient(circle at 8% 12%, rgba(49, 186, 155, 0.24), transparent 44%),
			radial-gradient(circle at 86% 82%, rgba(249, 150, 64, 0.22), transparent 38%),
			linear-gradient(145deg, #03110f, #041e1b 58%, #0a2d28);
	}

	.page {
		min-height: 100vh;
		padding: 1rem;
		display: grid;
		grid-template-columns: minmax(18rem, 23rem) 1fr;
		gap: 1rem;
		box-sizing: border-box;
	}

	.panel {
		backdrop-filter: blur(12px);
		background: linear-gradient(160deg, rgba(6, 33, 29, 0.86), rgba(5, 18, 16, 0.74));
		border: 1px solid rgba(116, 244, 215, 0.2);
		border-radius: 1rem;
		padding: 1rem;
		display: flex;
		flex-direction: column;
		gap: 1rem;
	}

	h1 {
		margin: 0;
		font-size: clamp(1.6rem, 2vw, 2.05rem);
		letter-spacing: 0.02em;
	}

	.subtitle {
		margin: 0;
		color: rgba(243, 247, 236, 0.82);
		line-height: 1.4;
		font-size: 0.95rem;
	}

	.controls {
		display: flex;
		flex-direction: column;
		gap: 0.8rem;
	}

	label {
		display: grid;
		gap: 0.45rem;
		font-size: 0.88rem;
	}

	select,
	input[type='range'],
	button {
		font-family: inherit;
	}

	select {
		background: rgba(8, 31, 27, 0.88);
		color: #f3f7ec;
		border: 1px solid rgba(117, 239, 211, 0.34);
		border-radius: 0.6rem;
		padding: 0.55rem 0.68rem;
	}

	input[type='range'] {
		accent-color: #70f2d3;
	}

	.toggles {
		display: flex;
		flex-direction: column;
		gap: 0.4rem;
	}

	.toggles label {
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}

	.actions {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 0.45rem;
	}

	button {
		background: rgba(13, 47, 41, 0.87);
		color: #ebfff7;
		border: 1px solid rgba(117, 239, 211, 0.34);
		border-radius: 0.62rem;
		padding: 0.52rem 0.65rem;
		font-size: 0.84rem;
		cursor: pointer;
		transition: background 140ms ease, transform 140ms ease;
	}

	button:hover:enabled {
		background: rgba(24, 74, 65, 0.92);
		transform: translateY(-1px);
	}

	button:disabled {
		cursor: not-allowed;
		opacity: 0.55;
	}

	button.accent {
		background: rgba(126, 67, 19, 0.92);
		border-color: rgba(255, 196, 117, 0.58);
	}

	.stats {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 0.55rem;
	}

	.stats div {
		padding: 0.58rem 0.65rem;
		border-radius: 0.58rem;
		background: rgba(7, 29, 26, 0.68);
		border: 1px solid rgba(118, 239, 214, 0.15);
	}

	.stats span {
		display: block;
		font-size: 0.72rem;
		color: rgba(243, 247, 236, 0.72);
	}

	.stats strong {
		font-family: 'Courier New', monospace;
		font-size: 1.04rem;
	}

	.status {
		margin: 0;
		font-size: 0.84rem;
		color: rgba(243, 247, 236, 0.9);
	}

	.error {
		margin: 0;
		font-size: 0.82rem;
		color: #ffcc9f;
	}

	.viewport-shell {
		position: relative;
		border-radius: 1rem;
		border: 1px solid rgba(116, 244, 215, 0.2);
		overflow: hidden;
		background: linear-gradient(180deg, rgba(6, 19, 17, 0.66), rgba(4, 9, 8, 0.92));
		min-height: 28rem;
	}

	.viewport {
		width: 100%;
		height: 100%;
		min-height: 28rem;
	}

	.overlay {
		position: absolute;
		left: 1rem;
		bottom: 1rem;
		background: rgba(4, 18, 16, 0.72);
		border: 1px solid rgba(116, 244, 215, 0.19);
		border-radius: 0.65rem;
		padding: 0.56rem 0.7rem;
		max-width: 21rem;
	}

	.overlay h2 {
		margin: 0;
		font-size: 1rem;
	}

	.overlay p {
		margin: 0.26rem 0 0;
		font-size: 0.79rem;
		color: rgba(243, 247, 236, 0.75);
	}

	@media (max-width: 980px) {
		.page {
			grid-template-columns: 1fr;
			padding: 0.75rem;
		}

		.viewport-shell,
		.viewport {
			min-height: 24rem;
		}
	}
</style>
