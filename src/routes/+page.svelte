<script lang="ts">
	import { onMount, tick } from 'svelte';
	import { KnotEngine, type KnotMaterialPreset, type KnotMetrics } from '$lib/knot';

	let viewport: HTMLDivElement | undefined;
	let engine = $state<KnotEngine | null>(null);

	let crossingTarget = $state(50);
	let seed = $state(newSeed());
	let autoRelax = $state(true);
	let solidLink = $state(true);
	let colorizeLinks = $state(true);
	let showControlPoints = $state(false);
	let arcGuideLayout = $state(true);
	let repulsionStrength = $state(1.0);
	let smoothness = $state(0.62);
	let relaxSpeed = $state(3);
	let lengthTargetScale = $state(0.88);
	let materialPreset = $state<KnotMaterialPreset>('liquid_metal');
	let materialRoughness = $state(0.18);
	let materialMetalness = $state(1);
	let materialTransmission = $state(0);
	let materialClearcoat = $state(0.88);
	let materialEnvironment = $state(1.6);
	let materialEmissive = $state(0.8);
	let materialAnimationSpeed = $state(1.25);
	let materialTextureScale = $state(1.65);
	let renderExposure = $state(0.62);
	let ambientLightIntensity = $state(0.12);
	let keyLightIntensity = $state(1.3);
	let fillLightIntensity = $state(0.06);
	let rimLightIntensity = $state(0.42);
	let generating = $state(false);
	let status = $state('Preparing WebGPU renderer...');
	let bootError = $state<string | null>(null);
	let metrics = $state<KnotMetrics>({
		preset: 'random_shadow',
		label: 'Shadow',
		crossings: 0,
		targetCrossings: 50,
		energy: 0,
		nodeCount: 0,
		minClearance: Number.POSITIVE_INFINITY,
		clearanceRatio: Number.POSITIVE_INFINITY,
		maxLengthDrift: 0,
		edgeUniformity: 1,
		acceptedSteps: 0,
		rejectedSteps: 0
	});

	function relaxStep(): void {
		engine?.stepRelax(16);
	}

	function showSolution(): void {
		engine?.showSolution();
	}

	function applyMaterialPresetDefaults(preset: KnotMaterialPreset): void {
		switch (preset) {
			case 'rope': {
				materialRoughness = 0.28;
				materialMetalness = 0.08;
				materialTransmission = 0;
				materialClearcoat = 0.55;
				materialEnvironment = 0.72;
				materialEmissive = 1;
				materialAnimationSpeed = 0.8;
				materialTextureScale = 1.45;
				break;
			}
			case 'glass': {
				materialRoughness = 0.08;
				materialMetalness = 0;
				materialTransmission = 1;
				materialClearcoat = 0.98;
				materialEnvironment = 1.2;
				materialEmissive = 0.9;
				materialAnimationSpeed = 0.35;
				materialTextureScale = 1.7;
				break;
			}
			case 'liquid_metal': {
				materialRoughness = 0.18;
				materialMetalness = 1;
				materialTransmission = 0;
				materialClearcoat = 0.88;
				materialEnvironment = 1.6;
				materialEmissive = 0.8;
				materialAnimationSpeed = 1.25;
				materialTextureScale = 1.65;
				break;
			}
			case 'energy_field': {
				materialRoughness = 0.3;
				materialMetalness = 0.22;
				materialTransmission = 0.55;
				materialClearcoat = 0.74;
				materialEnvironment = 0.86;
				materialEmissive = 1.9;
				materialAnimationSpeed = 1.8;
				materialTextureScale = 1.95;
				break;
			}
			case 'vector_field': {
				materialRoughness = 0.34;
				materialMetalness = 0.26;
				materialTransmission = 0.2;
				materialClearcoat = 0.72;
				materialEnvironment = 0.82;
				materialEmissive = 1.5;
				materialAnimationSpeed = 1.45;
				materialTextureScale = 2.2;
				break;
			}
		}
	}

	async function generateChallenge(randomizeSeed: boolean): Promise<void> {
		if (!engine || generating) return;
		if (randomizeSeed) seed = newSeed();
		const safeCrossings = clampInteger(crossingTarget, 3, 128);
		crossingTarget = safeCrossings;
		generating = true;
		bootError = null;
		status = `Generating ${safeCrossings}-crossing challenge...`;
		await tick();
		await waitFrame();

		try {
			engine.setCrossingChallenge(safeCrossings, seed);
		} catch (error) {
			bootError = error instanceof Error ? error.message : 'Failed to generate challenge.';
		} finally {
			generating = false;
		}
	}

	$effect(() => {
		if (engine) engine.setAutoRelax(autoRelax);
	});

	$effect(() => {
		if (engine) engine.setShowControlPoints(showControlPoints);
	});

	$effect(() => {
		if (engine) engine.setSolidLink(solidLink);
	});

	$effect(() => {
		if (engine) engine.setColorizeLinks(colorizeLinks);
	});

	$effect(() => {
		if (engine) engine.setArcGuideLayout(arcGuideLayout);
	});

	$effect(() => {
		if (engine) engine.setRepulsionStrength(repulsionStrength);
	});

	$effect(() => {
		if (engine) engine.setRelaxSmoothness(smoothness);
	});

	$effect(() => {
		if (engine) engine.setRelaxIterations(relaxSpeed);
	});

	$effect(() => {
		if (engine) engine.setLengthTargetScale(lengthTargetScale);
	});

	$effect(() => {
		if (!engine) return;
		engine.setMaterialSettings({
			preset: materialPreset,
			roughness: materialRoughness,
			metalness: materialMetalness,
			transmission: materialTransmission,
			clearcoat: materialClearcoat,
			envMapIntensity: materialEnvironment,
			emissiveIntensity: materialEmissive,
			animationSpeed: materialAnimationSpeed,
			textureScale: materialTextureScale
		});
	});

	$effect(() => {
		if (!engine) return;
		engine.setLightingSettings({
			exposure: renderExposure,
			ambientIntensity: ambientLightIntensity,
			keyIntensity: keyLightIntensity,
			fillIntensity: fillLightIntensity,
			rimIntensity: rimLightIntensity
		});
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
				engine.setSolidLink(solidLink);
				engine.setColorizeLinks(colorizeLinks);
				engine.setShowControlPoints(showControlPoints);
				engine.setArcGuideLayout(arcGuideLayout);
					engine.setRepulsionStrength(repulsionStrength);
					engine.setRelaxSmoothness(smoothness);
					engine.setRelaxIterations(relaxSpeed);
					engine.setLengthTargetScale(lengthTargetScale);
					engine.setMaterialSettings({
						preset: materialPreset,
						roughness: materialRoughness,
						metalness: materialMetalness,
						transmission: materialTransmission,
						clearcoat: materialClearcoat,
						envMapIntensity: materialEnvironment,
						emissiveIntensity: materialEmissive,
						animationSpeed: materialAnimationSpeed,
						textureScale: materialTextureScale
					});
					engine.setLightingSettings({
						exposure: renderExposure,
						ambientIntensity: ambientLightIntensity,
						keyIntensity: keyLightIntensity,
						fillIntensity: fillLightIntensity,
						rimIntensity: rimLightIntensity
					});
					await generateChallenge(false);
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

	function waitFrame(): Promise<void> {
		return new Promise((resolve) => requestAnimationFrame(() => resolve()));
	}

	function newSeed(): number {
		return (Math.random() * 0xffffffff) >>> 0;
	}

	function clampInteger(value: number, min: number, max: number): number {
		return Math.min(max, Math.max(min, Math.round(value)));
	}
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
				Target crossings
				<input type="number" min="3" max="128" step="1" bind:value={crossingTarget} />
			</label>

			<label>
				Seed
				<input type="number" min="1" max="4294967295" step="1" bind:value={seed} />
			</label>

			<div class="toggles">
				<label><input type="checkbox" bind:checked={autoRelax} /> Auto relax</label>
				<label><input type="checkbox" bind:checked={solidLink} /> Solid link</label>
				<label><input type="checkbox" bind:checked={colorizeLinks} /> Colorize links</label>
				<label><input type="checkbox" bind:checked={showControlPoints} /> Show control points</label>
				<label><input type="checkbox" bind:checked={arcGuideLayout} /> Arc-guide layout</label>
			</div>

			<label>
				Repulsion force: {repulsionStrength.toFixed(2)}
				<input type="range" min="0.2" max="3.5" step="0.05" bind:value={repulsionStrength} />
			</label>

			<label>
				Smoothness: {smoothness.toFixed(2)}
				<input type="range" min="0" max="1" step="0.02" bind:value={smoothness} />
			</label>

			<label>
				Relax speed: {relaxSpeed}
				<input type="range" min="1" max="8" step="1" bind:value={relaxSpeed} />
			</label>

				<label>
					Tension (target length): {lengthTargetScale.toFixed(2)}
					<input type="range" min="0.6" max="1.05" step="0.01" bind:value={lengthTargetScale} />
				</label>

				<section class="material-lab">
					<h3>Material Lab</h3>
						<label>
							Material preset
							<select
								bind:value={materialPreset}
								onchange={() => applyMaterialPresetDefaults(materialPreset)}
							>
								<option value="rope">Classic rope</option>
								<option value="glass">Glass</option>
								<option value="liquid_metal">Shiny metal</option>
							<option value="energy_field">Energy field</option>
							<option value="vector_field">Vector field</option>
						</select>
					</label>

					<label>
						Roughness: {materialRoughness.toFixed(2)}
						<input type="range" min="0.02" max="1" step="0.01" bind:value={materialRoughness} />
					</label>

					<label>
						Metalness: {materialMetalness.toFixed(2)}
						<input type="range" min="0" max="1" step="0.01" bind:value={materialMetalness} />
					</label>

					<label>
						Transmission: {materialTransmission.toFixed(2)}
						<input type="range" min="0" max="1" step="0.01" bind:value={materialTransmission} />
					</label>

					<label>
						Clearcoat: {materialClearcoat.toFixed(2)}
						<input type="range" min="0" max="1" step="0.01" bind:value={materialClearcoat} />
					</label>

					<label>
						Environment reflections: {materialEnvironment.toFixed(2)}
						<input type="range" min="0" max="3" step="0.05" bind:value={materialEnvironment} />
					</label>

					<label>
						Emissive boost: {materialEmissive.toFixed(2)}
						<input type="range" min="0" max="4" step="0.05" bind:value={materialEmissive} />
					</label>

					<label>
						Flow speed: {materialAnimationSpeed.toFixed(2)}
						<input type="range" min="0" max="6" step="0.05" bind:value={materialAnimationSpeed} />
					</label>

					<label>
						Texture scale: {materialTextureScale.toFixed(2)}
						<input type="range" min="0.2" max="8" step="0.05" bind:value={materialTextureScale} />
					</label>

					<label>
						Exposure: {renderExposure.toFixed(2)}
						<input type="range" min="0.5" max="2.7" step="0.05" bind:value={renderExposure} />
					</label>

					<label>
						Ambient light: {ambientLightIntensity.toFixed(2)}
						<input type="range" min="0" max="2.4" step="0.05" bind:value={ambientLightIntensity} />
					</label>

					<label>
						Key light: {keyLightIntensity.toFixed(2)}
						<input type="range" min="0" max="4" step="0.05" bind:value={keyLightIntensity} />
					</label>

					<label>
						Fill light: {fillLightIntensity.toFixed(2)}
						<input type="range" min="0" max="3" step="0.05" bind:value={fillLightIntensity} />
					</label>

					<label>
						Rim light: {rimLightIntensity.toFixed(2)}
						<input type="range" min="0" max="3" step="0.05" bind:value={rimLightIntensity} />
					</label>
				</section>

				<div class="actions">
					<button onclick={() => generateChallenge(false)} disabled={!engine || generating}>
						{generating ? 'Generating...' : 'Generate'}
				</button>
				<button onclick={() => generateChallenge(true)} disabled={!engine || generating}>
					New seed + generate
				</button>
				<button onclick={relaxStep} disabled={!engine}>Relax step</button>
				<button onclick={showSolution} disabled={!engine}>Show solution</button>
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
				<div>
					<span>Min clearance / (2r)</span>
					<strong>{Number.isFinite(metrics.clearanceRatio) ? metrics.clearanceRatio.toFixed(2) : 'n/a'}</strong>
				</div>
				<div>
					<span>Max length drift</span>
					<strong>{(metrics.maxLengthDrift * 100).toFixed(2)}%</strong>
				</div>
				<div>
					<span>Edge uniformity</span>
					<strong>{metrics.edgeUniformity.toFixed(2)}x</strong>
				</div>
				<div>
					<span>Line search</span>
					<strong>{metrics.acceptedSteps}/{metrics.rejectedSteps}</strong>
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
		height: 100svh;
		overflow: hidden;
		font-family: 'Trebuchet MS', 'Segoe UI', sans-serif;
		color: #f3f7ec;
		background:
			radial-gradient(circle at 8% 12%, rgba(49, 186, 155, 0.24), transparent 44%),
			radial-gradient(circle at 86% 82%, rgba(249, 150, 64, 0.22), transparent 38%),
			linear-gradient(145deg, #03110f, #041e1b 58%, #0a2d28);
	}

	.page {
		height: 100svh;
		padding: 1rem;
		display: grid;
		grid-template-columns: minmax(18rem, 23rem) 1fr;
		gap: 1rem;
		box-sizing: border-box;
		overflow: hidden;
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
		min-height: 0;
		overflow: auto;
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

	input[type='number'],
	input[type='range'],
	select,
	button {
		font-family: inherit;
	}

	input[type='number'],
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

	.material-lab {
		display: grid;
		gap: 0.62rem;
		padding: 0.72rem;
		border-radius: 0.72rem;
		background: rgba(7, 27, 24, 0.62);
		border: 1px solid rgba(118, 239, 214, 0.18);
	}

	.material-lab h3 {
		margin: 0;
		font-size: 0.92rem;
		letter-spacing: 0.01em;
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
		height: 100%;
		min-height: 0;
	}

	.viewport {
		width: 100%;
		height: 100%;
		min-height: 0;
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
		:global(body) {
			height: auto;
			overflow: auto;
		}

		.page {
			height: auto;
			min-height: 100svh;
			grid-template-columns: 1fr;
			padding: 0.75rem;
			overflow: visible;
		}

		.panel {
			overflow: visible;
		}

		.viewport-shell,
		.viewport {
			min-height: 24rem;
			height: 24rem;
		}
	}
</style>
