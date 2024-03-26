<script lang="ts">
	import type { ScaleOrdinal } from 'd3-scale';
	import Path from './Path.svelte';
	import type { Curve, Shape } from '$lib/types';
	import Download from './Download.svelte';

	export let curves: Curve[];
	export let index: number;
	export let color: ScaleOrdinal<Shape, string>;
	export let aspectRatio: number;
	export let setIndex: (i: number) => void;

	let div: HTMLDivElement;

	function scrollTo(index: number) {
		if (div) {
			div.scrollTo(0, index * 32);
		}
	}

	$: scrollTo(index);
</script>

<div class="container">
	<Download {curves} />
	<div class="curves" bind:this={div}>
		{#each curves as curve, i}
			<button class="row" class:highlight={i === index} on:click={() => setIndex(i)}>
				<div class="index">{i + 1}</div>
				<Path
					{curve}
					{color}
					strokeWidth={1.5}
					circleRadius={1.5}
					width={24 * aspectRatio}
					height={24}
				/>
				<svg
					xmlns="http://www.w3.org/2000/svg"
					fill="none"
					viewBox="0 0 24 24"
					stroke-width="1.5"
					stroke={color(curve.shape)}
					style:width="24px"
					style:height="24px"
				>
					{#if curve.shape === 'increasing'}
						<path
							stroke-linecap="round"
							stroke-linejoin="round"
							d="M4.5 10.5 12 3m0 0 7.5 7.5M12 3v18"
						/>
					{:else if curve.shape === 'decreasing'}
						<path
							stroke-linecap="round"
							stroke-linejoin="round"
							d="M19.5 13.5 12 21m0 0-7.5-7.5M12 21V3"
						/>
					{:else if curve.shape === 'mixed'}
						<path
							stroke-linecap="round"
							stroke-linejoin="round"
							d="M3 7.5 7.5 3m0 0L12 7.5M7.5 3v13.5m13.5 0L16.5 21m0 0L12 16.5m4.5 4.5V7.5"
						/>
					{/if}
				</svg>
				<svg
					xmlns="http://www.w3.org/2000/svg"
					fill="none"
					viewBox="0 0 24 24"
					stroke-width="1.5"
					stroke={color(curve.shape)}
					style:width="24px"
					style:height="24px"
				>
					{#if curve.unclear}
						<path
							stroke-linecap="round"
							stroke-linejoin="round"
							d="M9.879 7.519c1.171-1.025 3.071-1.025 4.242 0 1.172 1.025 1.172 2.687 0 3.712-.203.179-.43.326-.67.442-.745.361-1.45.999-1.45 1.827v.75M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9 5.25h.008v.008H12v-.008Z"
						/>
					{/if}
				</svg>
			</button>
		{/each}
	</div>
</div>

<style>
	.container {
		height: 100%;
		flex: none;
		border-right: 1px solid var(--gray-1);

		display: flex;
		flex-direction: column;
	}

	.row {
		padding: 0.25em;
		display: flex;
		justify-content: space-between;
		align-items: center;
	}

	.row:hover {
		background-color: var(--gray-1);
	}

	.highlight {
		background-color: var(--gray-1);
	}

	.index {
		width: 48px;
	}

	button {
		border: none;
	}

	.curves {
		flex: 1;
		display: flex;
		flex-direction: column;
		overflow: auto;
	}
</style>
