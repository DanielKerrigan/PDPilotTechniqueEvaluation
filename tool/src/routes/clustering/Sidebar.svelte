<script lang="ts">
	import type { Feature } from '$lib/types';
	import Download from './Download.svelte';

	export let features: Feature[];
	export let index: number;
	export let setFeatureIndex: (i: number) => void;

	let div: HTMLDivElement;

	function scrollTo(index: number) {
		if (div) {
			div.scrollTo(0, index * 32);
		}
	}

	$: scrollTo(index);

	$: numComplete = features.filter((d) => d.scores.every(({ score }) => score !== -1)).length;
</script>

<div class="container">
	<Download {features} />
	<div class="progress">
		<div>Progress: {numComplete}/{features.length}</div>
	</div>
	<div class="features" bind:this={div}>
		{#each features as feature, i}
			<button class="row" class:highlight={i === index} on:click={() => setFeatureIndex(i)}>
				<div class="index">{i + 1}</div>
				<svg
					xmlns="http://www.w3.org/2000/svg"
					fill="none"
					viewBox="0 0 24 24"
					stroke-width="1.5"
					stroke="black"
					style:width="24px"
					style:height="24px"
				>
					{#if feature.scores.every((d) => d.score !== -1)}
						<path stroke-linecap="round" stroke-linejoin="round" d="m4.5 12.75 6 6 9-13.5" />
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
		width: 175px;

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

	.features {
		flex: 1;
		display: flex;
		flex-direction: column;
		overflow: auto;
	}

	.progress {
		height: 32px;
		display: flex;
		align-items: center;
		justify-content: center;
	}
</style>
