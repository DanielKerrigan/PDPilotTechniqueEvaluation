<script lang="ts">
	import type { ScaleOrdinal } from 'd3-scale';
	import type { Curve, Shape } from '$lib/types';
	import Path from './Path.svelte';
	import SegmentedButton from './SegmentedButton.svelte';
	import { createEventDispatcher } from 'svelte';

	export let curve: Curve;
	export let color: ScaleOrdinal<Shape, string>;
	export let aspectRatio: number;
	export let next: () => void;
	export let previous: () => void;

	const dispatch = createEventDispatcher<{ setUnclear: boolean }>();

	function onkeydown(ev: KeyboardEvent) {
		if (ev.key === 'ArrowLeft') {
			previous();
		} else if (ev.key === 'ArrowRight') {
			next();
		} else if (ev.key === 'u') {
			onChangeUnclear();
		}
	}

	function onChangeUnclear() {
		dispatch('setUnclear', !curve.unclear);
	}

	let borderBoxSize: ResizeObserverSize[] | undefined | null;

	$: divWidth = borderBoxSize ? borderBoxSize[0].inlineSize : 100;
	$: divHeight = borderBoxSize ? borderBoxSize[0].blockSize : 100;

	// https://stackoverflow.com/a/1373879

	$: scaleFactor = Math.min(divWidth / aspectRatio, divHeight);

	$: plotWidth = aspectRatio * scaleFactor;
	$: plotHeight = scaleFactor;
</script>

<svelte:window on:keydown={onkeydown} />

<div class="container">
	<div class="controls">
		<button on:click={previous}>Previous</button>
		<button on:click={next}>Next</button>
	</div>
	<div class="plot" bind:borderBoxSize>
		<Path width={plotWidth} height={plotHeight} {curve} {color} />
	</div>
	<div class="unclear">
		<label>
			<input type="checkbox" checked={curve.unclear} on:change={onChangeUnclear} />
			<div>Unclear</div>
		</label>
	</div>
	<div class="choice">
		<SegmentedButton selectedValue={curve.shape} {color} enableNumberKeys={true} on:setShape />
	</div>
</div>

<style>
	.container {
		width: 100%;
		height: 100%;
		padding: 1em;
		display: flex;
		flex-direction: column;
		gap: 1em;
	}

	.controls {
		display: flex;
		justify-content: space-between;
		flex: none;
	}

	.plot {
		flex: 1;
		display: flex;
		align-items: center;
		justify-content: center;
	}

	.unclear,
	.choice {
		flex: none;
		display: flex;
		justify-content: center;
	}

	.unclear label {
		font-size: 32px;
		display: flex;
		gap: 0.25em;
		align-items: center;
	}

	input[type='checkbox'] {
		appearance: none;
		-webkit-appearance: none;
		width: 32px;
		height: 32px;
		background: white;
		border-radius: 5px;
		border: 2px solid black;
	}

	input[type='checkbox']:checked {
		background: black;
	}

	button {
		padding: 0.5em 1em;
		font-size: 32px;
	}
</style>
