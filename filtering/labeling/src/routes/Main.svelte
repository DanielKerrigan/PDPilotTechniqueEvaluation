<script lang="ts">
	import type { ScaleOrdinal } from 'd3-scale';
	import type { Curve, Shape } from './types';
	import Path from './Path.svelte';
	import SegmentedButton from './SegmentedButton.svelte';

	export let curve: Curve;
	export let color: ScaleOrdinal<Shape, string>;
	export let next: () => void;
	export let previous: () => void;

	function onkeydown(ev: KeyboardEvent) {
		if (ev.key === 'ArrowLeft') {
			previous();
		} else if (ev.key === 'ArrowRight') {
			next();
		}
	}

	let borderBoxSize: ResizeObserverSize[] | undefined | null;
	$: width = borderBoxSize ? borderBoxSize[0].inlineSize : 100;
	$: height = borderBoxSize ? borderBoxSize[0].blockSize : 100;
</script>

<svelte:window on:keydown={onkeydown} />

<div class="container">
	<div class="controls">
		<button on:click={previous}>previous</button>
		<button on:click={next}>next</button>
	</div>
	<div class="plot" bind:borderBoxSize>
		<Path {width} {height} {curve} {color} />
	</div>
	<div class="choice">
		<SegmentedButton selectedValue={curve.shape} {color} on:setShape />
	</div>
</div>

<style>
	.container {
		width: 100%;
		height: 100%;
		padding: 4em;
		display: flex;
		flex-direction: column;
	}

	.controls {
		display: flex;
		justify-content: space-between;
		flex: none;
	}

	.plot {
		flex: 1;
	}

	.choice {
		flex: none;
		display: flex;
		justify-content: center;
	}

	button {
		padding: 0.5em 1em;
		font-size: 32px;
	}
</style>
