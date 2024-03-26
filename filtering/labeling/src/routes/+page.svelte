<script lang="ts">
	import './style.css';
	import Sidebar from './Sidebar.svelte';
	import Main from './Main.svelte';
	import type { Curve, Shape } from '$lib/types';
	import { scaleOrdinal } from 'd3-scale';

	export let data: { curves: Curve[] };

	let index = 0;
	let curves = data.curves;

	const color = scaleOrdinal<Shape, string>()
		.domain(['decreasing', 'mixed', 'increasing', ''])
		.range(['#d95f02', '#7570b3', '#1b9e77', '#666666']);

	const aspectRatio = 1.95;

	function setIndex(i: number) {
		if (i >= 0 && i < curves.length) {
			index = i;
		}
	}

	function setShape(event: CustomEvent<Shape>) {
		curves[index].shape = event.detail;
	}

	function setUnclear(event: CustomEvent<boolean>) {
		curves[index].unclear = event.detail;
	}
</script>

<div class="app">
	<Sidebar {curves} {index} {setIndex} {color} {aspectRatio} />
	<Main
		curve={curves[index]}
		{color}
		{aspectRatio}
		next={() => setIndex(index + 1)}
		previous={() => setIndex(index - 1)}
		on:setShape={setShape}
		on:setUnclear={setUnclear}
	/>
</div>

<style>
	.app {
		width: 100vw;
		height: 100vh;

		font-family: system-ui, sans-serif;
		font-size: 16px;

		display: flex;
	}
</style>
