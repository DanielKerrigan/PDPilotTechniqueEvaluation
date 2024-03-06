<script lang="ts">
	import './style.css';
	import Sidebar from './Sidebar.svelte';
	import Main from './Main.svelte';
	import type { Curve, Shape } from './types';
	import { scaleOrdinal } from 'd3-scale';

	export let data: { curves: Curve[] };

	let index = 0;
	let curves = data.curves;

	const color = scaleOrdinal<Shape, string>()
		.domain(['decreasing', 'mixed', 'increasing', ''])
		.range(['#d95f02', '#7570b3', '#1b9e77', '#666666']);

	function setIndex(i: number) {
		if (i >= 0 && i < curves.length) {
			index = i;
		}
	}

	function setShape(event: CustomEvent<Shape>) {
		curves[index].shape = event.detail;
	}
</script>

<div class="app">
	<Sidebar {curves} {index} {setIndex} {color} />
	<Main
		curve={curves[index]}
		{color}
		next={() => setIndex(index + 1)}
		previous={() => setIndex(index - 1)}
		on:setShape={setShape}
	/>
</div>

<style>
	.app {
		width: 100vw;
		height: 100vh;

		font-family: system-ui, sans-serif;
		font-size: 16px;

		display: flex;

		--blue: rgb(0, 95, 204);
		--dark-blue: rgb(1, 51, 104);

		--gray-0: rgb(247, 247, 247);
		--gray-1: rgb(226, 226, 226);
		--gray-2: rgb(198, 198, 198);
		--gray-3: rgb(171, 171, 171);
		--gray-4: rgb(145, 145, 145);
		--gray-5: rgb(119, 119, 119);
		--gray-6: rgb(94, 94, 94);
		--gray-7: rgb(71, 71, 71);
		--gray-8: rgb(48, 48, 48);
		--gray-9: rgb(27, 27, 27);
	}
</style>
