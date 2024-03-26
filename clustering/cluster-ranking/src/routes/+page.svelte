<script lang="ts">
	import './style.css';
	import Sidebar from './Sidebar.svelte';
	import Main from './Main.svelte';
	import type { Feature } from '../lib/types';

	export let data: { features: Feature[] };

	let index = 0;
	let features = data.features;

	function setFeatureIndex(i: number) {
		if (i >= 0 && i < features.length) {
			index = i;
		}
	}

	function setLabelIndex(event: CustomEvent<number>) {
		features[index].labelIndex = event.detail;
		features[index].label = features[index].labels[event.detail];
	}
</script>

<div class="app">
	<Sidebar {features} {index} {setFeatureIndex} />
	<Main
		feature={features[index]}
		next={() => setFeatureIndex(index + 1)}
		previous={() => setFeatureIndex(index - 1)}
		on:setSelectedIndex={setLabelIndex}
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
