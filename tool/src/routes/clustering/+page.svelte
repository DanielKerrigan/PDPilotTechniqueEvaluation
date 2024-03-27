<script lang="ts">
	import './style.css';
	import Sidebar from './Sidebar.svelte';
	import Main from './Main.svelte';
	import Upload from './Upload.svelte';
	import type { Feature } from '$lib/types';

	let index = 0;
	let features: Feature[] = [];

	function onUpload(event: CustomEvent<Feature[]>) {
		features = event.detail;
	}

	function setFeatureIndex(i: number) {
		if (i >= 0 && i < features.length) {
			index = i;
		}
	}

	function setScore(event: CustomEvent<{ index: number; score: number }>) {
		console.log('setScore', event.detail.index, event.detail.score);
		features[index].scores[event.detail.index].score = event.detail.score;
	}
</script>

<div class="app">
	{#if features.length > 0}
		<Sidebar {features} {index} {setFeatureIndex} />
		<Main
			feature={features[index]}
			next={() => setFeatureIndex(index + 1)}
			previous={() => setFeatureIndex(index - 1)}
			on:setScore={setScore}
		/>
	{:else}
		<Upload on:upload={onUpload} />
	{/if}
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
