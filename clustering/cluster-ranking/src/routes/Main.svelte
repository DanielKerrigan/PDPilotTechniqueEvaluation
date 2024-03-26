<script lang="ts">
	import type { Feature } from '../lib/types';
	import SegmentedButton from './SegmentedButton.svelte';
	import SmallMultiplesClusters from './SmallMultiplesClusters.svelte';
	import ClustersOnePlot from './ClustersOnePlot.svelte';
	import { format } from 'd3-format';
	import type { AnonymousLabel } from '../lib/types';
	import Unclustered from './Unclustered.svelte';

	export let feature: Feature;
	export let next: () => void;
	export let previous: () => void;

	const pctFormat = format('.2~%');

	function onkeydown(ev: KeyboardEvent) {
		if (ev.key === 'ArrowLeft') {
			previous();
		} else if (ev.key === 'ArrowRight') {
			next();
		}
	}

	const anonymousLabels: AnonymousLabel[] = [
		'Left much better',
		'Left somewhat better',
		'Neutral',
		'Right somewhat better',
		'Right much better'
	];

	const visualizations = ['Clusters - small multiples', 'Clusters - one plot', 'Unclustered'];
	let selectedVisualizationIndex = 0;

	function onChangeVisualization(event: CustomEvent<number>) {
		selectedVisualizationIndex = event.detail;
	}
</script>

<svelte:window on:keydown={onkeydown} />

<div class="container">
	<div class="segmented-buttons">
		<SegmentedButton
			labels={visualizations}
			selectedIndex={selectedVisualizationIndex}
			on:setSelectedIndex={onChangeVisualization}
		/>
	</div>

	{#if selectedVisualizationIndex !== 2}
		<div class="info">
			{feature.intersection}/{feature.ice.length} instances are clustered the same ({pctFormat(
				feature.percent_overlap
			)}).
		</div>
	{/if}

	<div class="plots">
		{#if selectedVisualizationIndex === 0}
			{#each feature.order as method}
				<SmallMultiplesClusters {feature} {method} />
			{/each}
		{:else if selectedVisualizationIndex === 1}
			{#each feature.order as method}
				<ClustersOnePlot {feature} {method} />
			{/each}
		{:else}
			<Unclustered {feature} />
		{/if}
	</div>

	<div class="segmented-buttons">
		<SegmentedButton
			labels={anonymousLabels}
			selectedIndex={feature.labelIndex}
			on:setSelectedIndex
		/>
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

	.info {
		text-align: center;
	}

	.plots {
		flex: 1;
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 4em;
	}

	.segmented-buttons {
		flex: none;
		display: flex;
		justify-content: center;
	}
</style>
