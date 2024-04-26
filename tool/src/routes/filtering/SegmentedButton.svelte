<script lang="ts">
	import type { ScaleOrdinal } from 'd3-scale';
	import type { Shape } from '$lib/types';
	import { createEventDispatcher } from 'svelte';

	export let selectedValue: string;
	export let color: ScaleOrdinal<Shape, string>;
	export let enableNumberKeys = false;

	const dispatch = createEventDispatcher<{ setShape: Shape }>();

	const segments: { value: Shape; label: string }[] = [
		{
			value: 'decreasing',
			label: 'Decreasing'
		},
		{ value: 'mixed', label: 'Mixed' },
		{ value: 'increasing', label: 'Increasing' }
	];

	function onChangeValue(value: Shape) {
		dispatch('setShape', value);
	}

	function onkeydown(ev: KeyboardEvent) {
		if (!enableNumberKeys) {
			return;
		}

		if (ev.key === '1' || ev.key === '2' || ev.key === '3') {
			dispatch('setShape', segments[+ev.key - 1].value);
		}
	}
</script>

<svelte:window on:keydown={onkeydown} />

<div class="segmented-control-container">
	<div class="segmented-control-buttons-row">
		{#each segments as { value, label }}
			<button
				class:selected-segment={value === selectedValue}
				on:click={() => onChangeValue(value)}
				style:color={value === selectedValue ? 'white' : 'black'}
				style:background={value === selectedValue ? color(value) : 'white'}>{label}</button
			>
		{/each}
	</div>
</div>

<style>
	.segmented-control-container {
		display: flex;
		gap: 0.25em;
		align-items: center;
	}

	.segmented-control-buttons-row {
		display: inline-grid;
		grid-auto-columns: 1fr;
		grid-auto-flow: column;
	}

	.segmented-control-buttons-row button {
		border-left: none;
		padding: 0.5em 1em;
		font-size: 32px;
	}

	.segmented-control-buttons-row button:first-child {
		border-left: 1px solid var(--gray-7);
	}

	.segmented-control-buttons-row button.selected-segment {
		/* color: white; */
		background-color: var(--gray-7);
	}

	.segmented-control-buttons-row button.selected-segment:active {
		/* color: white; */
		background-color: var(--gray-9);
		border-color: var(--gray-9);
	}
</style>
