<script lang="ts">
	import { createEventDispatcher } from 'svelte';

	export let selectedIndex: number;
	export let labels: string[];

	const dispatch = createEventDispatcher<{ setSelectedIndex: number }>();

	function onChangeValue(value: number) {
		dispatch('setSelectedIndex', value);
	}

	function onkeydown(ev: KeyboardEvent) {
		if (ev.key >= '1' && ev.key <= '5') {
			dispatch('setSelectedIndex', +ev.key - 1);
		}
	}
</script>

<svelte:window on:keydown={onkeydown} />

<div class="segmented-control-container">
	<div class="segmented-control-buttons-row">
		{#each labels as label, i}
			<button class:selected-segment={i === selectedIndex} on:click={() => onChangeValue(i)}
				>{label}</button
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
	}

	.segmented-control-buttons-row button:first-child {
		border-left: 1px solid var(--gray-7);
	}

	.segmented-control-buttons-row button.selected-segment {
		color: white;
		background-color: var(--gray-7);
	}

	.segmented-control-buttons-row button.selected-segment:active {
		color: white;
		background-color: var(--gray-9);
		border-color: var(--gray-9);
	}
</style>
