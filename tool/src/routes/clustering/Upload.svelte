<script lang="ts">
	import type { Feature, Method } from '$lib/types';
	import { shuffle } from 'd3-array';
	import { createEventDispatcher } from 'svelte';

	const dispatch = createEventDispatcher<{ upload: Feature[] }>();

	let fileInput: HTMLInputElement;

	function onchange(event: Event & { currentTarget: EventTarget & HTMLInputElement }) {
		const files = event.currentTarget?.files;

		if (files === null || files.length === 0) {
			return;
		}

		const file: File = files[0];

		const reader: FileReader = new FileReader();

		reader.onload = function (event) {
			if (event.target === null) {
				return;
			}
			const text = event.target.result as string;
			const json = JSON.parse(text) as Omit<Feature, 'scores'>[];

			const features: Feature[] = json.map((d) => {
				const order = shuffle(Object.keys(d.clusters) as Method[]);
				return {
					...d,
					scores: order.map((method) => ({ method, score: -1 }))
				};
			});

			dispatch('upload', features);
		};

		reader.readAsText(file);
	}

	function onclick() {
		if (fileInput) {
			fileInput.click();
		}
	}
</script>

<div class="container">
	<div>
		<input
			bind:this={fileInput}
			type="file"
			accept=".json"
			style="display:none"
			on:change={onchange}
		/>
		<button on:click={onclick} class="small">Upload Data</button>
	</div>
</div>

<style>
	.container {
		width: 100%;
		height: 100%;
		display: flex;
		align-items: center;
		justify-content: center;
	}

	button {
		padding: 0.5em 1em;
		font-size: 32px;
	}
</style>
