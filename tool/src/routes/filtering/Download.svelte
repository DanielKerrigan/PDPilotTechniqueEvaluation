<!--
  References:
  https://stackoverflow.com/questions/34156282/how-do-i-save-json-to-local-text-file
  https://stackoverflow.com/questions/2897619/using-html5-javascript-to-generate-and-save-a-file
  https://stackoverflow.com/questions/19721439/download-json-object-as-a-file-from-browser
-->

<script lang="ts">
	import type { Curve } from '$lib/types';

	export let curves: Curve[];

	let download: HTMLAnchorElement;

	function onclick() {
		if (!download) {
			return;
		}
		const prefix: string = 'data:text/plain;charset=utf-8,';
		const url: string = prefix + encodeURIComponent(JSON.stringify(curves));
		download.href = url;
		download.click();
	}
</script>

<button on:click={onclick}>Download</button>

<a href="." download={'shapes.json'} bind:this={download}>Download</a>

<style>
	a {
		display: none;
	}

	button {
		height: 2em;
		color: white;
		background-color: var(--gray-7);
		border: 1px solid var(--gray-7);
	}

	button:disabled {
		cursor: not-allowed;
		background-color: var(--gray-3);
		border-color: var(--gray-3);
	}
</style>
