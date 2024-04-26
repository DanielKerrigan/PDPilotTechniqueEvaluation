<script lang="ts">
	import './style.css';
</script>

<div class="app">
	<slot />
</div>

<style>
	.app {
		width: 100vw;
		height: 100vh;

		font-family: system-ui, sans-serif;
		font-size: 16px;
	}
</style>
