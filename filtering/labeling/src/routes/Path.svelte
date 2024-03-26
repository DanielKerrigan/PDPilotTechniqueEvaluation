<script lang="ts">
	import type { ScaleOrdinal } from 'd3-scale';
	import type { Curve, Shape } from '$lib/types';
	import { scaleLinear, scalePoint } from 'd3-scale';
	import { range } from 'd3-array';
	import { line } from 'd3-shape';

	export let curve: Curve;
	export let width: number;
	export let height: number;
	export let color: ScaleOrdinal<Shape, string>;
	export let strokeWidth = 2.5;
	export let circleRadius = 5;
	export let marginTop = 5;
	export let marginRight = 5;
	export let marginBottom = 5;
	export let marginLeft = 5;

	$: x =
		curve.kind === 'quantitative'
			? scaleLinear()
					.domain([curve.x[0], curve.x[curve.x.length - 1]])
					.range([marginLeft, width - marginRight])
			: scalePoint<number>()
					.domain(curve.x)
					.range([marginLeft, width - marginRight])
					.padding(0.5);

	$: y = scaleLinear()
		.domain([Math.min(...curve.y), Math.max(...curve.y)])
		.nice()
		.range([height - marginBottom, marginTop]);

	$: I = range(curve.x.length);

	$: path = line<number>()
		.x((i) => x(curve.x[i]) ?? 0)
		.y((i) => y(curve.y[i]));
</script>

<svg {width} {height}>
	<rect {width} {height} fill="none" stroke="var(--gray-3)" />
	<path stroke={color(curve.shape)} fill="none" stroke-width={strokeWidth} d={path(I)} />
	{#if curve.kind === 'categorical'}
		{#each I as i}
			<circle cx={x(curve.x[i])} cy={y(curve.y[i])} r={circleRadius} fill={color(curve.shape)} />
		{/each}
	{/if}
</svg>
