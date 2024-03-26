<script lang="ts">
	import type { Feature, Method, Cluster } from '../lib/types';
	import { scaleLinear, scalePoint, scaleOrdinal } from 'd3-scale';
	import type { Line } from 'd3-shape';
	import { line as d3line } from 'd3-shape';
	import XAxis from './XAxis.svelte';
	import YAxis from './YAxis.svelte';
	import { categoricalColors, scaleCanvas } from '$lib/vis-utils';
	import { onMount } from 'svelte';
	import { centerIceLines } from '$lib/utils';

	export let feature: Feature;
	export let method: Method;

	let canvas: HTMLCanvasElement;
	let ctx: CanvasRenderingContext2D;

	const margin = {
		top: 10,
		right: 10,
		bottom: 32,
		left: 50
	};

	let borderBoxSize: ResizeObserverSize[] | undefined | null;
	$: width = borderBoxSize ? borderBoxSize[0].inlineSize : 100;
	$: height = borderBoxSize ? borderBoxSize[0].blockSize : 100;

	$: clusters = feature.clusters[method];
	$: clusterIds = clusters.map((d) => d.aligned_id);

	$: x =
		feature.kind === 'quantitative'
			? scaleLinear()
					.domain([feature.x_values[0], feature.x_values[feature.x_values.length - 1]])
					.range([margin.left, width - margin.right])
			: scalePoint<number>()
					.domain(feature.x_values)
					.range([margin.left, width - margin.right])
					.padding(0.5);

	$: y = scaleLinear()
		.domain([feature.centered_ice_min, feature.centered_ice_max])
		.nice()
		.range([height - margin.bottom, margin.top]);

	$: dark = scaleOrdinal<number, string>().domain(clusterIds).range(categoricalColors.dark);

	$: light = scaleOrdinal<number, string>().domain(clusterIds).range(categoricalColors.light);

	$: line = d3line<number>()
		.x((_, i) => x(feature.x_values[i]) ?? 0)
		.y((d) => y(d))
		.context(ctx);

	$: centeredIceLines = centerIceLines(feature.ice);

	// canvas

	onMount(() => {
		ctx = canvas.getContext('2d') as CanvasRenderingContext2D;
	});

	function drawLines(
		centeredIceLines: number[][],
		centeredPdpLine: number[],
		clusters: Cluster[],
		ctx: CanvasRenderingContext2D,
		line: Line<number>,
		width: number,
		height: number
	) {
		// TODO: is this check needed?
		if (ctx === null || ctx === undefined || line === null || line === undefined) {
			return;
		}
		ctx.save();

		ctx.clearRect(0, 0, width, height);

		const clusterLabels = clusters
			.map((cluster) => cluster.indices.map((index) => ({ index, aligned_id: cluster.aligned_id })))
			.flat();

		// ice lines colored by cluster
		clusterLabels.forEach(({ index, aligned_id }) => {
			ctx.lineWidth = 1.0;
			ctx.globalAlpha = 0.25;

			ctx.beginPath();
			ctx.strokeStyle = light(aligned_id);
			line(centeredIceLines[index]);
			ctx.stroke();
		});

		// cluster mean lines
		clusters.forEach((cluster) => {
			ctx.lineWidth = 2.0;
			ctx.strokeStyle = dark(cluster.aligned_id);
			ctx.globalAlpha = 1.0;

			ctx.beginPath();
			line(cluster.centered_mean);
			ctx.stroke();
		});

		// pdp line

		ctx.lineWidth = 1.0;
		ctx.strokeStyle = 'black';
		ctx.globalAlpha = 1.0;

		ctx.beginPath();
		line(centeredPdpLine);
		ctx.stroke();

		ctx.restore();
	}

	/*
    TODO: is this true in this case?
    If scaleCanvas is called after drawLines, then it will clear the canvas.
    We need the draw function so that the reactive statement for scaleCanvas is
    not dependent on pd or line.
    */
	function draw() {
		drawLines(centeredIceLines, feature.centered_pdp, clusters, ctx, line, width, height);
	}

	$: if (ctx) {
		scaleCanvas(canvas, ctx, width, height);
		draw();
	}

	$: drawLines(centeredIceLines, feature.centered_pdp, clusters, ctx, line, width, height);
</script>

<div class="cluster-lines-chart" bind:borderBoxSize>
	<canvas bind:this={canvas} />
	<svg class="svg-for-clusters" {height} {width}>
		<g>
			<YAxis scale={y} x={margin.left} label={'Centered prediction'} />
			<XAxis
				scale={x}
				y={height - margin.bottom}
				label={feature.feature}
				integerOnly={feature.subkind === 'discrete'}
			/>
		</g>
	</svg>
</div>

<style>
	.cluster-lines-chart {
		position: relative;
		width: 100%;
		height: 100%;
	}

	.svg-for-clusters,
	canvas {
		position: absolute;
		top: 0;
		left: 0;
	}
</style>
