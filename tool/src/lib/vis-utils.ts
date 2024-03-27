import { format } from 'd3-format';
export { scaleCanvas, defaultFormat, categoricalColors };

// Adapted from https://www.html5rocks.com/en/tutorials/canvas/hidpi/
function scaleCanvas(
	canvas: HTMLCanvasElement,
	context: CanvasRenderingContext2D,
	width: number,
	height: number
): void {
	// assume the device pixel ratio is 1 if the browser doesn't specify it
	const devicePixelRatio = window.devicePixelRatio || 1;

	// set the 'real' canvas size to the higher width/height
	canvas.width = width * devicePixelRatio;
	canvas.height = height * devicePixelRatio;

	// ...then scale it back down with CSS
	canvas.style.width = `${width}px`;
	canvas.style.height = `${height}px`;

	// scale the drawing context so everything will work at the higher ratio
	context.scale(devicePixelRatio, devicePixelRatio);
}

function defaultFormat(x: number): string {
	/* [0, 1] is a common range for predictions and features.
  With SI suffixes, 0.5 becomes 500m. I'd rather it just be 0.5. */

	if ((x >= 0.001 && x <= 1) || (x >= -1 && x <= 0.001)) {
		return format('.3~f')(x);
	} else {
		return format('~s')(x);
	}
}

const categoricalColors = {
	light: ['#a6cee3', '#fdbf6f', '#b2df8a', '#fb9a99', '#cab2d6', '#FFB3F2', '#D4906D'],
	medium: ['#67a3cb', '#fea13f', '#7abf5a', '#f4645d', '#9c76b7', '#fb9ada', '#c3744a'],
	dark: ['#1f78b4', '#ff7f00', '#33a02c', '#e31a1c', '#6a3d9a', '#f781bf', '#B15928']
};
