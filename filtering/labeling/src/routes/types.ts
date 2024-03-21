export type Shape = 'increasing' | 'decreasing' | 'mixed' | '';

export type Curve = {
	x: number[];
	y: number[];
	kind: 'quantitative' | 'categorical';
	shape: Shape;
	unclear: boolean;
};
