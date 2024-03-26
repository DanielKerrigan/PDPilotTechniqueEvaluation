export type Shape = 'increasing' | 'decreasing' | 'mixed' | '';

export type Curve = {
	index: number;
	dataset: string;
	feature: string;
	x: number[];
	y: number[];
	kind: 'quantitative' | 'categorical';
	shape: Shape;
	unclear: boolean;
};
