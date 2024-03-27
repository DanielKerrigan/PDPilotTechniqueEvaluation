// filtering

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

// clustering

export type Method = 'diff' | 'center';

export type Cluster = {
	id: number;
	aligned_id: number;
	indices: number[];
	centered_mean: number[];
	distance: number;
};

export type Feature = {
	id: number;
	dataset: string;
	feature: string;
	kind: 'quantitative' | 'categorical';
	subkind: 'nominal' | 'continuous' | 'discrete';
	x_values: number[];
	pdp: number[];
	centered_pdp: number[];
	ice: number[][];
	centered_ice_min: number;
	centered_ice_max: number;
	clusters: Record<Method, Cluster[]>;
	intersection: number;
	percent_overlap: number;
	scores: {
		method: Method;
		score: number;
	}[];
};
