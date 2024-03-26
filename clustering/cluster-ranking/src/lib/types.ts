export type Method = 'diff' | 'center';

export type AnonymousLabel =
	| 'Left much better'
	| 'Left somewhat better'
	| 'Neutral'
	| 'Right somewhat better'
	| 'Right much better'
	| '';

export type Label =
	| 'diff much better'
	| 'diff somewhat better'
	| 'neutral'
	| 'center somewhat better'
	| 'center much better'
	| '';

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
	order: Method[];
	labels: Label[];
	labelIndex: number;
	label: Label;
};
