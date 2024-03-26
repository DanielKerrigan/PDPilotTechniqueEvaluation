import type { Feature, Method } from '../lib/types';
import { PUBLIC_INPUT_FILE } from '$env/static/public';
import { shuffle } from 'd3-array';

export async function load({ fetch }) {
	const res = await fetch(`/${PUBLIC_INPUT_FILE}`);
	const json = (await res.json()) as Omit<Feature, 'order' | 'labels' | 'label' | 'labelIndex'>[];

	const features: Feature[] = json.map((d) => {
		const order = shuffle(Object.keys(d.clusters) as Method[]);
		return {
			...d,
			order: order,
			labels: [
				`${order[0]} much better`,
				`${order[0]} somewhat better`,
				'neutral',
				`${order[1]} somewhat better`,
				`${order[1]} much better`
			],
			label: '',
			labelIndex: -1
		};
	});

	return { features };
}
