import type { Curve } from './types';
import { PUBLIC_INPUT_FILE } from '$env/static/public';

export async function load({ fetch }) {
	const res = await fetch(`/${PUBLIC_INPUT_FILE}`);
	const json = (await res.json()) as {
		x: number[];
		y: number[];
		kind: 'quantitative' | 'categorical';
	}[];
	const curves = json.map(({ x, y, kind }) => ({ x, y, kind, shape: '' as const }) as Curve);
	return { curves };
}
