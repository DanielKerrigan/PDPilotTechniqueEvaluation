import type { Curve } from '$lib/types';
import { PUBLIC_INPUT_FILE } from '$env/static/public';

export async function load({ fetch }) {
	const res = await fetch(`/${PUBLIC_INPUT_FILE}`);
	const json = (await res.json()) as Omit<Curve, 'shape' | 'unclear'>[];
	const curves = json.map((d) => {
		const curve: Curve = { ...d, shape: '', unclear: false };
		return curve;
	});
	return { curves };
}
