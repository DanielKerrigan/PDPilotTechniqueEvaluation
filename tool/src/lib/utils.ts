export { centerIceLines };

/**
 * Centers the ICE lines.
 * @param iceLines standard ICE lines
 * @returns centered ICE lines
 */
function centerIceLines(iceLines: number[][]): number[][] {
	return iceLines.map((line) => line.map((d) => d - line[0]));
}
