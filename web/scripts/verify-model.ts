import fs from "fs";
import path from "path";

// polyfill fetch to serve the real local data file, so this test exercises
// the exact same loadWeights() function the browser will actually run
const dataPath = path.join(__dirname, "../public/data/model_weights.json");
(global as any).fetch = async (_url: string) => {
  const text = fs.readFileSync(dataPath, "utf-8");
  return { json: async () => JSON.parse(text) };
};

async function main() {
  const { loadWeights, runModel } = await import("../src/lib/model");
  const { CODE_TO_ROLE, ROLE_TO_CHANNEL } = await import("../src/lib/grid");

  const { weights, meta } = await loadWeights();
  console.log("loaded weight keys:", Object.keys(weights).length);
  console.log("conv1.weight length:", weights["conv1.weight"].length, "expected", 32*4*3*3);
  console.log("fc1.weight length:", weights["fc1.weight"].length, "expected", 128*2561);
  console.log("meta:", meta);

  const cells: [number, number, number][] = [
    [19, 5, 15], [21, 7, 15], [13, 9, 15], [21, 11, 15], [13, 13, 15],
    [19, 15, 12], [23, 15, 12], [23, 19, 13], [15, 21, 13], [25, 21, 15],
    [19, 27, 13], [19, 29, 13], [23, 33, 13], [23, 37, 14], [27, 37, 14],
  ];
  const image = new Float32Array(4 * 38 * 47);
  for (const [col, row, code] of cells) {
    const role = CODE_TO_ROLE[code];
    const ch = ROLE_TO_CHANNEL[role];
    image[ch * 38 * 47 + row * 47 + col] = 1.0;
  }
  console.log("image nonzero count:", Array.from(image).filter(v => v !== 0).length, "expected 15");

  const probs = runModel(image, 30 / meta.max_angle, weights, meta);
  console.log("REAL loadWeights() + runModel() probs:", probs.map((p) => Math.round(p * 10000) / 10000));
  console.log("expected:                              [0.6288, 0.3026, 0.0596, 0.0088, 0.0002, 0,0,0,0,0,0,0,0,0]");
}
main();
