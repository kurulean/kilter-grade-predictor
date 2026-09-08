// Hand-written forward pass for KilterCNN, ported from model.py.
// Verified against the real PyTorch model with a numpy reimplementation
// before porting -- same math, same weights, no shortcuts.

export type ModelWeights = Record<string, Float32Array>;

export interface ModelMeta {
  bn_eps: number;
  max_angle: number;
  num_classes: number;
}

async function base64ToFloat32(b64: string): Promise<Float32Array> {
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const view = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) view[i] = bin.charCodeAt(i);
  return new Float32Array(buf);
}

export async function loadWeights(): Promise<{ weights: ModelWeights; meta: ModelMeta }> {
  const res = await fetch("/data/model_weights.json");
  const data = await res.json();
  const weights: ModelWeights = {};
  for (const name in data.weights) {
    weights[name] = await base64ToFloat32(data.weights[name]);
  }
  return { weights, meta: data.meta as ModelMeta };
}

// x is (Cin,H,W) flat, channel-major. padding=1, stride=1.
function conv2d(
  x: Float32Array,
  Cin: number,
  H: number,
  Wd: number,
  weight: Float32Array,
  bias: Float32Array,
  Cout: number
): Float32Array {
  const out = new Float32Array(Cout * H * Wd);
  for (let oc = 0; oc < Cout; oc++) {
    const wBase = oc * Cin * 9;
    for (let oy = 0; oy < H; oy++) {
      for (let ox = 0; ox < Wd; ox++) {
        let acc = bias[oc];
        for (let ic = 0; ic < Cin; ic++) {
          const xBase = ic * H * Wd;
          const wBase2 = wBase + ic * 9;
          for (let ky = -1; ky <= 1; ky++) {
            const iy = oy + ky;
            if (iy < 0 || iy >= H) continue;
            const rowBase = xBase + iy * Wd;
            const wRow = wBase2 + (ky + 1) * 3;
            for (let kx = -1; kx <= 1; kx++) {
              const ix = ox + kx;
              if (ix < 0 || ix >= Wd) continue;
              acc += weight[wRow + (kx + 1)] * x[rowBase + ix];
            }
          }
        }
        out[oc * H * Wd + oy * Wd + ox] = acc;
      }
    }
  }
  return out;
}

function relu(x: Float32Array): Float32Array {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = x[i] > 0 ? x[i] : 0;
  return out;
}

function maxpool2(x: Float32Array, C: number, H: number, Wd: number) {
  const H2 = H >> 1;
  const W2 = Wd >> 1;
  const out = new Float32Array(C * H2 * W2);
  for (let c = 0; c < C; c++) {
    const cBase = c * H * Wd;
    const oBase = c * H2 * W2;
    for (let i = 0; i < H2; i++) {
      for (let j = 0; j < W2; j++) {
        const a = x[cBase + 2 * i * Wd + 2 * j];
        const b = x[cBase + 2 * i * Wd + 2 * j + 1];
        const c2 = x[cBase + (2 * i + 1) * Wd + 2 * j];
        const d = x[cBase + (2 * i + 1) * Wd + 2 * j + 1];
        out[oBase + i * W2 + j] = Math.max(a, b, c2, d);
      }
    }
  }
  return { data: out, H: H2, W: W2 };
}

function batchnorm(
  x: Float32Array,
  C: number,
  H: number,
  Wd: number,
  weights: ModelWeights,
  prefix: string,
  eps: number
): Float32Array {
  const weight = weights[`${prefix}.weight`];
  const bias = weights[`${prefix}.bias`];
  const mean = weights[`${prefix}.running_mean`];
  const varr = weights[`${prefix}.running_var`];
  const out = new Float32Array(x.length);
  for (let c = 0; c < C; c++) {
    const scale = weight[c] / Math.sqrt(varr[c] + eps);
    const shift = bias[c] - mean[c] * scale;
    const base = c * H * Wd;
    for (let i = 0; i < H * Wd; i++) out[base + i] = x[base + i] * scale + shift;
  }
  return out;
}

function linear(
  x: Float32Array,
  outDim: number,
  inDim: number,
  weight: Float32Array,
  bias: Float32Array
): Float32Array {
  const out = new Float32Array(outDim);
  for (let o = 0; o < outDim; o++) {
    let acc = bias[o];
    const base = o * inDim;
    for (let i = 0; i < inDim; i++) acc += weight[base + i] * x[i];
    out[o] = acc;
  }
  return out;
}

function softmax(x: Float32Array): number[] {
  const m = Math.max(...x);
  const exps = Array.from(x).map((v) => Math.exp(v - m));
  const s = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / s);
}

// image: Float32Array(4*38*47), channel-major. angleNorm: 0..1
export function runModel(
  image: Float32Array,
  angleNorm: number,
  weights: ModelWeights,
  meta: ModelMeta
): number[] {
  let x = conv2d(image, 4, 38, 47, weights["conv1.weight"], weights["conv1.bias"], 32);
  x = relu(x);
  let p = maxpool2(x, 32, 38, 47);
  x = batchnorm(p.data, 32, p.H, p.W, weights, "bn1", meta.bn_eps);

  x = conv2d(x, 32, p.H, p.W, weights["conv2.weight"], weights["conv2.bias"], 64);
  x = relu(x);
  p = maxpool2(x, 64, p.H, p.W);
  x = batchnorm(p.data, 64, p.H, p.W, weights, "bn2", meta.bn_eps);

  x = conv2d(x, 64, p.H, p.W, weights["conv3.weight"], weights["conv3.bias"], 128);
  x = relu(x);
  p = maxpool2(x, 128, p.H, p.W);
  x = batchnorm(p.data, 128, p.H, p.W, weights, "bn3", meta.bn_eps);

  const flat = new Float32Array(x.length + 1);
  flat.set(x);
  flat[x.length] = angleNorm;

  let h = linear(flat, 128, flat.length, weights["fc1.weight"], weights["fc1.bias"]);
  h = relu(h);
  const logits = linear(h, meta.num_classes, 128, weights["fc2.weight"], weights["fc2.bias"]);
  return softmax(logits);
}
