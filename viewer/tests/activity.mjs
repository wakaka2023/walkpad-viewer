import assert from 'node:assert/strict';
import fs from 'node:fs';
import {activityEngine} from '../src/activity-engine.ts';
import {ACTIVITY_WALK_RUN_UNCERTAIN, analyzeGaitWindow, reviewGaitSpan, windowFeatures, windowProbabilities} from '../src/gait-window.ts';
import {ACTIVITY_STILL, ACTIVITY_TURN_L, ACTIVITY_TURN_R} from '../src/core/constants.ts';
import {analyzeGaitExtraction, annotateActivity4, screenGait, WALK_REVIEW} from '../src/turn-rule.ts';
import {analyzeMoticonTas, decodeMoticonTas, predictMoticonTas} from '../src/moticon-tas.ts';
import {analyzeActivity4, predictActivity4} from '../src/activity4.ts';
const contract = JSON.parse(fs.readFileSync(new URL('../src/o1-model.json', import.meta.url)));
const fixture = JSON.parse(fs.readFileSync(new URL('./o1-parity.json', import.meta.url)));
const engine = activityEngine(), models = engine.load(contract), n = fixture.input[0].length;
const x = Float32Array.from(fixture.input.flat());
const tic = performance.now(), p = engine.predict(models, x, n), expected = fixture.probabilities.flat();
let error = 0; for (let i = 0; i < p.length; i++) error = Math.max(error, Math.abs(p[i] - expected[i]));
assert.ok(error < 2e-5, `PyTorch mismatch ${error}`);
console.log('PASS O1 four-checkpoint PyTorch parity', error, 'milliseconds', performance.now() - tic);
const exact = new Float32Array(90);
for (let k = 0; k < 30; k++) for (let c = 0; c < 3; c++) exact[c * 30 + k] = c === Math.floor(k / 10) ? 1 : 0;
assert.deepEqual([...engine.decode(exact, 2)], Array.from({length: 30}, (_, i) => Math.floor(i / 10)));
const make = () => ({t: Float64Array.from({length: 501}, (_, i) => i / 50), seq: Float64Array.from({length: 501}, (_, i) => i), ok: new Uint8Array(501).fill(1), axes: Array.from({length: 6}, (_, c) => Float32Array.from({length: 501}, (_, i) => c === 2 ? 1 : Math.sin(i / 10) * (c > 2 ? 30 : .1)))});
const left = make(), right = make(); right.ok.fill(0, 200, 250);
const out = await engine.analyze(contract, [left, right], 10);
assert.equal(out.ticks, 100); assert.ok([...out.labels.slice(40, 50)].every(v => v === -1)); assert.ok([...out.labels.slice(60, 80)].every(v => v >= 0));
assert.ok([...out.labels.slice(0, 2)].every(v => v === -1));
const a = engine.sample(left, 500); assert.equal(a.group[0] >= 0, true);
const missing = make(); missing.ok.fill(0); const no = await engine.analyze(contract, [left, missing], 10); assert.equal(no.validTicks, 0);
const reset = make(); reset.t[250] = 0; assert.throws(() => engine.sample(reset, 500), /时间倒退/);
const full = JSON.parse(fs.readFileSync(new URL('./o1-preprocess-parity.json', import.meta.url)));
const fullSources = full.sources.map(s => ({t: Float64Array.from(s.t), seq: Float64Array.from(s.seq), ok: Uint8Array.from(s.ok), axes: s.axes.map(a => Float32Array.from(a))}));
const fullOut = await engine.analyze(contract, fullSources, full.duration);
assert.deepEqual([...fullOut.labels], full.labels);
let fullError = 0; for (let c = 0; c < 3; c++) for (let k = 0; k < fullOut.ticks; k++) if (full.probabilities[c][k] != null) fullError = Math.max(fullError, Math.abs(fullOut.probs[c * fullOut.ticks + k] - full.probabilities[c][k]));
assert.ok(fullError < 2e-5, String(fullError));
const part = new Float32Array(p.length); const total = n / 5;
for (let s = 0; s < total; s += 47) { const e = Math.min(total, s + 47), a = Math.max(0, s - 41), b = Math.min(total, e + 41), nn = (b - a) * 5, xx = new Float32Array(12 * nn); for (let c = 0; c < 12; c++) xx.set(x.subarray(c * n + a * 5, c * n + b * 5), c * nn); const pp = engine.predict(models, xx, nn); for (let c = 0; c < 3; c++) part.set(pp.subarray(c * (b - a) + s - a, c * (b - a) + e - a), c * total + s); }
let chunkError = 0; for (let k = 0; k < p.length; k++) chunkError = Math.max(chunkError, Math.abs(p[k] - part[k])); assert.ok(chunkError < 2e-5);
console.log('PASS full SciPy preprocessing + PyTorch + Viterbi pipeline', fullError, 'chunk error', chunkError);
console.log('PASS global decoder, 200ms recovery, dual-foot gaps, missing-foot, clock-reset guards');

const tasRef = JSON.parse(fs.readFileSync(new URL('./moticon-tas-parity.json', import.meta.url)));
const ns = tasRef.n, nt = ns / 5, classMap = [ACTIVITY_STILL, 0, 1, 2];
const tasInput = Float32Array.from({length: 12 * ns}, (_, k) => {
  const c = Math.floor(k / ns), i = k % ns;
  return (Math.sin(i * (.017 + c * .002)) + .27 * Math.cos(i * .071 + c)) * (c % 6 < 3 ? .8 : 210) + (c % 6 === 2 ? 1 : 0);
});
const tas = predictMoticonTas(tasInput, ns);
let tasError = 0, boundaryError = 0;
for (let i = 0; i < nt; i++) {
  const v = [0, 1, 2, 3].map(c => tas.logits[c * nt + i]), max = Math.max(...v);
  const exp = v.map(x => Math.exp(x - max)), total = exp.reduce((a, b) => a + b);
  for (let c = 0; c < 4; c++) tasError = Math.max(tasError, Math.abs(exp[c] / total - tasRef.probabilities[c][i]));
  boundaryError = Math.max(boundaryError, Math.abs(tas.boundary[i] - tasRef.boundary[i]));
}
assert.ok(tasError < 2e-5 && boundaryError < 2e-5, `PyTorch error state=${tasError} boundary=${boundaryError}`);
assert.deepEqual([...decodeMoticonTas(tas.logits, tas.boundary)], tasRef.labels);
assert.deepEqual(predictMoticonTas(Float32Array.from([...tasInput.subarray(6 * ns), ...tasInput.subarray(0, 6 * ns)]), ns), tas);
const tasGap = await analyzeMoticonTas(engine, [left, right], 10);
assert.ok(tasGap.labels.slice(40, 52).every(v => v === -1));
assert.ok(tasGap.labels.slice(60, 80).every(v => v >= 0));
assert.equal((await analyzeMoticonTas(engine, [left, missing], 10)).validTicks, 0);
const inputSource = [0, 1].map(foot => ({
  t: Float64Array.from({length: ns}, (_, i) => (i + .5) / 50), seq: Float64Array.from({length: ns}, (_, i) => i),
  ok: new Uint8Array(ns).fill(1), axes: Array.from({length: 6}, (_, c) => tasInput.subarray((foot * 6 + c) * ns, (foot * 6 + c + 1) * ns))
}));
const chunked = await analyzeMoticonTas(engine, inputSource, ns / 50);
const filtered = engine.filterSpan(tasInput.subarray(0, 6 * ns), tasInput.subarray(6 * ns), ns, 0, ns,
  [1, 1.561018075800718, .641351538057563], [.8005924034645703, 1.6011848069291406, .8005924034645703]);
const len = ns - 10, tk = len / 5;
const filteredTasInput = new Float32Array(12 * len);
for (let c = 0; c < 12; c++) filteredTasInput.set(filtered.subarray(c * ns + 10, (c + 1) * ns), c * len);
const singlePass = predictMoticonTas(filteredTasInput, len);
assert.deepEqual([...chunked.labels.slice(2)], [...decodeMoticonTas(singlePass.logits, singlePass.boundary)].map(c => classMap[c]));
assert.equal(chunked.validTicks, tk);
assert.equal(chunked.edge[257], 0);
for (let i = 0; i < tk; i++) {
  const logits = classMap.map((_, c) => singlePass.logits[c * tk + i]), max = Math.max(...logits);
  const exp = logits.map(v => Math.exp(v - max)), total = exp.reduce((a, b) => a + b);
  for (const [c, cls] of classMap.entries()) assert.ok(Math.abs(chunked.probs[cls * nt + i + 2] - exp[c] / total) < 2e-5);
}
console.log('PASS Moticon TAS PyTorch state/boundary/decode parity, chunk joins and gap isolation', tasError, boundaryError);

const activity4Ref = JSON.parse(fs.readFileSync(new URL('./activity4-parity.json', import.meta.url)));
const makeActivity4 = n => Float32Array.from({length: 12 * n}, (_, k) => {
  const c = Math.floor(k / n), i = k % n;
  return (Math.sin(i * (.024 + c * .003)) + .25 * Math.cos(i * .077 + c)) * (c % 6 < 3 ? .8 : 220) + (c % 6 === 2 ? 1 : 0);
});
let activity4Error = 0, activity4BoundaryError = 0;
for (const {samples, logits, boundaryLogits, decoded} of activity4Ref) {
  const actual = predictActivity4(makeActivity4(samples), samples), target = logits.flat();
  assert.equal(actual.logits.length, target.length);
  assert.equal(actual.boundaryLogits.length, boundaryLogits.length);
  for (let i = 0; i < target.length; i++) activity4Error = Math.max(activity4Error, Math.abs(actual.logits[i] - target[i]));
  for (let i = 0; i < boundaryLogits.length; i++) activity4BoundaryError = Math.max(activity4BoundaryError, Math.abs(actual.boundaryLogits[i] - boundaryLogits[i]));
  const boundary = actual.boundaryLogits.map(v => 1 / (1 + Math.exp(-v)));
  assert.deepEqual([...decodeMoticonTas(actual.logits, boundary, 5)], decoded);
}
assert.ok(activity4Error < 2e-5 && activity4BoundaryError < 2e-5, `ONNX activity4 mismatch: ${activity4Error}, ${activity4BoundaryError}`);
const blipLogits = new Float32Array(20 * 4);
blipLogits.fill(3, 0, 20);
blipLogits[20 + 8] = 4;
assert.ok(decodeMoticonTas(blipLogits, new Float32Array(20), 5).every(c => c === 0));
blipLogits.fill(7, 20 + 10, 40);
assert.deepEqual([...decodeMoticonTas(blipLogits, new Float32Array(20), 5)], [...Array(10).fill(0), ...Array(10).fill(1)]);
const a4n = activity4Ref.at(-1).samples, a4x = makeActivity4(a4n), a4ticks = a4n / 5;
const a4sources = [0, 1].map(foot => ({
  t: Float64Array.from({length: a4n}, (_, i) => (i + .5) / 50), seq: Float64Array.from({length: a4n}, (_, i) => i),
  ok: new Uint8Array(a4n).fill(1), axes: Array.from({length: 6}, (_, c) => a4x.subarray((foot * 6 + c) * a4n, (foot * 6 + c + 1) * a4n))
}));
const a4full = await analyzeActivity4(engine, a4sources, a4n / 50);
assert.equal(a4full.validTicks, a4ticks);
assert.equal(a4full.edge[250], 0);
assert.equal(a4full.edge[255], 1);
for (let i = 0; i < a4ticks; i++) {
  const z = activity4Ref.at(-1).logits.map(c => c[i]), max = Math.max(...z), p = z.map(v => Math.exp(v - max));
  const sum = p.reduce((a, b) => a + b);
  assert.equal(a4full.labels[i], classMap[activity4Ref.at(-1).decoded[i]]);
  for (const [c, cls] of classMap.entries()) assert.ok(Math.abs(a4full.probs[cls * a4ticks + i] - p[c] / sum) < 2e-5);
}
const a4gap = await analyzeActivity4(engine, [left, right], 10);
assert.ok(a4gap.labels.slice(40, 50).every(v => v === -1));
assert.ok(a4gap.labels.slice(60, 80).every(v => v >= 0));
assert.equal((await analyzeActivity4(engine, [left, missing], 10)).validTicks, 0);
console.log('PASS Activity4 ONNX state and boundary parity, joint decoding, chunk joins and gap isolation', activity4Error, activity4BoundaryError);

const feetFor = (seconds, swing, yaw = () => 0) => [0, 1].map(foot => {
  const count = seconds * 50, source = {
    t: Float64Array.from({length: count}, (_, i) => (i + .5) / 50), seq: Float64Array.from({length: count}, (_, i) => i),
    ok: new Uint8Array(count).fill(1), axes: Array.from({length: 6}, () => new Float32Array(count))
  };
  source.axes[2].fill(1);
  for (let i = 0; i < count; i++) {
    source.axes[3][i] = swing(i / 50, foot) * Math.sin(i * 2 * Math.PI / 50 + foot * Math.PI);
    source.axes[5][i] = yaw(i / 50);
  }
  return source;
});
const screen = (swing, labels = new Int8Array(100)) => screenGait(labels, new Int8Array(100), feetFor(10, swing).map(s => engine.sample(s, 500)), 500);
assert.ok(screen(() => 220).labels.every(v => v === 0));
assert.ok(screen(() => 95).walkReview.every(v => v === WALK_REVIEW.lowSwing));
assert.ok(screen((_, foot) => foot ? 95 : 350).labels.every(v => v === 2));
const pause = screen(t => t >= 4 && t < 6 ? 0 : 250);
assert.equal(pause.labels[50], 2); assert.ok(pause.walkReview[50] & WALK_REVIEW.pause);
assert.equal(pause.labels[30], 0); assert.equal(pause.labels[70], 0);
const shortWalk = new Int8Array(100).fill(2); shortWalk.fill(0, 30, 49);
assert.ok(screen(() => 220, shortWalk).walkReview.slice(30, 49).every(v => v === WALK_REVIEW.short));
const turnFeet = feetFor(23, () => 0, t => t >= 4 && t < 7 ? 80 : t >= 13 && t < 16 ? -80 : 0);
turnFeet[1].ok.fill(0, 500, 525);
const takeoff = new Int8Array(230).fill(ACTIVITY_STILL); takeoff.fill(1, 110, 230); takeoff.fill(-1, 100, 105);
const extracted = await analyzeGaitExtraction(engine, turnFeet, 23, () => {}, 'extraction', {labels: takeoff, edge: new Uint8Array(230), probs: new Float32Array(1150)});
assert.equal(extracted.labels[20], 2);
assert.equal(extracted.labels[50], ACTIVITY_TURN_L);
assert.equal(extracted.labels[140], ACTIVITY_TURN_R);
assert.equal(extracted.labels[200], 1);
assert.ok(extracted.labels.slice(100, 105).every(v => v === -1));
assert.equal(extracted.turnRuns.length, 2);
const baseAnnotated = {...a4gap, labels: takeoff.slice(), edge: new Uint8Array(230), probs: new Float32Array(1150),
  ticks: 230, duration: 23, validTicks: 225};
baseAnnotated.labels.fill(0, 35, 75);
const annotated = await annotateActivity4(engine, turnFeet, 23, () => {}, 'annotated', baseAnnotated);
assert.strictEqual(annotated.labels, baseAnnotated.labels);
assert.strictEqual(annotated.probs, baseAnnotated.probs);
assert.strictEqual(annotated.edge, baseAnnotated.edge);
assert.equal(annotated.labels[20], ACTIVITY_STILL);
assert.equal(annotated.labels[50], 0);
assert.equal(annotated.labels[140], 1);
assert.equal(annotated.turns[50], ACTIVITY_TURN_L);
assert.equal(annotated.turns[140], ACTIVITY_TURN_R);
assert.ok(annotated.walkReview[50] & WALK_REVIEW.lowSwing);
assert.equal(annotated.walkReview[140], 0);
assert.ok(annotated.turns.slice(100, 105).every(v => v === 0));
const broken = feetFor(4, () => 0, () => 35);
for (const foot of broken) foot.ok.fill(0, 96, 104);
const brokenResult = await analyzeGaitExtraction(engine, broken, 4, () => {}, 'extraction', {
  labels: new Int8Array(40).fill(-1), edge: new Uint8Array(40), probs: new Float32Array(200)
});
assert.equal(brokenResult.turnRuns.length, 0);
console.log('PASS step extraction: dual-foot swing, stop, minimum duration, independent turns, gaps and straight running');

const turnProbe = async (feet, seconds) => annotateActivity4(engine, feet, seconds, () => {}, 'turn-probe', {
  labels: new Int8Array(seconds * 10).fill(1), edge: new Uint8Array(seconds * 10)
});
const uturn = feetFor(10, () => 220, t => t >= 4 && t < 5.2 ? 150 : 0);
const fastTurn = await turnProbe(uturn, 10);
assert.equal(fastTurn.turnRuns.length, 1);
const fast = fastTurn.turnRuns[0];
assert.ok(fast.a >= 3.7 && fast.b <= 5.5, JSON.stringify(fast));
assert.ok(fast.angle >= 165 && fast.angle <= 185, JSON.stringify(fast));
assert.equal(fastTurn.turns[33], 0);
assert.equal(fastTurn.turns[46], ACTIVITY_TURN_L);
assert.equal(fastTurn.turns[58], 0);
assert.ok(fastTurn.labels.every(v => v === 1));
const extractionTurn = await analyzeGaitExtraction(engine, uturn, 10, () => {}, 'extraction', {
  labels: fastTurn.labels, edge: new Uint8Array(100), probs: new Float32Array(500)
});
assert.equal(extractionTurn.turnRuns.length, 1);
const wide = extractionTurn.turnRuns[0];
assert.ok(wide.a < fast.a && wide.b > fast.b);
assert.equal(wide.angle, fast.angle);
assert.equal(wide.dir, fast.dir);
uturn[1].ok.fill(0, 180, 280);
assert.equal((await turnProbe(uturn, 10)).turnRuns.length, 0);

const gradual = await turnProbe(feetFor(12, () => 220, t => t >= 3 && t < 9 ? -29 : 0), 12);
assert.equal(gradual.turnRuns.length, 1);
assert.equal(gradual.turns[60], ACTIVITY_TURN_R);
assert.ok(gradual.turnRuns[0].b - gradual.turnRuns[0].a >= 5.5);

const doubleArc = await turnProbe(feetFor(10, () => 220, t => t >= 3 && t < 4 || t >= 4.8 && t < 5.8 ? 125 : 0), 10);
assert.equal(doubleArc.turnRuns.length, 1);
assert.ok(doubleArc.turnRuns[0].a < 3 && doubleArc.turnRuns[0].b > 5.8, JSON.stringify(doubleArc.turnRuns));
const strideYaw = await turnProbe(feetFor(12, () => 250, t => 185 * Math.sin(4 * Math.PI * t)), 12);
assert.equal(strideYaw.turnRuns.length, 0);
console.log('PASS turn duration refinement, slow and multi-peak fallback, gap isolation and alternating stride yaw');

const nReview = 160, first = 3, grid = (nReview + first) * 5;
const gyro = (lag, amplitude = 100, phase = 0) => {
  const v = new Float32Array(6 * grid);
  for (let i = 0; i < grid; i++) v[3 * grid + i] = amplitude * Math.sin((i + phase) * 2 * Math.PI / lag);
  return v;
};
const votes = (walk, run) => Float32Array.from({length: 3 * nReview}, (_, i) => [walk, run, 1 - walk - run][Math.floor(i / nReview)]);
const other = () => new Int8Array(nReview).fill(2);
const weak = votes(.05, .02), strong = votes(.05, .75);
function checked(y, v, r) {
  const changed = new Uint8Array(nReview + first);
  engine.reviewRunning(y, v, gyro(34), r, grid, first, changed);
  return [...changed.entries()].filter(([, flag]) => flag).map(([i]) => i - first);
}
const y = other(); y.fill(0, 0, 25); y.fill(1, 130);
assert.deepEqual(checked(y, [weak, strong], gyro(34, 100, 11)), Array.from({length: 105}, (_, i) => i + 25));
assert.ok(y.slice(0, 25).every(v => v === 0));
assert.ok(y.slice(130).every(v => v === 1));
assert.deepEqual(checked(other(), [weak, strong], gyro(48)), []);
assert.deepEqual(checked(other(), [weak, strong], gyro(34, 10)), []);
assert.deepEqual(checked(other(), [weak, votes(.30, .64)], gyro(34)), []);
assert.deepEqual(checked(other(), [weak, votes(.05, .59)], gyro(34)), []);
const short = other(); short.fill(0); short.fill(2, 50, 89);
assert.deepEqual(checked(short, [weak, strong], gyro(34)), []);
console.log('PASS dual-foot periodicity, member support, minimum span, and existing label preservation');

const raw = Float32Array.from({length: 12 * 180}, (_, k) => {
  const j = Math.floor(k / 180), i = k % 180;
  return (Math.sin((i + 1) * (j + 2) * .071) + .35 * Math.cos((i + 4) * (j + 1) * .113)) * (j % 6 < 3 ? 1 : 170) + (j % 6 === 2 ? .9 : 0);
});
const input = [0, 1].map(foot => ({
  t: Float64Array.from({length: 180}, (_, i) => (i + .5) / 50), seq: Float64Array.from({length: 180}, (_, i) => i), ok: new Uint8Array(180).fill(1),
  axes: Array.from({length: 6}, (_, c) => raw.subarray((foot * 6 + c) * 180, (foot * 6 + c + 1) * 180))
}));
const gaitModelId = 'WT-offline-GaitWindow';
const gw = await analyzeGaitWindow(engine, gaitModelId, input, 3.6);
assert.equal(gw.modelId, gaitModelId);
assert.equal(gw.validTicks, 12);
assert.ok(gw.labels.slice(0, 13).every(v => v === -1) && gw.labels.slice(25).every(v => v === -1));
assert.ok(Number.isNaN(gw.probs[0]));
const referenceRun = [7.403558379910698e-6, 6.639819408006743e-6, 8.591744442789277e-6, 6.48918278141201e-6];
for (const [i, tick] of [14, 16, 18, 24].entries()) {
  assert.equal(gw.labels[tick], 2);
  assert.ok(Math.abs(gw.probs[gw.ticks + tick] / referenceRun[i] - 1) < 1e-6);
  assert.equal(gw.probs[gw.ticks + tick], gw.probs[gw.ticks + tick - 1]);
}
const gwGap = await analyzeGaitWindow(engine, gaitModelId, [left, right], 10);
assert.ok(gwGap.labels.slice(27, 63).every(v => v === -1));
assert.ok(gwGap.labels[20] >= 0 && gwGap.labels[70] >= 0);
assert.equal((await analyzeGaitWindow(engine, gaitModelId, [left, missing], 10)).validTicks, 0);
const gv = windowFeatures(raw, 180, 20), swapped = Float32Array.from([...raw.subarray(1080), ...raw.subarray(0, 1080)]);
assert.equal(gv.length, 128);
assert.deepEqual(windowFeatures(swapped, 180, 20), gv);
assert.ok(Math.abs(windowProbabilities(gv).reduce((s, v) => s + v, 0) - 1) < 1e-12);
console.log('PASS GaitWindow frozen SciPy/NumPy inference parity, foot symmetry, full window and gap guards');

const stillFoot = () => {
  const f = make(); f.axes.forEach((a, c) => a.fill(c === 2 ? 1 : 0)); return f;
};
const stillLeft = stillFoot(), stillRight = stillFoot(); stillRight.ok.fill(0, 200, 250);
const refined = (feet, seconds) => analyzeGaitWindow(engine, 'WT-offline-GaitWindow-Refined', feet, seconds, undefined, true);
const stationary = await refined([stillLeft, stillRight], 10);
assert.equal(stationary.labels[0], -1);
assert.equal(stationary.labels[1], ACTIVITY_STILL);
assert.equal(stationary.labels[98], ACTIVITY_STILL);
assert.equal(stationary.labels[99], -1);
assert.ok(stationary.labels.slice(40, 50).every(v => v === -1));
assert.equal(stationary.validTicks, stationary.labels.filter(v => v >= 0).length);
assert.equal((await refined([stillLeft, missing], 10)).validTicks, 0);
const oneFootMoving = await refined([left, stillFoot()], 10);
assert.equal(oneFootMoving.labels[0], -1);
assert.equal(oneFootMoving.labels[99], -1);
assert.deepEqual((await refined(input, 3.6)).probs, gw.probs);

const demo = (flank, insert, scores, ticks = 2) => {
  const n = 60, grid = n * 5, y = new Int8Array(n).fill(flank), changed = new Uint8Array(n);
  y.fill(insert, 22, 22 + ticks);
  const p = Float32Array.from({length: 3 * n}, (_, k) => scores[Math.floor(k / n)]);
  const moving = new Float32Array(6 * grid); moving.fill(100, 3 * grid, 4 * grid);
  reviewGaitSpan(y, p, changed, moving, moving, grid, 0, grid, Array.from({length: n / 2}, (_, i) => i * 2));
  return [y[22], changed.reduce((a, b) => a + b)];
};
assert.deepEqual(demo(0, 1, [.42, .54, .04]), [0, 2]);
assert.deepEqual(demo(0, 1, [.02, .97, .01]), [1, 0]);
assert.deepEqual(demo(2, 0, [.54, .04, .42]), [0, 0]);
assert.deepEqual(demo(0, 1, [.42, .54, .04], 12), [1, 0]);
console.log('PASS refined stillness at record edges, real gaps, unchanged raw probability and weak isolated-segment review');

const temporal = (feet, seconds) => analyzeGaitWindow(engine, 'GaitWindow-Temporal', feet, seconds, undefined, 'temporal');
assert.deepEqual((await temporal([stillLeft, stillRight], 10)).labels, stationary.labels);
assert.equal((await temporal([stillLeft, missing], 10)).validTicks, 0);
assert.deepEqual((await temporal(input, 3.6)).probs, gw.probs);

function decodeCluster({strong = false, mismatch = false, gap = false} = {}) {
  const n = 100, grid = n * 5, a = 22, b = a + 12;
  const y = new Int8Array(n).fill(1), changed = new Uint8Array(n), p = new Float32Array(3 * n);
  const signal = (phase, alters) => {
    const x = new Float32Array(6 * grid);
    for (let i = 0; i < grid; i++) {
      x[2 * grid + i] = 1;
      const lag = alters && i >= a * 5 && i < b * 5 ? 58 : 34;
      x[3 * grid + i] = 140 * Math.sin((i + phase) * 2 * Math.PI / lag);
      x[4 * grid + i] = 100 * Math.cos((i + phase) * 2 * Math.PI / lag);
    }
    return x;
  };
  for (let i = 0; i < n; i++) {
    const island = i >= a && i < b;
    const c = island ? [0, 1, 0, 0, 1, 0][Math.floor((i - a) / 2)] : 1;
    const q = island ? (strong && i === a ? .97 : .55) : .96;
    y[i] = c; p[c * n + i] = q; p[(1 - c) * n + i] = 1 - q - .01; p[2 * n + i] = .01;
  }
  const before = y.slice(), pairs = Array.from({length: n / 2}, (_, i) => i * 2);
  if (gap) {
    y.fill(-1, 26, 28);
    pairs.splice(pairs.indexOf(26), 1);
  }
  reviewGaitSpan(y, p, changed, signal(0, false), signal(15, mismatch), grid, 0, grid, pairs, true);
  return {y, changed, before, a, b};
}
const joined = decodeCluster();
assert.ok(joined.y.every(c => c === 1));
assert.equal(joined.changed.filter(Boolean).length, 8);
const uncertain = decodeCluster({mismatch: true});
assert.deepEqual([...uncertain.y.slice(uncertain.a, uncertain.b)], Array(12).fill(ACTIVITY_WALK_RUN_UNCERTAIN));
assert.ok(uncertain.y.slice(uncertain.b).every(c => c === 1));
const protectedCluster = decodeCluster({strong: true});
assert.deepEqual(protectedCluster.y, protectedCluster.before);
const disconnected = decodeCluster({gap: true});
assert.deepEqual([...disconnected.y.slice(26, 28)], [-1, -1]);
assert.ok(!disconnected.changed.some(Boolean));
console.log('PASS temporal refinement resolves one low-evidence cluster, abstains on bilateral mismatch, preserves strong evidence and gaps');
