import assert from 'node:assert/strict';
import {ankleTimebase} from '../src/squat-clock.ts';

const HZ = 50, N = 1500;

/** t/receive/elapsed 三列：offset 为接收偏移（秒），jitter 逐帧附加抖动（秒）。 */
function stream(offset, jitter = () => 0, origin = 0) {
  const t = new Float64Array(N), receive = new Float64Array(N), elapsed = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    t[i] = origin + Math.round(i * 1e6 / HZ);
    elapsed[i] = t[i] + 250e3;
    receive[i] = Math.round((t[i] / 1e6 + offset + jitter(i)) * 1e9);
  }
  return {t, values: new Float64Array(N * 6), receive, elapsed};
}

const tail = i => (i % 10 === 3 ? 0.05 + (i % 7) * 0.025 : 0);   // 一成帧晚到 50–200 ms
const spread = i => (i % 2 ? 0.05 : -0.05);                       // 主体每帧来回摆 50 ms
const ramp = i => (i > N / 2 ? 0.15 : 0);                         // 后半段整体后移

// 尾部晚到的 BLE 包不影响中位数锚点：应当通过
{
  const tb = ankleTimebase([stream(3.2, tail), stream(3.21, tail)], 0);
  assert.ok(Math.abs(tb.shifts[0] + tb.shifts[1]) < 1e-6, '两脚位移应对称抵消');
  assert.ok(Math.abs(Math.abs(tb.shifts[0]) - 0.005) < 0.005, '位移应接近两脚中位偏移之差的一半：' + tb.shifts[0]);
}

// 主体抖动大：中位数不再代表对时，应当拒绝
assert.throws(() => ankleTimebase([stream(3.2, spread), stream(3.2, spread)], 0), /时间轴波动较大/);

// 锚点在段间漂移：应当拒绝
assert.throws(() => ankleTimebase([stream(3.2, ramp), stream(3.2, ramp)], 0), /时间轴波动较大/);

// 两脚时钟差过大：应当拒绝
assert.throws(() => ankleTimebase([stream(3.2), stream(3.6)], 0), /时间轴差异过大/);

// 抖动只发生在单侧也不该被放过
assert.throws(() => ankleTimebase([stream(3.2, ramp), stream(3.2)], 0), /时间轴波动较大/);

// 包内已声明会话时间轴：接收时间怎么抖都不该影响对时，位移取 0，只保留「同轴」校验
{
  const tb = ankleTimebase([stream(3.2, spread), stream(3.21, ramp)], 0, true);
  assert.deepEqual(tb.shifts, [0, 0], '包内对齐时不做接收位移');
  assert.ok(Math.abs(tb.offset - 0.25) < 1e-6, '偏移仍取有效时间，实际 ' + tb.offset);
}

// 包内对齐 + 两路到会话轴的偏移不同（差 100 ms）⇒ 仍然拒绝，不能因为「声明已对齐」就放过
{
  const a = stream(3.2), b = stream(3.2);
  for (let i = 0; i < N; i++) b.elapsed[i] = b.t[i] + 350e3;
  assert.throws(() => ankleTimebase([a, b], 0, true), /同一条会话时间轴/);
}

// 包内对齐时也要求采样时间可用：缺 elapsed 仍应报错，而不是静默当成 0
{
  const s = stream(3.2);
  s.elapsed = new Float64Array(N).fill(NaN);
  assert.throws(() => ankleTimebase([s, stream(3.2)], 0, true), /无法映射到查看器时间轴/);
}

// 原点参与偏移：包内对齐时 offset 就是「有效时间偏移 − 原点」
{
  const tb = ankleTimebase([stream(3.2), stream(3.2)], 2.5e6, true);
  assert.ok(Math.abs(tb.offset - (0.25 - 2.5)) < 1e-6, '实际 ' + tb.offset);
}

console.log('ankle-clock: ok');
