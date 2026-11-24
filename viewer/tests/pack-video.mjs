import assert from 'node:assert/strict';
import {packVideoOffset, packVideoTiming} from '../src/pack-video.ts';

// 真实包（五个设备深蹲.zip）的数据：五台设备最早有效帧 2 688 079 µs，视频首帧 551 431 µs
const REAL = {
  path: 'videos/clip-001.mp4',
  alignmentStatus: 'aligned',
  firstFrameVideoPtsUs: 0,
  lastFrameVideoPtsUs: 26961611,
  firstFrameEffectiveElapsedUs: 551431,
  lastFrameEffectiveElapsedUs: 27513041,
  uncertaintyUs: 1058,
};
const ORIGIN_US = 2688079;
const manifest = videos => ({videos});

// 真实包：视频比设备早 2.137 s 开录，起点必须是负数，不能被夹到 0
{
  const t = packVideoTiming(manifest([REAL]), 'videos/clip-001.mp4');
  assert.ok(t, '已对齐的包应当采信');
  assert.equal(t.firstUs, 551431);
  assert.equal(t.uncertaintyUs, 1058);
  const off = packVideoOffset(t, ORIGIN_US);
  assert.ok(Math.abs(off - -2.136648) < 1e-6, '起点应为 −2.136648 s，实际 ' + off);
  // 覆盖区间落回会话轴后，视频末端应当在会话末端之前一点
  const end = off + (REAL.lastFrameVideoPtsUs - REAL.firstFrameVideoPtsUs) / 1e6;
  assert.ok(Math.abs(end - (REAL.lastFrameEffectiveElapsedUs - ORIGIN_US) / 1e6) < 2e-3, '末帧应落在会话轴同一位置');
}

// 多段视频时按路径匹配，不能套用第一段的时间
assert.equal(packVideoTiming(manifest([REAL]), 'videos/clip-002.mp4'), null);

// 未对齐的包没有可用起点
assert.equal(packVideoTiming(manifest([{...REAL, alignmentStatus: 'unmatched'}]), 'videos/clip-001.mp4'), null);

// 首末帧的「有效时间 − pts」不一致 ⇒ 视频时钟在漂，不能拿首帧当唯一基准
assert.equal(packVideoTiming(manifest([{...REAL, lastFrameEffectiveElapsedUs: 27513041 + 30000}]), 'videos/clip-001.mp4'), null);

// 字段缺失 / 顺序颠倒
assert.equal(packVideoTiming(manifest([{...REAL, firstFrameEffectiveElapsedUs: undefined}]), 'videos/clip-001.mp4'), null);
assert.equal(packVideoTiming(manifest([{...REAL, lastFrameEffectiveElapsedUs: REAL.firstFrameEffectiveElapsedUs - 1}]), 'videos/clip-001.mp4'), null);
assert.equal(packVideoTiming({}, 'videos/clip-001.mp4'), null);

// 视频晚于会话原点开录：正起点，与旧启发式的值域一致
{
  const t = packVideoTiming(manifest([{...REAL, firstFrameEffectiveElapsedUs: 5688079,
    lastFrameEffectiveElapsedUs: 27513041 + (5688079 - 551431)}]), 'videos/clip-001.mp4');
  assert.ok(Math.abs(packVideoOffset(t, ORIGIN_US) - 3) < 1e-6);
}

// 原点缺失时不给起点（宁可退回旧启发式，也不要算出 NaN 偏移）
assert.equal(packVideoOffset(packVideoTiming(manifest([REAL]), 'videos/clip-001.mp4'), NaN), undefined);
assert.equal(packVideoOffset(null, ORIGIN_US), undefined);

console.log('pack-video: ok');
