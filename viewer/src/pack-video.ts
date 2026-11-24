// 包内落地对时：视频首帧在会话轴上的时刻由导出端写进 manifest.videos[]。
// 单独成模块，不依赖 parquet / DOM，便于用真实包做回归。

export interface PackVideoTiming {
  /** 首帧在会话轴上的时刻（µs） */
  firstUs: number;
  /** 导出端给出的对时不确定度（µs），没有就是 null */
  uncertaintyUs: number | null;
}

/** 视频时钟在首末帧之间允许的不一致；超过就说明 pts 与有效时间不是同一速度，不能采信。 */
const MAX_DRIFT_US = 5000;

export function packVideoTiming(manifest: any, path: string): PackVideoTiming | null {
  const list = Array.isArray(manifest?.videos) ? manifest.videos : [];
  const v = list.find((e: any) => e && e.path === path);
  if (!v || v.alignmentStatus !== 'aligned') return null;
  const first = Number(v.firstFrameEffectiveElapsedUs), last = Number(v.lastFrameEffectiveElapsedUs);
  const firstPts = Number(v.firstFrameVideoPtsUs), lastPts = Number(v.lastFrameVideoPtsUs);
  if (![first, last, firstPts, lastPts].every(Number.isFinite) || first < 0 || last <= first) return null;
  if (Math.abs(last - lastPts - (first - firstPts)) > MAX_DRIFT_US) return null;
  const unc = Number(v.uncertaintyUs);
  return {firstUs: first, uncertaintyUs: Number.isFinite(unc) ? unc : null};
}

/** 会话轴上的起点（秒）。允许为负：摄像头常比设备先开，视频 0 s 落在会话原点之前。 */
export function packVideoOffset(timing: PackVideoTiming | null, originUs: number): number | undefined {
  if (!timing || !Number.isFinite(originUs)) return undefined;
  return (timing.firstUs - originUs) / 1e6;
}
