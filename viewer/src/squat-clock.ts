// 左右踝对时的判据单独成模块：不依赖模型权重，便于单元测试。

export interface AnkleStream { t: Float64Array; values: Float64Array; receive: Float64Array; elapsed: Float64Array; angles?: [Float64Array,Float64Array]; heading?: Float64Array }

const sorted = (x: ArrayLike<number>) => Float64Array.from(x).sort();
const quantile = (x: Float64Array, q: number) => {
  const k = q * (x.length - 1), i = Math.floor(k), f = k - i;
  return f < .5 ? x[i] + f * (x[Math.ceil(k)] - x[i]) : x[Math.ceil(k)] - (1 - f) * (x[Math.ceil(k)] - x[i]);
};
const median = (x: ArrayLike<number>) => quantile(sorted(x), .5);

function frameSanity(s: AnkleStream): void {
  if (s.t.length<30) throw new Error('每只脚踝至少需要 30 帧有效 IMU 数据。');
  for(let i=0;i<s.t.length;i++) {
    if (!Number.isSafeInteger(s.t[i]) || (i>0 && s.t[i]<=s.t[i-1])) throw new Error('踝部采样时间缺失或重复。');
  }
}

// 有效时间相对采样时间的常数偏移（秒）。包内已把两列映射到同一会话轴时它是常数 0。
function elapsedOffset(s: AnkleStream): number {
  const e: number[] = [];
  for(let i=0;i<s.t.length;i++) if (Number.isSafeInteger(s.elapsed[i])) e.push((s.elapsed[i]-s.t[i])/1e6);
  if (e.length<.95*s.t.length) throw new Error('采样时间无法映射到查看器时间轴。');
  const se=sorted(e);
  if (quantile(se,.95)-quantile(se,.05)>.02) throw new Error('会话时间和采样时间之间不是稳定偏移。');
  return quantile(se,.5);
}

export function summarizeClock(s: AnkleStream, receiveOnly=false) {  frameSanity(s);
  const v: number[] = [];
  for(let i=0;i<s.t.length;i++) if (Number.isSafeInteger(s.receive[i])) v.push((s.receive[i]-s.t[i]*1000)/1e9);
  if (v.length<30 || v.length<.95*s.t.length) throw new Error('缺少可用于左右踝对时的手机接收时间。');
  // 接收偏移的尾部只是个别 BLE 包晚到，对时取的是中位数，尾部再长也不影响它；
  // 真正会毁掉对时的是主体抖动和锚点本身在段间漂移。
  const m=v.length>>1, srt=sorted(v), drift=median(v.slice(m))-median(v.slice(0,m));
  const seg=v.length>>2;
  const meds=[0,1,2,3].map(k=>median(v.slice(k*seg, k===3 ? v.length : (k+1)*seg)));
  if (Math.abs(drift)>.06 || quantile(srt,.75)-quantile(srt,.25)>.06
    || Math.max(...meds)-Math.min(...meds)>.06) throw new Error('踝部时间轴波动较大，无法可靠对齐。');
  if (receiveOnly) return [quantile(srt,.5),NaN];
  return [quantile(srt,.5),elapsedOffset(s)];
}

/** 包内已对齐时两路之间的时间差：直接取有效时间偏移之差，不看 BLE 到达顺序。 */
export function elapsedDelta(a: AnkleStream, b: AnkleStream): number {
  frameSanity(a); frameSanity(b);
  return elapsedOffset(a) - elapsedOffset(b);
}

/** 包内已声明会话时间轴（manifest.sessionTimeline）：两路采样时间本就落在同一轴上，
    只校验「同轴」与偏移稳定，不再用 BLE 到达时间反推位移。 */
export function ankleTimebase(streams: [AnkleStream,AnkleStream], origin: number, aligned=false) {
  const clocks: [number,number][] = streams.map(s=>{
    if (!aligned) return summarizeClock(s) as [number,number];
    frameSanity(s);
    return [0,elapsedOffset(s)];
  });
  if (Math.abs(clocks[0][1]-clocks[1][1])>.02) throw new Error('左右踝无法映射到查看器的同一条会话时间轴。');
  if (aligned) return {shifts:[0,0], offset:(clocks[0][1]+clocks[1][1])/2-origin/1e6};
  const d=clocks[0][0]-clocks[1][0];
  if (Math.abs(d)>.12) throw new Error('左右踝的时间轴差异过大，无法可靠对齐。');
  const average=(clocks[0][0]+clocks[1][0])/2, shifts=clocks.map(d=>d[0]-average);
  const offset=(clocks[0][1]+clocks[1][1])/2-origin/1e6;
  return {shifts,offset};
}
