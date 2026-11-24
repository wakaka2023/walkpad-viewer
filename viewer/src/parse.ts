/* parquet / 维特 TXT 读列、按侧分组、展开成通道数组。 */

import {ParquetHQ} from './data/parquet';
import {headingCursor} from './data/heading';
import {spreadNotified} from './data/notification-time';
import {renderAll, setTab, updateChips} from './app';
import {GRID, NCH, SIDE_CN} from './core/constants';
import type {AxisSupport} from './axis/offline';
import {el} from './core/dom';
import {state} from './core/state';
import {blkOf, clearErr, fail, niceMax} from './core/util';
import {ictoReset} from './icto/view';
import {clearOvCache, redrawImu} from './imu';
import {lodDone, lodFeed, lodInit, useLod} from './lod';
import {packVideoOffset, packVideoTiming} from './pack-video';
import {drawScrub, ovFeed, ovInitAll} from './scrub';
import {redrawWave, updateWinBar} from './timewin';
import type {PkgFile} from './types';
import {attachVideo, clearVideo, verifiedVideoOffset} from './video';

// ---------------------------------------------------------------- 解析
// 意外异常只报「一句话 + 哪个文件哪一行」：整段栈太长，但只报英文消息又没法定位
function errLoc(e) {
  const st = (e && e.stack) ? String(e.stack) : '';
  for (const line of st.split(/\r?\n/)) {
    const m = /([\w.\/\-]+\.js)(?:\?[^:]*)?:(\d+):\d+/.exec(line);
    if (m) return ' @' + m[1].split('/').pop() + ':' + m[2];
  }
  return '';
}

// 读列与展开通道都是同步阻塞的，先让遮罩画出来再开工
async function parse(file: File) {
  el.busy.hidden = false;
  clearVideo();
  await new Promise(r => setTimeout(r, 30));
  try { await parseFile(file); }
  // 解析里的意外异常（而不是显式 fail）也得让人看见：否则页面只是「什么都没发生」
  catch (e) { fail('解析失败：' + ((e && e.message) || e) + errLoc(e)); }
  finally { el.busy.hidden = true; }
}

async function parseFile(file) {
  clearErr();
  const name = (file.name || '').toLowerCase();
  if (name.endsWith('.zip')) return parsePackage(file);
  if (name.endsWith('.parquet')) return parseParquet(file);
  if (name.endsWith('.txt') || name.endsWith('.csv')) return parseWit(file);
  fail('支持 .zip（测量包导出）、.parquet 与维特 IMU 导出的 .txt / .csv。');
}

// ---------------------------------------------------------------- parquet
// 侧别名归一化。新版文件 assigned_side 全空、改在 placement_code 写 left_instep / right_instep，
// 而 GRID / SIDE_CN 只认 left / right —— 不归一化的话两脚会被并成一个 'unknown' 侧，
// 镜像布局也失配（左右脚 56 格的坐标表是两套）。
function sideOf(r) {
  const raw = r.assigned_side || r.placement_code || '';
  const m = /^(left|right)/i.exec(String(raw));
  return m ? m[1].toLowerCase() : (String(raw) || 'unknown');
}

// 帧时间。⚠ 必须先判 null 再转：Number(null) 是 0 而不是 NaN，会把缺时间的帧全堆到时间轴起点。
function numOrNaN(v) {
  if (v == null) return NaN;
  const n = Number(v);
  return Number.isFinite(n) ? n : NaN;
}

async function parseParquet(file) {

  let ab;
  try { ab = await file.arrayBuffer(); } catch (e) { fail('读取文件失败：' + e.message); return; }

  let md, rows, srcInfo = null;
  // 温度列名要在 try 外面用（逐行取值），所以声明在块外
  let tempCol = null;
  // 侧别列有两代写法：老版在 assigned_side 里写 left / right；新版那一列留空，
  // 改在 placement_code 里写 left_instep / right_instep。两列都按「有则取」处理。
  const need = ['effective_elapsed_us', 'pressure_values'];
  try {
    md = ParquetHQ.parquetMetadata(ab);
    const names = new Set((md.schema || []).map(s => s.name));
    const miss = need.filter(c => !names.has(c));
    const sideCols = ['assigned_side', 'placement_code'].filter(c => names.has(c));
    if (!sideCols.length) miss.push('assigned_side 或 placement_code');
    if (miss.length) { fail('这个 parquet 缺少必需列：' + miss.join('、')); return; }
    // IMU 列不是每份导入都有，有才读；缺了就只少一个页签，不影响压力三页。
    // physical（已换算成 g / °/s）与 raw（硬件 LSB 计数）两套并列读，页面上可切换着看。
    const imuCols = ['imu_physical_values', 'imu_raw_values', 'imu_unit_status'].filter(c => names.has(c));
    // 帧序号同样是有才读：丢包页靠它算缺口，缺了只是少一个页签。
    // unwrapped 已处理 uint32 回绕，优先；ordinal 兜底。
    const seqCols = ['source_frame_number_unwrapped', 'source_frame_ordinal'].filter(c => names.has(c));
    // 有效帧时间缺失时的退路（少数未映射到会话时间轴的边界帧）
    const tAlt = ['observed_session_elapsed_us'].filter(c => names.has(c));
    // 文件自称的格式版本：只用来在顶栏标出这份文件是哪一代，不参与解析分支
    const verCols = ['schema_version'].filter(c => names.has(c));
    // 温度：app 5.x 叫 gyro_temperature_celsius，网页自己 1.x 导出叫 temperature_c —— 两个名字都认
    tempCol = ['gyro_temperature_celsius', 'temperature_c'].find(c => names.has(c)) || null;
    // IC/TO 检测要的输入：包序号、时间映射状态、对齐时间。这三列老导出没有，缺了只是不给 IC/TO 页签，
    // 不影响绘图 —— 与其它列一样按列名探测，不做 schema_version 分支。
    const ictoCols = ['source_native_sequences', 'effective_time_status',
      'aligned_session_elapsed_us', 'source_frame_ordinal'].filter(c => names.has(c));
    const cols = need.concat(sideCols, tAlt, imuCols, seqCols, verCols, ictoCols, tempCol ? [tempCol] : []);
    // 顶层列数取 root 的 num_children：schema 是拍平列表，数名字会把 LIST 的 list/element 也算进去
    const ncols = (md.schema && md.schema[0] && md.schema[0].num_children) || null;
    rows = await ParquetHQ.parquetReadObjects({
      file: ab, compressors: ParquetHQ.compressors, columns: cols
    });
    srcInfo = {ver: (rows[0] && rows[0].schema_version) || null, ncols, tempCol};
  } catch (e) {
    fail('解析失败：' + (e && e.message ? e.message : e) + errLoc(e));
    return;
  }

  const groups = new Map();
  for (const r of rows) {
    const pv = r.pressure_values;
    if (!pv) continue;
    const ph0 = r.imu_physical_values, rw0 = r.imu_raw_values;
    // 5.x 起同一份导出里混着磁力计 / 四元数帧：imu_* 与 pressure_* 全空，
    // 只带 magnetic_raw_values / device_quaternion_wxyz。这类行既画不出波形，
    // 也不该进帧数与采样率 —— 否则 50 Hz 的 IMU 会被摊成 63.7 Hz。
    const hasP = typeof pv.length === 'number' && pv.length > 0;
    const hasI = (!!ph0 && ph0.length > 0) || (!!rw0 && rw0.length > 0);
    if (!hasP && !hasI) continue;
    const side = sideOf(r);
    let g = groups.get(side);
    if (!g) { g = { eff: [], obs: [], v: [], iph: [], irw: [], ist: [], fno: [], tmp: [], st: [], ns: [], al: [], sfo: [] }; groups.set(side, g); }
    // 两种时间都留着：选哪一种在按侧建数组时定（见下面 src 的说明）
    g.eff.push(numOrNaN(r.effective_elapsed_us));
    g.obs.push(numOrNaN(r.observed_session_elapsed_us));
    g.v.push(hasP ? pv : []);   // 只有 IMU 的帧：压力通道按空处理，下面按 length 兜空值
    g.tmp.push(tempCol ? numOrNaN(r[tempCol]) : NaN);
    // IC/TO 用：映射状态决定该行是有效样本还是分段边界；包序号倒退/跳号也要切段
    g.st.push(r.effective_time_status || null);
    g.al.push(numOrNaN(r.aligned_session_elapsed_us));
    const nseq = r.source_native_sequences;
    const n0 = (nseq && typeof nseq.length === 'number' && nseq.length) ? nseq[0] : null;
    g.ns.push(n0 == null ? NaN : Number(n0));
    g.sfo.push(numOrNaN(r.source_frame_ordinal));
    g.iph.push(ph0 && typeof ph0.length === 'number' ? ph0 : null);
    g.irw.push(rw0 && typeof rw0.length === 'number' ? rw0 : null);
    g.ist.push(r.imu_unit_status || null);
    // int64 可能是 BigInt，Number() 才转得动
    const fn = r.source_frame_number_unwrapped != null ? r.source_frame_number_unwrapped : r.source_frame_ordinal;
    g.fno.push(fn == null ? NaN : Number(fn));
  }
  if (!groups.size) { fail('文件里没有可用的帧：压力列全为空，或帧时间全都取不到。'); return; }
  const sides = expandSides(groups);
  if (!sides.length) { fail('文件里没有可用的帧：每侧都取不到帧时间。'); return; }
  rebaseCommon(sides);
  applySides(sides, file.name, 'pressure', {rebuilt: false, src: srcInfo});
}

// 一组按侧分好的行 → 每侧一套定长数组。旧 parquet 与新版测量包共用这一段：
// 新包先把 camelCase 列映射成同样的字段（eff / v / iph …）再进来，
// 否则两代格式各写一套时间与丢帧口径，同一个文件在两代里会给出不同帧数。
function expandSides(groups) {
  const sides = [];
  for (const [side, g] of groups) {
    // effective（设备日历映射到会话轴）与 observed（页面收到帧的时刻）是两个基准，
    // 实测相差 0.7~0.9 s 且随会话缓慢漂移，逐帧混用等于把两种时间拼在一条轴上。
    // 所以：一侧里只要有任意一帧拿到有效帧时间，就只认它，其余帧丢弃并计数；
    // 整侧一个都没有时才整侧退回到达时刻 —— 此时基准统一，相对时序照常可用。
    const src = g.eff.some(Number.isFinite) ? g.eff : g.obs;
    const order = [];
    let nDrop = 0;
    for (let i = 0; i < src.length; i++) {
      if (Number.isFinite(src[i])) order.push(i); else nDrop++;
    }
    const n = order.length;
    if (!n) continue;                                   // 整组都没取到时间，没什么可画
    order.sort((a, b) => src[a] - src[b]);
    const nch = Math.min(NCH, Math.max(...order.slice(0, 200).map(i => g.v[i].length)));
    const tt = spreadNotified(Float64Array.from(order, i => src[i]));
    const t0 = tt[0];
    const t = new Float64Array(n);
    const ch = Array.from({length: nch}, () => new Float32Array(n));
    const mx = new Float32Array(nch);
    const mn = new Float32Array(nch).fill(Infinity);   // 最小非零读数，用于说明 FSR 阈值
    const zc = new Float32Array(nch);   // 读数为 0 的帧数（只算有值的帧）
    const miss = new Float32Array(nch); // 缺失帧数（null / undefined / 非数值）
    const fno = new Float64Array(n).fill(NaN);   // 帧序号，丢包页按它的缺口算
    // 温度：整列缺就留 null（老 62 列导出没有温度），有值时顺手记下全段范围给纵轴用
    const tp = new Float64Array(n).fill(NaN);
    let tLo = Infinity, tHi = -Infinity;
    let hasSeq = false;
    for (let k = 0; k < n; k++) {
      const i = order[k];
      t[k] = (tt[k] - t0) / 1e6;
      if (Number.isFinite(g.fno[i])) { fno[k] = g.fno[i]; hasSeq = true; }
      const tv = g.tmp[i];
      if (Number.isFinite(tv)) { tp[k] = tv; if (tv < tLo) tLo = tv; if (tv > tHi) tHi = tv; }
      const pv = g.v[i];
      for (let j = 0; j < nch; j++) {
        const rv = pv[j];
        // int64 列读出来是 BigInt，不能用 + / isFinite 直接转（会抛 TypeError）
        const val = typeof rv === 'number' ? rv : (rv == null ? NaN : Number(rv));
        // 缺值要写成 NaN 而不是留 0：数组初值是 0，不写就等于把「没有读数」画成「实测为零」
        if (!Number.isFinite(val)) { ch[j][k] = NaN; miss[j]++; continue; }
        ch[j][k] = val;
        if (val > mx[j]) mx[j] = val;
        if (val > 0 && val < mn[j]) mn[j] = val;
        if (val === 0) zc[j]++;
      }
    }
    let dt = 0;
    for (let k = 1; k < n; k++) dt += t[k] - t[k - 1];

    // IMU 与压力同帧同序：每帧 6 个，前 3 个加速度、后 3 个角速度。
    // physical 已换算成 g / °/s；raw 是硬件 LSB 计数，两者通道序不同（raw 的 1/2 对应物理 Y/X 且 Y 反号）。
    const IMU_N = 6;
    const mkCh = () => Array.from({length: IMU_N}, () => new Float32Array(n).fill(NaN));
    const iph = mkCh(), irw = mkCh();
    let hasPhys = false, hasRaw = false, nPart = 0;
    for (let k = 0; k < n; k++) {
      const i = order[k];
      const pv = g.iph[i], rv = g.irw[i];
      if (pv && pv.length >= IMU_N) {
        hasPhys = true;
        for (let j = 0; j < IMU_N; j++) {
          const x = pv[j];
          const val = typeof x === 'number' ? x : (x == null ? NaN : Number(x));
          if (Number.isFinite(val)) iph[j][k] = val;
        }
      }
      if (rv && rv.length >= IMU_N) {
        hasRaw = true;
        for (let j = 0; j < IMU_N; j++) {
          const x = rv[j];
          const val = typeof x === 'number' ? x : (x == null ? NaN : Number(x));
          if (Number.isFinite(val)) irw[j][k] = val;
        }
      }
      const st = g.ist[i];
      if (st && st !== 'confirmed' && st !== 'absent') nPart++;
    }
    const imu = (hasPhys || hasRaw)
      ? {nch: IMU_N, phys: hasPhys ? iph : null, raw: hasRaw ? irw : null, n, nPart}
      : null;
    // IC/TO 的输入按文件行序排（不是绘图用的时间序）：算法要求同脚源帧序号严格递增，
    // 而且要靠 unmapped 行与包序号倒退切分段，所以不能沿用上面按时间过滤后的数组。
    let ictoIn = null;
    if (hasPhys) {
      const nr = g.eff.length;
      const inp = {n: nr, tUs: new Float64Array(nr), seq: new Float64Array(nr),
        ord: new Float64Array(nr), ok: new Uint8Array(nr), ax: new Float64Array(nr * 6),
        row: new Int32Array(nr), sfo: new Float64Array(nr)};
      let prevT = NaN, nBad = 0;
      for (let i = 0; i < nr; i++) {
        const st = g.st[i], eff = g.eff[i], al = g.al[i];
        // 老导出没有 effective_time_status 列：取到有效帧时间就算 mapped
        const mapped = st ? st === 'mapped' : Number.isFinite(eff);
        inp.ok[i] = mapped && Number.isFinite(eff) ? 1 : 0;
        let tv = Number.isFinite(al) ? al : (Number.isFinite(eff) ? eff
          : (Number.isFinite(g.obs[i]) ? g.obs[i] : prevT + 1));
        if (!Number.isFinite(tv)) tv = 0;
        inp.tUs[i] = tv; prevT = tv;
        inp.ord[i] = i; inp.row[i] = i; inp.sfo[i] = g.sfo[i];
        // 包序号优先用真实计数器；没有就退到帧号，再没有才用行序（行序永远连号，等于不分段）
        inp.seq[i] = Number.isFinite(g.ns[i]) ? g.ns[i] : (Number.isFinite(g.fno[i]) ? g.fno[i] : i);
        const ph = g.iph[i];
        for (let j = 0; j < IMU_N; j++) {
          const x = ph ? ph[j] : null;
          const v = typeof x === 'number' ? x : (x == null ? NaN : Number(x));
          if (!Number.isFinite(v)) nBad++;
          inp.ax[i * 6 + j] = v;
        }
      }
      // 六轴里有空值就不给 IC/TO：算法全程用插值前的原始值，NaN 会一路污染到门限
      if (!nBad) ictoIn = inp;
    }
    const support = g.axisPressure ? collectAxisSupport(side, g.v) : undefined;
    const activityIn = {t: Float64Array.from(spreadNotified(g.eff), (v: number) => (v - t0) / 1e6),
      support,
      deviceAngles: deviceAngleArrays(g.deviceAngles),
      seq: Float64Array.from(g.ns.map((v, i) => Number.isFinite(v) ? v : g.fno[i])),
      ok: Uint8Array.from(g.eff, (v, i) => Number.isFinite(v) && (!g.st[i] || g.st[i] === 'mapped') && g.ist[i] === 'confirmed' ? 1 : 0),
      axes: Array.from({length: 6}, (_, j) => Float32Array.from(g.iph, ph => numOrNaN(ph?.[j])))};
    const temp = tHi >= tLo ? tp : null;
    sides.push({ activityIn, side, n, nch, t, ch, mx, mn, zc, miss, imu, fno: hasSeq ? fno : null,
                 temp, tLo, tHi, t0, ictoIn, nT: nDrop, cal: g.cal, dur: t[n - 1], span: t[n - 1],
                 fs: n > 1 ? 1 / (dt / (n - 1)) : NaN,
                 // 绘图一律读 blk：整段载入时与上面是同一批数组，懒加载时只含窗口内的帧
                 blk: {n, t, ch, imu, temp, ictoIn, a: t[0], b: t[n - 1],
                       deviceAngles: deviceAngleArrays(g.deviceAngles)} });
  }
  sides.sort((a, b) => a.side.localeCompare(b.side));
  return sides;
}

// 同一份记录里各来源共用会话时间原点。expandSides / buildLazySide 各自减去「自己首帧」的
// 绝对时间，会把左右脚之间真实存在的偏移抹成 0（实测两份包分别是 37 ms 与 24 ms）。
// 这里统一减去整份数据最早的一帧：时间轴仍从 0 起，来源之间的相对偏移保留下来。
// 注意这只是「同一文件内的时间口径」，不等于两份独立文件之间的同步，也不等于硬件绝对同步。
function rebaseCommon(sides) {
  let t0 = Infinity;
  for (const s of sides) if (Number.isFinite(s.t0) && s.t0 < t0) t0 = s.t0;
  if (!Number.isFinite(t0)) return;
  const done = new Set();
  const shiftArr = (arr, d) => {
    if (!arr || done.has(arr)) return;
    done.add(arr);
    for (let k = 0; k < arr.length; k++) arr[k] += d;
  };
  for (const s of sides) {
    const d = (s.t0 - t0) / 1e6;
    if (s.activityIn) shiftArr(s.activityIn.t, d);
    if (!d) { s.t0 = t0; continue; }
    shiftArr(s.t, d);
    if (s.blk) { shiftArr(s.blk.t, d); s.blk.a += d; s.blk.b += d; }
    if (Number.isFinite(s.dur)) s.dur += d;
    s.t0 = t0;
  }
}
// ---------------------------------------------------------------- 测量包（zip）
// 新一版导出改成整包：manifest.json + timeline.json + frames/<设备目录>/frame-*.parquet + evidence/*。
// 帧表换成 camelCase、IMU 拆成 struct、侧别只写在 manifest 里，
// 所以先按 sourceId 找到各自的帧文件，把新列映射成 expandSides 认的那套字段再走同一条展开。
// 56 格布局与旧包同源（d56-14x5-56-side-mirrored-candidate-v3），core.js 的 GRID 直接可用。
const DEVICE_ANGLE_COLS = ['pitchDeg', 'sideTiltDeg'];
const HEADING_COLS = ['relativeHeadingDeg', 'referenceId'];
const PKG_COLS = ['effectiveElapsedUs', 'timeUs', 'validity', 'frameType', 'pressureAdc', 'accelerationG',
  'angularVelocityDps', 'accelerationRaw', 'angularVelocityRaw', 'deviceSequence', 'temperatureCelsius',
  'imuCalibration', ...DEVICE_ANGLE_COLS, ...HEADING_COLS];
const SQUAT_COLS = ['timeUs', 'effectiveElapsedUs', 'nativeReceiveNs', 'validity', 'accelerationG', 'imuCalibration'];
const SQUAT_HEAD_COLS = HEADING_COLS;
const WRIST_RAW_COLS = SQUAT_COLS.map(c=>c==='imuCalibration' ? 'angularVelocityDps' : c);
type SquatFile = PkgFile & {squat: string[]};
const squatFiles = new WeakMap<object, [SquatFile[],SquatFile[]]>();
const squatMeta = new WeakMap<object, {recordId: string|null; wrist?: {files: SquatFile[]; placement: string; family: string}; aligned?: boolean}>();

export function squatPackage(sides: object) { return squatFiles.get(sides); }
export function squatCaptureMeta(sides: object) { return squatMeta.get(sides); }

function collapseWristBatches(rows: number[][]) {
  const out: number[][]=[];
  for(let i=0,j;i<rows.length;i=j) {
    const r=rows[i];
    for(j=i+1;j<rows.length && r[0]===rows[j][0] && r[1]===rows[j][1] && r[2]===rows[j][2];j++);
    // 维特同一 BLE 通知可含两帧但无独立时间戳，只合并同批，不臆造间隔。
    const batch=rows.slice(i,j);
    out.push(batch.length===1 ? r : r.map((v,c)=>c<3 ? v : batch.reduce((s,x)=>s+x[c],0)/batch.length));
  }
  return out;
}

export async function squatInput(files: SquatFile[], alive: () => boolean, mode: 'ankle'|'wrist-comp'|'wrist-raw'='ankle') {
  const rows: number[][] = [], required=mode==='wrist-raw' ? WRIST_RAW_COLS : SQUAT_COLS;
  const hasAngles=mode==='ankle' && files.every(f=>DEVICE_ANGLE_COLS.every(c=>f.cols.includes(c)));
  const hasHead=mode==='ankle' && files.every(f=>SQUAT_HEAD_COLS.every(c=>f.squat.includes(c)));
  const columns=[...required,...(hasAngles?DEVICE_ANGLE_COLS:[]),...(hasHead?SQUAT_HEAD_COLS:[])];
  for (const f of files) {
    if (!alive()) throw new Error('读取已取消。');
    if (required.some(c => !f.squat.includes(c))) throw new Error(mode==='ankle' ? '缺少双踝时间戳、加速度或补偿角速度列。' : '腕部缺少时间戳或所需六轴列。');
    for (const [rowStart, rowEnd] of f.rgs) {
      if (!alive()) throw new Error('读取已取消。');
      const part = await ParquetHQ.parquetReadObjects({file: f.ab, columns, rowStart, rowEnd});
      for (const r of part) {
        if (r.validity !== 'valid' || r.accelerationG == null) continue;
        const gyro = mode==='wrist-raw' ? r.angularVelocityDps : r.imuCalibration?.angularVelocityCompensatedDps;
        const v = struct6(r.accelerationG, gyro);
        if (!gyro || !v.every(Number.isFinite)) {
          if (mode==='ankle') throw new Error('双踝数据有帧缺少补偿角速度或有效加速度，不能用原始角速度代替。');
          continue;
        }
        rows.push([numOrNaN(r.timeUs), numOrNaN(r.nativeReceiveNs), numOrNaN(r.effectiveElapsedUs), ...v,
          ...(hasAngles ? [numOrNaN(r.pitchDeg),numOrNaN(r.sideTiltDeg)] : []),
          ...(hasHead ? [numOrNaN(r.relativeHeadingDeg), r.referenceId] : [])]);
      }
    }
  }
  rows.sort((a,b)=>a[0]-b[0]);
  const samples=mode==='wrist-raw' ? collapseWristBatches(rows) : rows;
  const axis = (c: number) => Float64Array.from(samples, r=>r[c]);
  const values = new Float64Array(samples.length*6);
  samples.forEach((r,i)=>{for(let j=0;j<6;j++) values[i*6+j]=r[3+j];});
  const aOff=hasAngles ? 9 : -1, hOff=hasAngles ? 9+DEVICE_ANGLE_COLS.length : 9;
  let head: Float64Array;
  if (hasHead) {
    const cur=headingCursor();
    head=Float64Array.from(samples, r=>cur(numOrNaN(r[hOff] as number), r[hOff+1] as unknown as string));
  }
  return {t:axis(0),receive:axis(1),elapsed:axis(2),values,
    angles:hasAngles ? [axis(aOff),axis(aOff+1)] as [Float64Array,Float64Array] : undefined,
    heading:head};
}
const hasDeviceAngles = (files, cols = DEVICE_ANGLE_COLS) => files.some(f => cols.some(c => f.cols.includes(c)));
const collectAngles = files => {
  const pitch = hasDeviceAngles(files), head = hasDeviceAngles(files, HEADING_COLS);
  if (!pitch && !head) return undefined;
  return {pitch: [], sideTilt: [], heading: [], head: headingCursor()};
};
function appendDeviceAngles(angles, r) {
  if (!angles) return;
  angles.pitch.push(numOrNaN(r.pitchDeg));
  angles.sideTilt.push(numOrNaN(r.sideTiltDeg));
  if (angles.heading) angles.heading.push(angles.head(numOrNaN(r.relativeHeadingDeg), r.referenceId));
}
function angleArr(list) {
  const a = Float64Array.from(list);
  return a.some(Number.isFinite) ? a : null;
}
function deviceAngleArrays(angles) {
  if (!angles) return null;
  return {pitch: Float64Array.from(angles.pitch), sideTilt: Float64Array.from(angles.sideTilt),
    heading: angles.heading ? angleArr(angles.heading) : null};
}
const IMU_COLS = ['accelerationG', 'angularVelocityDps', 'accelerationRaw', 'angularVelocityRaw', 'imuCalibration'];

// 一次导出里除 inertial / insole 外还夹着 magnetic 与 deviceQuaternion 帧：它们既没有压力也没有六轴，
// 计进帧数与采样率就会把 50 Hz 的惯性流报成 70 多 Hz。短包按「有没有本页要画的量」剔，
// 长包不退化为逐列读全表，按帧类型剔 —— 两条路都要剔，否则同一个文件换条路就换个帧率。
const SKIP_TYPE = {magnetic: 1, deviceQuaternion: 1};
const isSignalType = ft => !ft || !SKIP_TYPE[ft];

// 压力与步态沿用 left/right；同侧已有鞋垫时，脚踝另用独立键。
const FOOT_PLACE = /(insole|ankle|foot|instep|shoe)/;
const PLACE_CN = {insole: '鞋垫', ankle: '踝', foot: '足', instep: '足背', shoe: '鞋',
  wrist: '腕', hand: '手', arm: '臂', waist: '腰', hip: '髋', chest: '胸', back: '背',
  thigh: '大腿', knee: '膝', shank: '小腿', calf: '小腿', head: '头'};
function sideOfPlacement(p) {
  const k = String(p || '').toLowerCase();
  if (FOOT_PLACE.test(k)) return k.startsWith('right') ? 'right' : k.startsWith('left') ? 'left' : 'unknown';
  return k || 'unknown';
}

// 非足部位置的显示名：manifest 里只有 leftWrist 这种驼峰英文，面板标题直接印英文不便于现场核对。
// 名字注册进 SIDE_CN，各处仍只读 SIDE_CN，不必逐处判断流是不是脚。
function registerSideLabel(key, src) {
  if (key === 'left' || key === 'right') return;
  if (SIDE_CN[key]) return;
  const k = String(src.placement || '').toLowerCase();
  const seg = Object.keys(PLACE_CN).find(x => k.indexOf(x) >= 0);
  const side = k.startsWith('left') ? '左' : k.startsWith('right') ? '右' : '';
  SIDE_CN[key] = seg ? side + PLACE_CN[seg] : (src.bluetoothName || k || '未标注位置');
}

// 极简 zip 读取：只认 store / deflate，解压交给浏览器自带的 DecompressionStream（file:// 下也可用）。
// ZIP64（>4 GB 或 >65535 条目）不支持，测量包十几 MB，碰不到。
async function unzip(ab) {
  const dv = new DataView(ab), u8 = new Uint8Array(ab), n = ab.byteLength;
  let eocd = -1;
  for (let i = n - 22; i >= Math.max(0, n - 22 - 65535); i--) {
    if (dv.getUint32(i, true) === 0x06054b50) { eocd = i; break; }
  }
  if (eocd < 0) throw new Error('不是有效的 zip（找不到目录结尾记录）');
  if (dv.getUint16(eocd + 10, true) === 0xFFFF || dv.getUint32(eocd + 12, true) === 0xFFFFFFFF) {
    throw new Error('压缩包用到 ZIP64，暂不支持');
  }
  const cnt = dv.getUint16(eocd + 10, true);
  let p = dv.getUint32(eocd + 16, true);
  const td = new TextDecoder('utf-8'), out = new Map();
  for (let i = 0; i < cnt; i++) {
    if (dv.getUint32(p, true) !== 0x02014b50) break;
    const method = dv.getUint16(p + 10, true);
    const csize = dv.getUint32(p + 20, true);
    const nlen = dv.getUint16(p + 28, true);
    const elen = dv.getUint16(p + 30, true);
    const clen = dv.getUint16(p + 32, true);
    const lho = dv.getUint32(p + 42, true);
    const name = td.decode(u8.subarray(p + 46, p + 46 + nlen));
    p += 46 + nlen + elen + clen;
    if (!name || name.endsWith('/')) continue;
    // 数据起点按本地头自己的 name/extra 长度算，中心目录里的长度不一定与本地头一致
    const start = lho + 30 + dv.getUint16(lho + 26, true) + dv.getUint16(lho + 28, true);
    out.set(name, {method, raw: u8.subarray(start, start + csize)});
  }
  return out;
}

async function unzipBytes(ent) {
  if (!ent) return null;
  if (ent.method === 0) return ent.raw;
  if (ent.method !== 8) throw new Error('压缩包里用了不认识的压缩方式（' + ent.method + '）');
  const ds = new DecompressionStream('deflate-raw');
  const w = ds.writable.getWriter();
  w.write(ent.raw);
  w.close();
  return new Uint8Array(await new Response(ds.readable).arrayBuffer());
}

// hyparquet 只吃独立 ArrayBuffer，而解压结果是原 buffer 上的 subarray 视图
async function parquetBuffer(zf, path) {
  const u8 = await unzipBytes(zf.get(path));
  if (!u8) throw new Error('压缩包里没有 ' + path);
  return u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength);
}

// 新包一行 -> 旧展开逻辑认的字段。validity 非 valid 的行（段首前 / 段尾后）没有有效帧时间，
// st 置空、eff 保持 NaN，交给 expandSides 按「时间取不到」丢掉并计数。
// ⚠ 同一次导出里混着 magnetic / deviceQuaternion 帧：它们既没有压力也没有六轴，
// 混进来会把 50 Hz 的惯性流摊成 70 多 Hz，所以只留本查看器真能画的两种行。
// 返回 false 的行不占帧数、不占下标。
function pushPkgRow(g, r, i) {
  if (!isSignalType(r.frameType)) return false;
  const gy = gyroOf(r);
  const ph = struct6(r.accelerationG, gy);
  const rw = struct6(r.accelerationRaw, r.angularVelocityRaw);
  // 口径计数：扣偏值是新建对象，回落的仍是原列引用，据此区分，不必再判一次
  if (gy && gy !== r.angularVelocityDps) g.cal.ready++; else if (gy) g.cal.raw++;
  const pa = (r.pressureAdc && r.pressureAdc.length) ? r.pressureAdc : null;
  if (!ph && !rw && !pa) return false;
  const ok = r.validity === 'valid', eff = numOrNaN(r.effectiveElapsedUs);
  g.eff.push(ok ? eff : NaN);
  // 段首前 / 段尾后：timeUs 虽是这一行的时间，但它不是本段的有效样本，
  // 不能拿来给整段都 invalid 的包当回退时间轴（那会把边界帧画进波形）
  g.obs.push(ok ? numOrNaN(r.timeUs) : NaN);
  g.v.push(pa || []);
  g.tmp.push(numOrNaN(r.temperatureCelsius));
  g.st.push(ok ? 'mapped' : null);
  g.al.push(ok ? eff : NaN);
  g.ns.push(numOrNaN(r.deviceSequence));    // 包序号：IC/TO 靠它切分段
  g.fno.push(numOrNaN(r.deviceSequence));   // 丢包页按缺口算；新序号本身单调、不回绕
  g.sfo.push(i);
  g.iph.push(ph);
  g.irw.push(rw);
  g.ist.push(ph ? 'confirmed' : null);
  appendDeviceAngles(g.deviceAngles, r);
  return true;
}

// IMU 在新包里是两组 struct（{x,y,z}），页面按「前 3 加速度、后 3 角速度」的 6 通道摆。
// physical 已是输出轴序（y,-x,z），raw 是寄存器原序，与旧文件同名两套的对应关系一致。
function struct6(a, b) {
  if (!a && !b) return null;
  const f = o => o ? [numOrNaN(o.x), numOrNaN(o.y), numOrNaN(o.z)] : [NaN, NaN, NaN];
  return f(a).concat(f(b));
}

// 角速度口径：该帧带扣偏结果就用扣偏值，没有（老包 / 未标定帧）才用采集端换算的原始值。
// 判定按帧走，不看整包：同一段记录里前几秒常在等零偏就绪。
// 原始值本身不改写，仍在 angularVelocityRaw 与本列里。
const gyroOf = r => {
  const c = r.imuCalibration;
  const v = c && c.status === 'ready' ? c.angularVelocityCompensatedDps : null;
  return v ? {x: numOrNaN(v.x), y: numOrNaN(v.y), z: numOrNaN(v.z)} : r.angularVelocityDps;
};
const phys6 = r => struct6(r.accelerationG, gyroOf(r));

// 测量包可以带一段现场视频：优先按 manifest 里声明的 videos/*，没有再扫包里的同名后缀。
// 视频坏了不该连波形一起废掉，所以这里只返回 null，错误不外抛。
const VIDEO_RE = /\.(mp4|m4v|mov|webm)$/i;
async function packageVideo(zf, manifest) {
  const declared = (manifest.outputs || []).map(o => o && o.path).filter(p => p && VIDEO_RE.test(p));
  const all = declared.length ? declared : [...zf.keys()].filter(p => VIDEO_RE.test(p));
  if (!all.length) return null;
  const [path] = all;
  const u8 = await unzipBytes(zf.get(path));
  return u8 && u8.byteLength ? {u8, path, name: path.split('/').pop(), total: all.length} : null;
}

async function parsePackage(file) {
  if (typeof DecompressionStream !== 'function') {
    fail('这个浏览器不支持页面内解压（DecompressionStream），请改用 Edge / Chrome。');
    return;
  }

  let zf, manifest, archive: ArrayBuffer;
  try {
    archive = await file.arrayBuffer();
    zf = await unzip(archive);
    const m = await unzipBytes(zf.get('manifest.json'));
    if (!m) {
      const logs = [...zf.keys()].filter(p => /\.jsonl$/i.test(p));
      if (logs.length === 1) return parseWtJournal(await unzipBytes(zf.get(logs[0])), file.name);
      fail('压缩包里没有 manifest.json，也没有唯一的 WT 本地版采集记录。'); return;
    }
    manifest = JSON.parse(new TextDecoder('utf-8').decode(m));
  } catch (e) { fail('读取压缩包失败：' + ((e && e.message) || e)); return; }

  let media = null;
  try { media = await packageVideo(zf, manifest); } catch { media = null; }

  const srcs = Array.isArray(manifest.sources) ? manifest.sources : [];
  if (!srcs.length) { fail('manifest 里没有 sources 段，判断不了每份帧文件属于哪只脚。'); return; }

  // 帧文件按 sourceId 归组：outputs 里同时给了 sourceId 与 path，比按目录名猜稳
  const bySrc = new Map();
  for (const o of manifest.outputs || []) {
    if (!o || !o.sourceId || !/^frames\/.+\.parquet$/i.test(String(o.path || ''))) continue;
    if (!bySrc.has(o.sourceId)) bySrc.set(o.sourceId, []);
    bySrc.get(o.sourceId).push(o.path);
  }

  const srcInfo = {ver: manifest.schemaVersion || null, ncols: null, tempCol: null};
  // 先把每只脚的帧文件与 footer 摸清：行数、row group 边界、列名。
  // 帧数超过阈值的走分段懒加载，短片段仍旧整段读进来 —— 两条路的数据形状一致，绘图不必分支。
  const jobs = [];
  const pressureSides = new Set(srcs.filter(s => Number(s.pressureChannelCount) > 0).map(s => sideOfPlacement(s.placement)));
  try {
    for (const src of srcs) {
      const paths = (bySrc.get(src.sourceId) || []).sort();   // frame-00001… 序号补零，字符串序即 part 序
      if (!paths.length) { fail('manifest 里的 source ' + src.sourceId + ' 没有对应的 frames/*.parquet。'); return; }
      const nch = Number(src.pressureChannelCount) || 0;
      const foot = sideOfPlacement(src.placement), placement = String(src.placement || '').toLowerCase();
      const side = !nch && pressureSides.has(foot) && /^(left|right)ankle$/.test(placement) ? placement : foot;
      registerSideLabel(side, src);
      const axisPressure = src.deviceFamily === 'd56' && nch === NCH && /^(left|right)insole$/.test(placement) &&
        src.pressureLayoutId === 'd56-14x5-56-side-mirrored-candidate-v3';
      const job = {side, sourceId: src.sourceId, placement, family: src.deviceFamily, nch, axisPressure, files: [], nAll: 0};
      for (const path of paths) {
        const ab = await parquetBuffer(zf, path);
        const md = ParquetHQ.parquetMetadata(ab);
        const names = new Set((md.schema || []).map(s => s.name));
        const miss = ['pressureAdc', 'effectiveElapsedUs', 'validity'].filter(c => !names.has(c));
        if (miss.length) { fail(path + ' 缺少列：' + miss.join('、')); return; }
        if (srcInfo.ncols == null) srcInfo.ncols = (md.schema && md.schema[0] && md.schema[0].num_children) || null;
        const rows = Number(md.num_rows);
        const rgs = [];
        let acc = 0;
        for (const rg of md.row_groups || []) {
          const nr = Number(rg.num_rows);
          if (nr > 0) { rgs.push([acc, acc + nr]); acc += nr; }
        }
        job.files.push({ab, base: job.nAll, rows, rgs, cols: PKG_COLS.filter(c => names.has(c)), squat: [...new Set([...SQUAT_COLS,...WRIST_RAW_COLS,...SQUAT_HEAD_COLS])].filter(c => names.has(c))});
        job.nAll += rows;
      }
      jobs.push(job);
    }
  } catch (e) { fail('读取帧文件失败：' + ((e && e.message) || e)); return; }

  srcInfo.tempCol = 'temperatureCelsius';
  const groups = new Map();
  const extra = [];
  const seen = new Map();      // side -> sourceId：同一侧只能有一条流
  for (const job of jobs) {
    if (seen.has(job.side)) {
      fail(`压缩包里有两只来源都标成「${SIDE_CN[job.side] || job.side}」（${seen.get(job.side)} 与 ${job.sourceId}）：`
        + '查看器按侧组织面板，两条流不能合成一条；请把它们分到两个包里分别导入。');
      return;
    }
    seen.set(job.side, job.sourceId);
    if (job.nAll > LAZY_MIN) {
      const s = await buildLazySide(job);
      if (!s) { fail('帧文件里没有可用的帧时间。'); return; }
      extra.push(s);
    } else {
      groups.set(job.side, await readAllRows(job));
    }
  }

  const sides = expandSides(groups).concat(extra);
  if (!sides.length) { fail('包里有帧，但没有可用帧时间（effectiveElapsedUs 全为空）。'); return; }
  rebaseCommon(sides);
  sides.sort((a, b) => a.side.localeCompare(b.side));
  const ankles = ['leftankle','rightankle'].map(p => jobs.filter(j => j.family === 'walkpadImu' && j.placement?.toLowerCase() === p));
  if (ankles.every(a => a.length === 1)) {
    squatFiles.set(sides, [ankles[0][0].files,ankles[1][0].files]);
    const wrists=jobs.filter(j=>/^(left|right)wrist$/.test(j.placement) && /^(walkpadImu|witWt9011dcl)$/.test(j.family));
    const [w]=wrists;
    squatMeta.set(sides, {recordId:typeof manifest.recordId==='string' && manifest.recordId ? manifest.recordId : null,
      // 声明了会话时间轴的包，各源的有效时间本就落在同一轴上，对时不再靠 BLE 到达顺序
      aligned: !!manifest.sessionTimeline,
      wrist:wrists.length===1 ? {files:w.files, placement:w.placement, family:w.family} : undefined});
  }
  const verified = media ? await verifiedVideoOffset(archive) : undefined;
  const pack = media ? packVideoTiming(manifest, media.path) : null;
  applySides(sides, file.name, 'pressure', {rebuilt: false, src: srcInfo});
  if (media) {
    // 会话原点 = 各源最早的绝对首帧（rebaseCommon 之后全体同一个 t0），包内时刻减去它就是查看器轴上的起点
    const start = packVideoOffset(pack, sides[0].t0);
    try { attachVideo(media.u8, media.name, media.total, verified, start, pack?.uncertaintyUs); } catch { /* 视频挂不上就只没有视频 */ }
  }
  // 全段统计放后台跑，不挡住首屏；换文件时代数一变，旧扫描自己退出
  if (state.lazy) {
    const gen = dataGen;
    for (const s of sides) if (s.lazy) scanStats(s, gen);
  }
}

// 帧数低于这个值就整段读进来：短片段全量更省事，也保证旧数据的表现一点不变
const LAZY_MIN = 30000;
const LAZY_WIN = 60;   // 懒加载时的默认窗口长度 s
const LAZY_PAD = 5;    // 窗口前后各多读的余量 s，平移时不必每动一下就重读

const axisLoads = (side: string, p?: ArrayLike<number>) => {
  if (p?.length !== NCH || !GRID[side]) return [NaN, NaN, NaN];
  let heel = 0, fore = 0, total = 0;
  for (let i = 0; i < NCH; i++) {
    const adc = numOrNaN(p[i]), row = GRID[side][i][0];
    if (!Number.isFinite(adc)) return [NaN, NaN, NaN];
    // ADC 基线可以为负；承重证据只累加正载荷，不改原始压力。
    const x = Math.max(0, adc);
    total += x;
    if (row <= 3) heel += x;
    if (row >= 8) fore += x;
  }
  return [heel, fore, total];
};
function collectAxisSupport(side: string, rows: ArrayLike<number>[]): AxisSupport {
  const out: AxisSupport = [new Float32Array(rows.length), new Float32Array(rows.length), new Float32Array(rows.length)];
  rows.forEach((p, i) => axisLoads(side, p).forEach((x, k) => { out[k][i] = x; }));
  return out;
}

// 短片段：整段读进对象数组再交给共用展开
async function readAllRows(job) {
  const g = {eff: [], obs: [], v: [], iph: [], irw: [], ist: [], fno: [], tmp: [], st: [], ns: [], al: [], sfo: [],
    axisPressure: job.axisPressure,
    cal: {ready: 0, raw: 0}, deviceAngles: collectAngles(job.files)};
  let rows = 0;
  for (const f of job.files) {
    const part = await ParquetHQ.parquetReadObjects({file: f.ab, compressors: ParquetHQ.compressors, columns: f.cols});
    for (const r of part) if (pushPkgRow(g, r, rows)) rows++;
  }
  return g;
}

// 长录制：全段只留时间轴、包序号与「有效帧 -> 全局行号」，压力与 IMU 等窗口再用 loadBlock 取。
// 行号必须一起留 —— validity 非 valid 的帧被丢掉后，数组下标与文件行号对不上了。
async function buildLazySide(job) {
  const base = ['effectiveElapsedUs', 'validity', 'deviceSequence'];
  const t = [], row = [], seq = [], head = [];
  const headCursor = headingCursor();
  // 同一侧可能由多个文件拼成：只有全都有航向列时才做全段展开，否则数组与 t 错位
  const allHead = job.files.every(f => HEADING_COLS.every(c => f.cols.includes(c)));
  for (const f of job.files) {
    const hasHead = allHead;
    const cols = base.concat(f.cols.indexOf('frameType') >= 0 ? ['frameType'] : [], hasHead ? HEADING_COLS : []);
    for (const [r0, r1] of f.rgs) {
      const part = await ParquetHQ.parquetReadObjects({
        file: f.ab, compressors: ParquetHQ.compressors, columns: cols, rowStart: r0, rowEnd: r1
      });
      for (let i = 0; i < part.length; i++) {
        const r = part[i];
        if (r.validity !== 'valid' || !isSignalType(r.frameType)) continue;
        const e = numOrNaN(r.effectiveElapsedUs);
        if (!Number.isFinite(e)) continue;
        t.push(e); row.push(f.base + r0 + i); seq.push(numOrNaN(r.deviceSequence));
        if (hasHead) head.push(headCursor(numOrNaN(r.relativeHeadingDeg), r.referenceId));
      }
    }
  }
  const n = t.length;
  if (!n) return null;
  const t0 = t[0];
  const tt = new Float64Array(n), fno = new Float64Array(n), rr = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    tt[i] = (t[i] - t0) / 1e6; rr[i] = row[i]; fno[i] = seq[i];
  }
  let nch = job.nch;
  if (!nch) { for (const f of job.files) { nch = 56; break; } }
  const s = {
    side: job.side, lazy: true, files: job.files, nch, axisPressure: job.axisPressure,
    n, nAll: job.nAll, t: tt, fno, row: rr, t0,
    dur: tt[n - 1], span: tt[n - 1], fs: n > 1 ? 1 / ((tt[n - 1] - tt[0]) / (n - 1)) : NaN,
    nT: job.nAll - n,
    // 全段统计（零值率 / 最大 / 温度范围 / 丢包）要扫一遍全表，扫描完成前留空
    ch: null, mx: null, mn: null, zc: null, miss: null,
    // 全段展开好的航向（与 t 同序）：块只切片，换窗口不重新起算
    head: allHead ? Float32Array.from(head) : null,
    imu: null, temp: null, tLo: NaN, tHi: NaN, ictoIn: null, cal: {ready: 0, raw: 0},
    blk: null, statPending: true
  };
  lodInit(s);          // 全段抽样：后台扫描时才填，扫完前「全段」是半截
  await loadBlock(s, 0, Math.min(LAZY_WIN, s.dur));
  return s;
}

// 全局行号区间 -> 各文件交叠部分，逐块读出有效帧
async function loadBlock(s, a, b) {
  const i0 = lowerIdx(s.t, a), i1 = upperIdx(s.t, b);
  if (i1 <= i0) { s.blk = emptyBlock(); return; }
  const r0 = s.row[i0], r1 = s.row[i1 - 1] + 1;
  const rows = [];
  for (const f of s.files) {
    const lo = Math.max(r0, f.base), hi = Math.min(r1, f.base + f.rows);
    if (hi <= lo) continue;
    const part = await ParquetHQ.parquetReadObjects({
      file: f.ab, compressors: ParquetHQ.compressors, columns: f.cols,
      rowStart: lo - f.base, rowEnd: hi - f.base
    });
    for (const r of part) {
      if (r.validity !== 'valid' || !isSignalType(r.frameType)) continue;
      const e = numOrNaN(r.effectiveElapsedUs);
      if (!Number.isFinite(e)) continue;
      rows.push(r);
    }
  }
  s.blk = makeBlock(rows, s.nch, s.t0, s.head && s.head.length === s.n && rows.length === i1 - i0
    ? s.head.subarray(i0, i1) : null);
  // ⚠ IMU / 温度 / IC-TO 输入跟着块走，压力通道留在 blk.ch 里。这里把块内容挂回全段对象只是让
  // 「有没有 IMU 列」这类判空继续可用：s.imu.n 是块内帧数、s.n 是全段帧数，两者不能互相索引。
  // 取值一律走 blkOf(s)，下标必须与它所读的数组来自同一块。
  s.imu = s.blk.imu;
  s.temp = s.blk.temp;
  s.ictoIn = s.blk.ictoIn;
  // 温度量程按全段扫描结果；扫描前先用块的，免得温度带画出 NaN 坐标
  if (!Number.isFinite(s.tLo) && Number.isFinite(s.blk.tLo)) { s.tLo = s.blk.tLo; s.tHi = s.blk.tHi; }
  // 全段统计要等后台扫描；在那之前先用首块的最大值顶住纵轴与格点标注
  if (!s.mx && s.blk.n) {
    const mx = new Float32Array(s.nch), mn = new Float32Array(s.nch).fill(Infinity);
    for (let j = 0; j < s.nch; j++) {
      const col = s.blk.ch[j];
      for (let k = 0; k < s.blk.n; k++) {
        const v = col[k];
        if (v > mx[j]) mx[j] = v;
        if (v > 0 && v < mn[j]) mn[j] = v;
      }
    }
    s.mx = mx; s.mn = mn;
  }
}

function emptyBlock() {
  return {n: 0, t: new Float64Array(0), ch: [], imu: null, temp: null, ictoIn: null, a: NaN, b: NaN};
}

// 全段每轴最大 |值|：维特 TXT 这种整段在内存的路径没有统计扫描，就地算一次
function maxOfImu(im) {
  if (!im) return null;
  const m = {phys: new Float32Array(6), raw: new Float32Array(6)};
  for (const k of ['phys', 'raw']) {
    const arr = im[k];
    if (!arr) continue;
    for (let j = 0; j < 6; j++) {
      const a = arr[j];
      let mx = 0;
      for (let i = 0; i < a.length; i++) {
        const v = a[i];
        if (Number.isFinite(v)) { const q = Math.abs(v); if (q > mx) mx = q; }
      }
      m[k][j] = mx;
    }
  }
  return m;
}

// 块内数据：时间转成与全段同一起点的绝对秒，绘图与时间窗直接用
// head：全段展开好的航向（与 rows 同序），块里直接切片，避免每块各自起算导致换窗口时差整圈
function makeBlock(rows, nch, t0, head = null) {
  const n = rows.length;
  if (!n) return emptyBlock();
  const t = new Float64Array(n);
  const ch = Array.from({length: nch}, () => new Float32Array(n));
  const six = () => Array.from({length: 6}, () => new Float32Array(n).fill(NaN));
  const ph = six(), rw = six();
  const phMax = new Float32Array(6), rwMax = new Float32Array(6);
  const tp = new Float64Array(n).fill(NaN);
  // 设备角（pitchDeg / sideTiltDeg）按块读：懒加载的长包只有窗口块可画
  const pit = new Float32Array(n).fill(NaN), tlt = new Float32Array(n).fill(NaN);
  let nPit = 0, nTlt = 0;
  let hasPhys = false, hasRaw = false, tLo = Infinity, tHi = -Infinity;
  for (let k = 0; k < n; k++) {
    const r = rows[k];
    t[k] = (numOrNaN(r.effectiveElapsedUs) - t0) / 1e6;
    const pv = r.pressureAdc || [];
    for (let j = 0; j < nch; j++) ch[j][k] = numOrNaN(pv[j]);
    const a = phys6(r);
    const b = struct6(r.accelerationRaw, r.angularVelocityRaw);
    if (a) { hasPhys = true; for (let j = 0; j < 6; j++) { ph[j][k] = a[j]; const q = Math.abs(a[j]); if (q > phMax[j]) phMax[j] = q; } }
    if (b) { hasRaw = true; for (let j = 0; j < 6; j++) { rw[j][k] = b[j]; const q = Math.abs(b[j]); if (q > rwMax[j]) rwMax[j] = q; } }
    const c = numOrNaN(r.temperatureCelsius);
    if (Number.isFinite(c)) { tp[k] = c; if (c < tLo) tLo = c; if (c > tHi) tHi = c; }
    const pv2 = numOrNaN(r.pitchDeg), tv2 = numOrNaN(r.sideTiltDeg);
    if (Number.isFinite(pv2)) { pit[k] = pv2; nPit++; }
    if (Number.isFinite(tv2)) { tlt[k] = tv2; nTlt++; }
  }
  const imu = (hasPhys || hasRaw) ? {nch: 6, phys: hasPhys ? ph : null, raw: hasRaw ? rw : null, n, nPart: 0} : null;
  const hd = head ? Float32Array.from(head) : null;
  const nHead = hd ? hd.reduce((c, v) => c + (Number.isFinite(v) ? 1 : 0), 0) : 0;
  const deviceAngles = (nPit || nTlt || nHead)
    ? {pitch: nPit ? pit : null, sideTilt: nTlt ? tlt : null, heading: nHead ? hd : null} : null;
  return {
    n, t, ch, imu, temp: tHi >= tLo ? tp : null, deviceAngles, ictoIn: null,
    a: t[0], b: t[n - 1], tLo, tHi,
    imuMax: imu ? {phys: phMax, raw: rwMax} : null
  };
}

function lowerIdx(arr, v) {
  let lo = 0, hi = arr.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] < v) lo = m + 1; else hi = m; }
  return lo;
}
function upperIdx(arr, v) {
  let lo = 0, hi = arr.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] <= v) lo = m + 1; else hi = m; }
  return lo;
}

// 零值率、全段最大、温度范围这些只有扫完整个表才知道，但没人愿意为此等一次全量读。
// 后台按 row group 顺序流式累加，只留统计量不留数组；扫完再回填并重绘当前页。
// 数据集代数：每次导入新文件 +1，异步任务（后台扫描、补读窗口）凭它判断自己是否已过期
let dataGen = 0;
// 全段抽样是边扫边填的：只有正看着全段才值得每几块重绘一次，窗口视图不受影响
function redrawFull() {
  if (!state.lazy || state.win) return;
  if (state.tab === 'wave') redrawWave();
  else if (state.tab === 'imu') { clearOvCache(); redrawImu(); }
}
async function scanStats(s, gen) {
  const nch = s.nch;
  let fed = 0;
  const zc = new Float32Array(nch), miss = new Float32Array(nch);
  const mx = new Float32Array(nch), mn = new Float32Array(nch).fill(Infinity);
  let tLo = Infinity, tHi = -Infinity;
  for (const f of s.files) {
    // IMU 列只为全段抽样读：抽帧要有真实帧值，不能拿压力顶替
    const cols = ['pressureAdc', 'validity', 'temperatureCelsius', 'effectiveElapsedUs']
      .concat(f.cols.indexOf('frameType') >= 0 ? ['frameType'] : [])
      .concat(IMU_COLS.filter(c => f.cols.indexOf(c) >= 0));
    for (const [r0, r1] of f.rgs) {
      if (gen !== dataGen) return;
      const part = await ParquetHQ.parquetReadObjects({
        file: f.ab, compressors: ParquetHQ.compressors, columns: cols, rowStart: r0, rowEnd: r1
      });
      let lastT = NaN;
      for (const r of part) {
        if (r.validity !== 'valid' || !isSignalType(r.frameType)) continue;
        const e = numOrNaN(r.effectiveElapsedUs);
        if (!Number.isFinite(e)) continue;
        const pv = r.pressureAdc;
        let sum = 0;
        for (let j = 0; j < nch; j++) {
          const v = pv ? pv[j] : null;
          if (v == null || !Number.isFinite(v)) { miss[j]++; continue; }
          if (v > mx[j]) mx[j] = v;
          if (v === 0) zc[j]++;
          else { if (v < mn[j]) mn[j] = v; sum += v; }
        }
        lastT = (e - s.t0) / 1e6;
        ovFeed(s, lastT, sum);
        const c = numOrNaN(r.temperatureCelsius);
        if (Number.isFinite(c)) { if (c < tLo) tLo = c; if (c > tHi) tHi = c; }
        const gy = gyroOf(r);
        if (gy && gy !== r.angularVelocityDps) s.cal.ready++; else if (gy) s.cal.raw++;
        if (s.lod) {
          const ph = (r.accelerationG || gy)
            ? struct6(r.accelerationG, gy) : null;
          const rw = (r.accelerationRaw || r.angularVelocityRaw)
            ? struct6(r.accelerationRaw, r.angularVelocityRaw) : null;
          lodFeed(s, lastT, pv || null, ph, rw, c);
        }
      }
      if (s.ov && Number.isFinite(lastT)) s.ov.upto = lastT;
      if (s.lod && Number.isFinite(lastT)) s.lod.upto = lastT;
      // 让出主线程，别让后台扫描把拖窗口的交互卡住；缩略条每几块刷一次就够看
      if (++fed % 4 === 0) {
        if (s.ov && drawScrub) drawScrub();
        redrawFull();          // 全段视图要看着它一格一格填进来
      }
      await new Promise(r => setTimeout(r, 0));
    }
  }
  if (gen !== dataGen) return;
  s.zc = zc; s.miss = miss; s.mx = mx; s.mn = mn;
  if (s.ov) { s.ov.partial = false; s.ov.upto = s.dur; }
  if (s.lod) {
    lodDone(s);
    s.imuMax = s.lod.imuMax;      // 全段每轴最大 |值|：纵轴按它定，不用抽样值
  }
  if (tHi >= tLo) { s.tLo = tLo; s.tHi = tHi; }
  s.statPending = false;
  // 全段最大变了，顶栏与自动纵轴要跟着改
  const gmax = Math.max(...state.sides.map(x => x.mx ? Math.max(...x.mx) : 0));
  state.gmax = gmax;
  if (state.auto) {
    state.ymax = niceMax(gmax);
    if (el.ymax) el.ymax.value = Math.round(state.ymax);
  }
  if (updateChips) updateChips();
  if (state.sides.length) setTab(state.tab);
}

export async function readCopBlock(s, a: number, b: number) {
  if (!s.lazy || blkCovers(s, a, b)) return s.blk;
  const isolated = {...s};
  await loadBlock(isolated, a, b);
  return isolated.blk;
}

let loadingWin = false, pendingWin = null;
// 块算不算覆盖窗口：看窗口里真正要画的首末样本是否都在块内，不能只看有没有交集。
// 已有 [0,60]、要看 [30,90] 时两者相交，但后 30 s 根本没读进来。
// 也不能直接比秒数：窗口端点通常没有正好落在采样点上，那样每次都会判成「没覆盖」而反复重读。
function needSpan(s, a, b) {
  const i0 = lowerIdx(s.t, a), i1 = upperIdx(s.t, b) - 1;
  return i1 < i0 ? null : {ta: s.t[i0], tb: s.t[i1]};
}
function blkCovers(s, a, b) {
  const c = s.blk, need = needSpan(s, a, b);
  if (!c || !(c.n > 0) || !need) return false;
  return c.a <= need.ta + 1e-9 && c.b >= need.tb - 1e-9;
}

async function ensureWin(a, b) {
  const ls = state.sides.filter(s => s.lazy);
  if (!ls.length) return;
  const todo = [];
  for (const s of ls) {
    if (useLod(s, a, b)) continue;            // 这个跨度画的是全段抽样，不必读块
    if (blkCovers(s, a, b)) continue;         // 前后 LAZY_PAD 只是读的余量，判据只看用户要的区间
    const aa = Math.max(0, a - LAZY_PAD), bb = Math.min(s.dur, b + LAZY_PAD);
    if (bb <= aa) continue;
    todo.push({s, aa, bb});
  }
  if (!todo.length) return;
  // 拖动过程中会连续触发：只记最新窗口，读完再看要不要补一次
  if (loadingWin) { pendingWin = [a, b]; return; }
  loadingWin = true;
  if (el.winText) el.winText.setAttribute('aria-busy', 'true');
  const mine = state.sides;      // 读的过程中换成了别的文件，就不要再去刷别人的界面
  try {
    for (const it of todo) await loadBlock(it.s, it.aa, it.bb);
    if (state.sides !== mine) return;
    clearOvCache();
    setTab(state.tab);
    updateWinBar();
  } finally {
    loadingWin = false;
    if (el.winText) el.winText.removeAttribute('aria-busy');
    if (pendingWin) {
      pendingWin = null;
      const w = state.win || {a: 0, b: Math.max(...state.sides.map(s => s.dur), 0)};
      void ensureWin(w.a, w.b).catch(e => fail('时间窗加载失败：' + (e?.message || e)));
    }
  }
}

function parseWtJournal(bytes, name) {
  const lines = new TextDecoder().decode(bytes).split(/\r?\n/).filter(l => l.trim());
  const rows = lines.map((l, i) => { try { return JSON.parse(l); } catch { throw new Error(`WT 记录第 ${i + 1} 行损坏`); } });
  const header = rows.find(r => r.type === 'start');
  if (!header || !Number.isFinite(header.phone_ms) || header.preprocess !== 'activity-wt-receipt-v1') throw new Error('无法识别 WT 本地版的采集契约');
  const origin = header.phone_ms;
  const streams = [0, 1].map(() => ({t: [], axes: Array.from({length: 6}, () => []), seq: [], ok: [], pending: [], connection: null, previousRx: NaN, last: NaN, counter: 0}));
  for (const row of rows) {
    const foot = Number(row.foot); if (foot !== 0 && foot !== 1) continue; const s = streams[foot];
    if (row.type === 'disconnect') { s.connection = null; s.pending = []; s.previousRx = NaN; continue; }
    if (row.type !== 'notify') continue;
    const rx = Number(row.receipt_ms); if (!Number.isFinite(rx)) continue;
    if (s.connection !== row.connection) { s.pending = []; s.previousRx = NaN; s.connection = row.connection; s.counter = 0; s.last = NaN; }
    const raw = atob(row.bytes || ''); const data = s.pending.concat(Array.from(raw, c => c.charCodeAt(0))); let i = 0; const values = [];
    const s16 = k => { const v = data[k] | data[k + 1] << 8; return v >= 32768 ? v - 65536 : v; };
    while (i < data.length) {
      if (data[i] !== 0x55) { i++; continue; } if (i + 1 >= data.length) break;
      const kind = data[i + 1], size = [0x57, 0x58, 0x59, 0x61, 0x71].includes(kind) ? 20 : ((kind >= 0x50 && kind <= 0x56) || kind === 0x5a || kind === 0x5f ? 11 : 0);
      if (!size) { i++; continue; } if (i + size > data.length) break;
      if (size === 11 && (data.slice(i, i + 10).reduce((a, b) => a + b, 0) & 255) !== data[i + 10]) { i++; continue; }
      if (kind === 0x61) values.push([s16(i + 2) / 2048, s16(i + 4) / 2048, s16(i + 6) / 2048, s16(i + 8) * 2000 / 32768, s16(i + 10) * 2000 / 32768, s16(i + 12) * 2000 / 32768]);
      i += size;
    }
    s.pending = data.slice(i); if (!values.length) continue;
    if (Number.isFinite(s.previousRx) && rx <= s.previousRx) continue;
    let end = rx - 20;
    if (Number.isFinite(s.last) && rx - s.previousRx <= 120) { const predicted = s.last + values.length * 20, residual = end - predicted; if (Math.abs(residual) <= 80) end = Math.min(rx, predicted + Math.max(-2, Math.min(2, residual))); }
    if (Number.isFinite(s.last) && end <= s.last) continue;
    const begin = Math.max(end - values.length * 20, Number.isFinite(s.last) ? s.last : -Infinity);
    values.forEach((v, j) => { const ms = begin + (end - begin) * (j + 1) / values.length; s.t.push((ms - origin) / 1000); s.seq.push(++s.counter); s.ok.push(1); v.forEach((x, c) => s.axes[c].push(x)); s.last = ms; });
    s.previousRx = rx;
  }
  const sides = streams.flatMap((s, f) => {
    if (!s.t.length) return [];
    const t = Float64Array.from(s.t), phys = s.axes.map(a => Float32Array.from(a)), n = t.length, imu = {nch: 6, phys, raw: null, n, nPart: 0};
    return [{side: f ? 'right' : 'left', n, nch: 0, t, ch: [], mx: new Float32Array(0), mn: new Float32Array(0), zc: new Float32Array(0), miss: new Float32Array(0), imu, fno: Float64Array.from(s.seq), temp: null, tLo: NaN, tHi: NaN, t0: 0, ictoIn: null, nT: 0, dur: t[n - 1], span: t[n - 1], fs: n > 1 ? (n - 1) / (t[n - 1] - t[0]) : NaN,
      activityIn: {t, axes: phys, seq: Float64Array.from(s.seq), ok: Uint8Array.from(s.ok)}}];
  });
  if (!sides.length) throw new Error('WT 记录里没有可解码的 0x61 六轴通知');
  applySides(sides, name, 'pressure', {rebuilt: false, src: {ver: 'WT 本地 ' + (header.app_version || ''), ncols: null, tempCol: null}});
  state.receiptTiming = true;
}

// ---------------------------------------------------------------- 维特 TXT
// 维特上位机（WT901 / JY901 等）导出的表格：制表符或逗号分隔，一台设备一段。
// 列名形如 时间 / 设备名称 / 加速度X(g) / 角速度X(°/s) / 角度X(°) / 磁场…，后面几列常是 null。
// ⚠ 时间戳只到毫秒，同一毫秒里常压着多帧（实测 40% 的相邻间隔为 0），另有 60 ms 的丢包间隔；
// 直接拿它当 t 会把采样率算成 1 ms 量级。这里把同一时间戳组内的帧均匀摊到下一组之前，
// 既消掉零间隔，又保留丢包留下的长间隔。
async function parseWit(file) {
  let txt;
  try { txt = await file.text(); } catch (e) { fail('读取文件失败：' + e.message); return; }
  if (txt.charCodeAt(0) === 0xFEFF) txt = txt.slice(1);

  const lines = txt.split(/\r?\n/);
  // 表头：第一行同时含「时间/time」和轴标记；前面可能有标题行或空行
  const hi = lines.findIndex(l => /时间|time/i.test(l) && /[XYZxyz]/.test(l));
  if (hi < 0) { fail('没找到维特的表头行（应含「时间」「加速度X」这类列名）。'); return; }
  const sep = lines[hi].includes('\t') ? '\t' : (lines[hi].includes(';') ? ';' : ',');
  const hdr = lines[hi].split(sep).map(s => s.trim());

  const find = re => hdr.findIndex(h => re.test(h));
  const iT = find(/^(时间|time)/i);
  const iDev = find(/设备名称|device/i);
  const iTp = find(/温度|temp/i);
  const iA = ['X', 'Y', 'Z'].map(a => find(new RegExp(`加速度\\s*${a}|^acc\\w*${a}`, 'i')));
  const iG = ['X', 'Y', 'Z'].map(a => find(new RegExp(`角速度\\s*${a}|^as\\w*${a}|gyro\\w*${a}`, 'i')));
  if (iT < 0 || iA.some(i => i < 0) || iG.some(i => i < 0)) {
    fail(`表头里缺列：${[iT < 0 ? '时间' : '', ...['X', 'Y', 'Z'].map((a, k) => iA[k] < 0 ? '加速度' + a : ''),
      ...['X', 'Y', 'Z'].map((a, k) => iG[k] < 0 ? '角速度' + a : '')].filter(Boolean).join('、')}。`);
    return;
  }

  const groups = new Map();
  for (let li = hi + 1; li < lines.length; li++) {
    const raw = lines[li];
    if (!raw || !raw.trim()) continue;
    const c = raw.split(sep);
    if (c.length < hdr.length * 0.5) continue;          // 半截行（文件尾被截断）
    const dev = iDev >= 0 ? (c[iDev] || '').trim() : '';
    const tRaw = Date.parse((c[iT] || '').trim());
    const num = i => {
      const v = Number((c[i] || '').trim());
      return Number.isFinite(v) ? v : NaN;
    };
    const a = iA.map(num), g = iG.map(num);
    if (a.some(v => !Number.isFinite(v)) && g.some(v => !Number.isFinite(v))) continue;
    const key = dev || '未命名设备';
    let grp = groups.get(key);
    if (!grp) { grp = {t: [], a: [], g: [], tp: []}; groups.set(key, grp); }
    grp.t.push(tRaw);
    grp.a.push(a);
    grp.g.push(g);
    grp.tp.push(iTp >= 0 ? num(iTp) : NaN);
  }
  if (!groups.size) { fail('文件里没有可用的 IMU 数据行。'); return; }

  const sides = [];
  let k = 0;
  // 多台设备共用同一份导出的墙钟：取全文件最早的时间戳当公共原点
  let tminMs = Infinity;
  for (const grp of groups.values()) for (const v of grp.t) if (Number.isFinite(v) && v < tminMs) tminMs = v;
  for (const [dev, grp] of groups) {
    k++;
    const n = grp.t.length;
    if (n < 30) continue;
    // 时间戳缺失或乱序时退化成按行序号 + 名义 50 Hz，至少能算
    let tRaw = grp.t;
    let ok = tRaw.every(v => Number.isFinite(v));
    if (ok) {
      const order = tRaw.map((_, i) => i).sort((x, y) => tRaw[x] - tRaw[y]);
      tRaw = order.map(i => grp.t[i]);
      grp.a = order.map(i => grp.a[i]);
      grp.g = order.map(i => grp.g[i]);
      grp.tp = order.map(i => grp.tp[i]);
    } else {
      tRaw = Array.from({length: n}, (_, i) => i * 20);   // 20 ms = 50 Hz
    }
    const wit = witBatches(tRaw);
    const t = rebuildTime(tRaw);
    // 各设备独立归零会把它们之间的相对偏移抹掉（同一份导出里两台设备是同时开录的）。
    // 偏移只用于对齐显示；写盘批次重建出的轴照旧不能当采样时刻。
    const off = ok && Number.isFinite(tminMs) ? (tRaw[0] - tminMs) / 1000 : 0;
    if (off) {
      for (let i = 0; i < n; i++) t[i] += off;
      for (let i = 0; i < wit.T.length; i++) wit.T[i] += off;
    }

    const ch = Array.from({length: 6}, () => new Float32Array(n).fill(NaN));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < 3; j++) {
        ch[j][i] = grp.a[i][j];
        ch[3 + j][i] = grp.g[i][j];
      }
    }
    const dt = (t[n - 1] - t[0]) / (n - 1);
    // 上位机的温度列（温度(°C)）是可选列，缺了就不画那一条
    const tp = new Float64Array(n).fill(NaN);
    let tLo = Infinity, tHi = -Infinity;
    for (let i = 0; i < n; i++) {
      const v = grp.tp[i];
      if (!Number.isFinite(v)) continue;
      tp[i] = v;
      if (v < tLo) tLo = v;
      if (v > tHi) tHi = v;
    }
    // 侧别名取 MAC 尾两段：一台设备一份文件时也能看出是哪台
    const mac = /\(([^)]+)\)/.exec(dev);
    const side = mac ? mac[1].split(/[:-]/).slice(-2).join(':') : `设备${k}`;
    // 维特 TXT 没有包序号，行序即顺序，分段边界只剩时间倒退（重建轴单调，等于整段不分）；
    // 时间轴是写盘批次重建的，IC/TO 只作参考候选
    const inp = {n, tUs: new Float64Array(n), seq: new Float64Array(n),
      ord: new Float64Array(n), ok: new Uint8Array(n).fill(1), ax: new Float64Array(n * 6),
      row: new Int32Array(n), sfo: new Float64Array(n).fill(NaN)};
    let witBad = 0;
    for (let i = 0; i < n; i++) {
      inp.tUs[i] = Math.round(t[i] * 1e6);
      inp.seq[i] = i; inp.ord[i] = i; inp.row[i] = i;
      for (let j = 0; j < 3; j++) {
        const av = grp.a[i][j], gv = grp.g[i][j];
        if (!Number.isFinite(av) || !Number.isFinite(gv)) witBad++;
        inp.ax[i * 6 + j] = av;
        inp.ax[i * 6 + 3 + j] = gv;
      }
    }
    sides.push({
      side, dev, n, nch: 0, t,
      ch: [], mx: new Float32Array(0), mn: new Float32Array(0),
      zc: new Float32Array(0), miss: new Float32Array(0),
      imu: {nch: 6, phys: ch, raw: null, n, nPart: 0},
      temp: tHi >= tLo ? tp : null, tLo, tHi, t0: 0, ictoIn: witBad ? null : inp,
      fno: null, wit, dur: t[n - 1], span: t[n - 1], fs: dt > 0 ? 1 / dt : NaN
    });
  }
  if (!sides.length) { fail('文件里的设备都没有足够的帧（每台至少 30 帧）。'); return; }
  sides.sort((a, b) => a.side.localeCompare(b.side));
  applySides(sides, file.name, 'imu',
    {rebuilt: true, src: {ver: null, ncols: null, tempCol: iTp >= 0 ? hdr[iTp] : null}});
}

// 同一时间戳组内均匀细分，消掉 0 间隔；末组沿用前一组间隔
function rebuildTime(tRaw) {
  const n = tRaw.length, t = new Float64Array(n);
  const uniq = [];
  for (let i = 0; i < n;) {
    let j = i;
    while (j + 1 < n && tRaw[j + 1] === tRaw[i]) j++;
    uniq.push([i, j + 1, tRaw[i]]);
    i = j + 1;
  }
  for (let u = 0; u < uniq.length; u++) {
    const [a, b, ts] = uniq[u];
    const m = b - a;
    const span = u + 1 < uniq.length ? uniq[u + 1][2] - ts : (u > 0 ? ts - uniq[u - 1][2] : 0);
    for (let q = 0; q < m; q++) t[a + q] = m > 1 ? ts + span * q / m : ts;
  }
  const t0 = t[0];
  for (let i = 0; i < n; i++) t[i] = (t[i] - t0) / 1000;   // ms -> s
  return t;
}

// 上位机是「攒一批再写盘」：同一批样本的时间戳挤在几毫秒里，与真实采样时刻无关。
// 相邻两帧相差 ≤3 ms 归为同一次写盘（按「离批首不超过 3 ms」归并会把 40 帧的大批切碎）。
function witBatches(tRaw) {
  const n = tRaw.length, T = [], K = [];
  const t0 = tRaw[0];
  for (let i = 0; i < n;) {
    let j = i;
    while (j + 1 < n && tRaw[j + 1] - tRaw[j] <= 3) j++;
    T.push((tRaw[i] - t0) / 1000);
    K.push(j - i + 1);
    i = j + 1;
  }
  return {T: Float64Array.from(T), K: Int32Array.from(K), span: (tRaw[n - 1] - t0) / 1000};
}

// ---------------------------------------------------------------- 收尾
function applySides(sides, name, mode, opt) {
  // 换主数据 = 旧数据集的异步任务全部作废（短文件也要 +1：之前只有长包会作废扫描）
  dataGen++;
  state.sides = sides;
  state.receiptTiming = false;
  // 纵轴定标要的是全段每轴最大 |值|：短文件就地算一次，长文件由后台扫描给
  for (const s of sides) if (!s.lazy) s.imuMax = (s.blk && s.blk.imuMax) || maxOfImu(s.imu);
  if (ovInitAll) ovInitAll(sides);
  state.name = name;
  state.mode = mode;
  // 纯 IMU 的 parquet（pressure_status = not_applicable）每帧的 pressure_values 都是空数组，
  // 通道数算出来是 0：压力三页没东西可画，默认页签也该落在 IMU 上。
  state.hasP = sides.some(s => s.nch > 0);
  state.rebuilt = !!(opt && opt.rebuilt);
  state.src = (opt && opt.src) || null;
  state.gmax = state.hasP ? Math.max(...sides.map(s => s.mx ? Math.max(...s.mx) : 0)) : 0;
  state.auto = true;
  state.ymax = niceMax(state.gmax);
  state.detail = null;
  // 懒加载的长录制默认只展开开头一段：整段几十万帧，全量画既慢也看不清
  const D = sides.length ? Math.max(...sides.map(s => s.dur)) : 0;
  state.lazy = sides.some(s => s.lazy);
  state.win = state.lazy && D > LAZY_WIN ? {a: 0, b: Math.min(LAZY_WIN, D)} : null;
  // 小时级的录制默认抽到 0.1 Hz：一屏四千点上下，看得出「哪一段在动」；
  // 短片段保持按屏宽，否则十分钟的记录只剩六十个点，线是断的
  state.drawHz = D >= 3600 ? 0.1 : null;
  if (el.rateSel) el.rateSel.value = state.drawHz ? '0.1' : 'auto';
  state.hoverT = null;
  state.copT = 0;
  state.playing = false;
  state.tab = state.hasP ? 'wave' : 'imu';
  // 丢帧阈值按时间轴来源给默认值：维特的时间轴是写盘批次重建的，40 ms 会标出满屏写盘抖动
  state.imuGapMs = state.rebuilt ? 100 : 40;
  if (el.imuGapMs) el.imuGapMs.value = state.imuGapMs;
  // 独立轴向页没有顶栏，只有主查看器才有纵轴输入框
  if (el.ymax) el.ymax.value = Math.round(state.ymax);
  document.querySelectorAll('[data-y]').forEach(b => b.classList.toggle('on', (b as HTMLElement).dataset.y === 'auto'));
  // IC/TO 要在第一次绘制前算好，否则顶栏与标记得等第二次重画才出来
  if (ictoReset) ictoReset();
  renderAll();
}

// ---------------------------------------------------------------- 布局
// 可用宽度不够时减少列数、增加行数，宁可整图变高也不把格子压小
function gridDims(W) {
  const padL = 56, padR = 12, gapX = 12, gapY = 16;
  const usable = W - padL - padR;
  let C = 4;
  for (const c of [8, 7, 4]) {                 // 只取能整除 56 的列数，避免末行残缺
    if ((usable - gapX * (c - 1)) / c >= 104) { C = c; break; }
  }
  return {padL, padR, gapX, gapY, C, R: Math.ceil(NCH / C)};
}

export async function activityInput(s, alive = () => true, withSupport = false) {
  if (s.activityIn) return s.activityIn;
  if (!s.lazy) throw new Error('没有可信的物理量与会话时间；不使用重建时间轴或原始计数推断');
  const raw = [], seq = [], ok = [], rails = [], axes = Array.from({length: 6}, () => []);
  const loads: [number[], number[], number[]] | undefined = withSupport && s.axisPressure ? [[], [], []] : undefined;
  const angles = collectAngles(s.files);
  for (const f of s.files) for (const [r0, r1] of f.rgs) {
    if (!alive()) throw new Error('已取消');
    const cols = ['effectiveElapsedUs', 'timeUs', 'validity', 'frameType', 'accelerationG', 'angularVelocityDps',
      'accelerationRaw', 'deviceSequence', 'imuCalibration', ...DEVICE_ANGLE_COLS,
      ...(loads ? ['pressureAdc'] : [])].filter(c => f.cols.includes(c));
    const rows = await ParquetHQ.parquetReadObjects({file: f.ab, compressors: ParquetHQ.compressors, columns: cols, rowStart: r0, rowEnd: r1});
    for (const r of rows) {
      if (!isSignalType(r.frameType)) continue;
      const ph = phys6(r);
      const effective = numOrNaN(r.effectiveElapsedUs), mapped = r.validity === 'valid' && Number.isFinite(effective);
      const us = mapped ? effective : numOrNaN(r.timeUs);
      raw.push(us); seq.push(numOrNaN(r.deviceSequence)); ok.push(mapped && ph ? 1 : 0);
      const a = r.accelerationRaw;
      rails.push(a && [a.x, a.y, a.z].some(v => Math.abs(numOrNaN(v)) >= 32760) ? 1 : 0);
      for (let j = 0; j < 6; j++) axes[j].push(ph?.[j] ?? NaN);
      if (loads) axisLoads(s.side, r.pressureAdc).forEach((x, k) => loads[k].push(x));
      appendDeviceAngles(angles, r);
    }
    await new Promise(r => setTimeout(r, 0));
  }
  const t = Float64Array.from(spreadNotified(raw), v => (v - s.t0) / 1e6);
  return {t, seq: Float64Array.from(seq), ok: Uint8Array.from(ok), accAtRail: Uint8Array.from(rails),
    deviceAngles: deviceAngleArrays(angles), axes: axes.map(a => Float32Array.from(a)),
    support: loads && [Float32Array.from(loads[0]), Float32Array.from(loads[1]), Float32Array.from(loads[2])] as AxisSupport | undefined};
}

export {ensureWin, gridDims, LAZY_WIN as lazyWin, parse};
