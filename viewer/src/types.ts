/* 数据模型。这是全工程的契约所在：
   一次导入的结果是 Side[]，其中「全段」字段与「窗口块」字段分属两层 —— 下标不能跨层用。
   s.n / s.t / s.dur 是全段；s.blk.n / s.blk.t / s.blk.ch 只有窗口内的帧。
   整段载入时两者是同一批数组，懒加载时不是。取块内数据一律走 blkOf()。 */

export type SideId = string;

/** IMU 六轴：前 3 加速度（g 或 LSB）、后 3 角速度（°/s 或 LSB） */
export interface ImuData {
  nch: 6;
  /** 采集端已换算的物理量；没有这一套时为 null */
  phys: Float32Array[] | null;
  /** 硬件整数计数；没有这一套时为 null */
  raw: Float32Array[] | null;
  n: number;
  /** imu_unit_status 既非 confirmed 也非 absent 的帧数 */
  nPart: number;
}

/** IC/TO 检测的输入，按文件行序（不是时间序）：算法要求包序号严格递增 */
export interface IctoInput {
  n: number;
  tUs: Float64Array;
  seq: Float64Array;
  ord: Float64Array;
  ok: Uint8Array;
  ax: Float64Array;
  row: Int32Array;
  sfo: Float64Array;
}

/** 维特上位机写盘批次：T = 批首时刻，K = 该批帧数 */
export interface WitBatches {
  T: Float64Array;
  K: Int32Array;
  span: number;
}

/** 测量包里的一个帧文件及其 row group 索引 */
export interface PkgFile {
  ab: ArrayBuffer;
  /** 该文件首行在整侧里的全局行号 */
  base: number;
  rows: number;
  /** [起始行, 结束行) 全局行号区间，逐 row group */
  rgs: [number, number][];
  cols: string[];
}

/** 窗口块：只含某个时间窗内的帧，时间是全段同一原点的绝对秒 */
export interface Block {
  n: number;
  t: Float64Array;
  ch: Float32Array[];
  imu: ImuData | null;
  temp: Float64Array | null;
  /** 设备自报俯仰 / 侧倾角与相对航向（度），列缺失或整段为空时为 null；航向已按 referenceId 展开 */
  deviceAngles?: {pitch: Float32Array | null; sideTilt: Float32Array | null; heading?: Float32Array | null} | null;
  ictoIn: IctoInput | null;
  a: number;
  b: number;
  tLo: number;
  tHi: number;
  /** 块内每轴最大 |值|；懒加载的块只有窗口，别当全段极值用 */
  imuMax?: {phys: Float32Array; raw: Float32Array} | null;
}

/** 全程缩略条：每桶内「通道读数之和」的峰值 */
export interface Overview {
  /** 每桶內「56 通道读数之和」的峰值 */
  v: Float32Array;
  n: number;
  dur: number;
  /** 后台扫描已覆盖到的时间；partial 为 true 时右侧画成「统计中」 */
  upto: number;
  partial: boolean;
  max: number;
}

/** 全段摘要：均匀时间网格上每格只留一帧真实读数（抽帧）。
    不做 min–max —— 一个异常值就会把整段压成一条带；抽帧时它顶多是个孤立的点。 */
export interface Lod {
  /** 格数；t 按格中心预填，缺帧的格子靠通道里的 NaN 断笔 */
  n: number;
  /** 每格秒数 */
  bin: number;
  dur: number;
  /** 扫描已覆盖到的时间；partial 为 true 时右侧还没数据 */
  upto: number;
  partial: boolean;
  t: Float64Array;
  ch: Float32Array[] | null;
  imu: {nch: 6; phys: Float32Array[] | null; raw: Float32Array[] | null} | null;
  temp: Float64Array | null;
  /** 每轴全段最大 |值|：纵轴按它定，不能用抽样值 */
  imuMax: {phys: Float32Array; raw: Float32Array} | null;
  tLo: number;
  tHi: number;
}

/** 一侧（一只脚 / 一台设备）的全部数据 */
export interface Side {
  side: SideId;
  /** 全段帧数 */
  n: number;
  nch: number;
  /** 全段时间轴（秒，同一文件内共用原点） */
  t: Float64Array;
  /** 全段通道数组；懒加载且扫描未完成时为 null */
  ch: Float32Array[] | null;
  mx: Float32Array | null;
  mn: Float32Array | null;
  zc: Float32Array | null;
  miss: Float32Array | null;
  imu: ImuData | null;
  fno: Float64Array | null;
  temp: Float64Array | null;
  tLo: number;
  tHi: number;
  /** 这一侧首帧的绝对时间（µs），rebaseCommon 之后同一文件内一致 */
  t0: number;
  ictoIn: IctoInput | null;
  /** 取不到有效帧时间而被丢掉的帧数 */
  nT: number;
  /** 角速度口径计数：ready = 用了包内扣偏值的帧，raw = 回落原始换算值的帧 */
  cal?: {ready: number; raw: number};
  /** 时间轴末端时刻（秒，会话原点）；图表横轴按它铺，别当录制时长用 */
  dur: number;
  /** 真实录制时长（末帧 − 首帧，秒）。一包里先出流的那台设备会把其余流的起点推到几秒之后，
      这时 dur ≠ span —— 报「多久」一律用它。 */
  span: number;
  /** 全段平均有效记录率 (n-1)/(t[last]-t[0]) */
  fs: number;
  /** 窗口块；整段载入时与上面的字段指向同一批数组 */
  blk: Block | null;
  activityIn?: {t: Float64Array; seq: Float64Array; ok: Uint8Array; axes: Float32Array[]};
  lazy?: boolean;
  files?: PkgFile[];
  nAll?: number;
  /** 全局行号，懒加载补读时按它定位 */
  row?: Float64Array;
  statPending?: boolean;
  ov?: Overview | null;
  /** 全段抽样（懒加载的长包才有）；没有它就只能看窗口块 */
  lod?: Lod | null;
  /** 每轴全段最大 |值|，导入时算一次，纵轴定标不再逐次扫全段 */
  imuMax?: {phys: Float32Array; raw: Float32Array} | null;
  wit?: WitBatches;
  /** 维特 TXT 的设备名 */
  dev?: string;
}

/** 块视图：把块的字段盖在侧对象上，绘图与计算都用这个。
    index 签名留给各视图缓存自己的派生量（如 IMU 页的丢帧区间 _gap），不进数据模型。 */
export type SideView = Side & Block & {[k: string]: any};

export interface SrcInfo {
  ver: string | null;
  ncols: number | null;
  tempCol: string | null;
}

/** 会话时刻 = 视频时刻 + offset。 */
export interface VideoInfo {
  url: string;
  blob: Blob;
  name: string;
  bytes: number;
  offset: number;
  offsetSource: 'estimate' | 'pack' | 'verified' | 'firstBottom' | 'motion' | 'manual';
  verifiedOffset?: number;
  /** 包内落地对时给出的不确定度（µs）；其他起点来源为 undefined */
  packUncertaintyUs?: number;
  /** 视频自身时长，元数据到达前是 NaN */
  dur: number;
  /** 包里一共几段视频（只播第一段，其余不静默假装存在） */
  total: number;
}

export type Win = {a: number; b: number} | null;

export type TabName = 'wave' | 'cop' | 'zero' | 'gap' | 'imu' | 'axis' | 'activity' | 'squat';

export interface DetailRef {
  side: SideId;
  j: number;
}


export interface ImuY {
  phys: {acc: number | null; gyro: number | null};
  raw: {acc: number | null; gyro: number | null};
}

/** 轴向推断结论：三轴各自的解剖语义与置信度 */
export interface AxisRow {
  nm: string;
  desc: string;
}

export interface AxisResult {
  err?: string;
  partial?: boolean;
  confA?: 'hi' | 'mid' | 'lo';
  confB?: 'hi' | 'mid' | 'lo';
  confC?: 'hi' | 'mid' | 'lo';
  rows?: AxisRow[];
  [k: string]: unknown;
}

export interface Seg {
  a: number;
  b: number;
  kind: string;
}
