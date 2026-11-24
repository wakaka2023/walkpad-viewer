import { shallowRef, ref } from 'vue'

export interface Attitude {
  degrees: [number, number, number | null]
  reference_id: string
}
export interface Frame {
  cursor: number
  device_id: string
  foot: string | null
  bias_key?: string
  profile: 'wt' | 'insole'
  receipt_ms: number
  epoch_ms: number
  values: (number | null)[] | null
  attitude?: Attitude | null
  pressure?: number[]
  temperature?: number
  device_ms?: number | null
  seq?: number
  // 服务端同帧生效的偏置；不能借后续快照补偿历史帧
  bias_used?: number[]
}
export interface Device {
  id: string; name: string; address?: string; profile: 'wt' | 'insole'; kind: 'insole' | 'imu' | 'wt'; has_pressure: boolean; foot: string | null; bias_key: string; placement: string | null; placement_label: string | null; source: string
  connected: boolean; reconnecting: boolean; frames: number; bad: number; rate: number
  last: Frame | null; topology: string; format: number | null; format_status: string; command_status: string
  attitude: Attitude | null
}
export interface Status {
  version: string; source: string; devices: Device[]; cursor: number; demo_allowed: boolean
  recorder: { recording: boolean; paused: boolean; session_id: string | null; path: string; frames: number; chunks: number; events: number; queued: number; periods: number; effective_us: number; error: string | null; recovered: string[]; draining: boolean }
  algorithm: { enabled: boolean; profile: string; result: Record<string, any>; bias: Record<string, any>; error: string | null; dropped: number; model?: Record<string, any> }
  phone: Record<string, any>; error: string | null; warning: string
  logs: { time: number; text: string; level: string }[]
}
export const status = shallowRef<Status | null>(null)
export const online = ref(false)
export const idleExit = ref(0)
export const error = ref('')
export const revision = ref(0)
export const buffers = new Map<string, Frame[]>()
let token = ''
let socket: WebSocket | null = null
let retry: ReturnType<typeof setTimeout> | undefined
let stopped = false
let previousServer = ''

export async function api<T = any>(action: string, args: Record<string, any> = {}): Promise<T> {
  const response = await fetch('/api/control', {
    method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Walkpad-Token': token },
    body: JSON.stringify({ action, ...args }),
  })
  const result = await response.json()
  if (!response.ok) throw new Error(result.detail || result.error || '请求失败')
  return result
}
export async function upload(path: string, blob: Blob) {
  const response = await fetch(path, { method: 'POST', headers: { 'X-Walkpad-Token': token, 'Content-Type': blob.type || 'video/webm' }, body: blob })
  const result = await response.json()
  if (!response.ok) throw new Error(result.detail || '录像写盘失败')
  return result
}
export async function submitOfflineVideo(file: File, preview: boolean) {
  const response = await fetch('/api/offline/jobs', {
    method: 'POST',
    // 头部只能是 ISO-8859-1，中文文件名必须先百分号编码
    headers: { 'X-Walkpad-Token': token, 'X-Filename': encodeURIComponent(file.name), 'X-Preview': preview ? '1' : '0' },
    body: file,
  })
  const result = await response.json()
  if (!response.ok) throw new Error(result.detail || result.error || '提交失败')
  return result
}
export async function submitOfflineSession(session: string, video: string, preview: boolean) {
  return postOffline('/api/offline/jobs/from-session', { session, video, preview })
}
export async function cancelOffline(job: string) {
  return postOffline(`/api/offline/jobs/${encodeURIComponent(job)}/cancel`, {})
}
async function postOffline<T = any>(path: string, body: Record<string, any>): Promise<T> {
  const response = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Walkpad-Token': token },
    body: JSON.stringify(body),
  })
  const result = await response.json()
  if (!response.ok) throw new Error(result.detail || result.error || '请求失败')
  return result
}
export async function getOffline<T = any>(path: string): Promise<T> {
  const response = await fetch(path)
  const result = await response.json()
  if (!response.ok) throw new Error(result.detail || result.error || '请求失败')
  return result
}
export type VisionTarget = 'workstation' | 'local'
export type IdModel = 'bytetrack' | 'reid' | 'tracktrack-reid' | 'collection-reid'
export type ActionRules = 'r13' | 'live'

export function openVisionSocket(target: VisionTarget, idModel: IdModel, actionRules: ActionRules) {
  if (!token) throw new Error('工作台未连接')
  return new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws/vision?target=${target}&id_model=${idModel}&action_rules=${actionRules}`, ['walkpad-vision', token])
}
export async function connect() {
  stopped = false
  try {
    const config = await (await fetch('/api/config')).json()
    token = config.control_token
    idleExit.value = Number(config.idle_exit) || 0
    if (previousServer && previousServer !== token) buffers.clear()
    previousServer = token
    socket = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws?token=${encodeURIComponent(token)}`)
    socket.onopen = () => { online.value = true }
    socket.onmessage = (event) => {
      const update = JSON.parse(event.data)
      status.value = update.state
      if (update.gap) error.value = '显示缓存出现缺口；请以后台保存的原始记录为准'
      for (const frame of update.frames as Frame[]) {
        const array = buffers.get(frame.device_id) || []
        if (!array.length || frame.cursor > array[array.length - 1]!.cursor) array.push(frame)
        const cutoff = frame.receipt_ms - 35000
        let cut = 0
        while (cut < array.length && array[cut]!.receipt_ms < cutoff) cut++
        if (cut) array.splice(0, cut)
        if (array.length > 6000) array.splice(0, array.length - 6000)
        buffers.set(frame.device_id, array)
      }
      revision.value++
    }
    socket.onclose = () => {
      online.value = false
      if (!stopped) retry = setTimeout(connect, 1500)
    }
    socket.onerror = () => { online.value = false }
  } catch (e) {
    online.value = false
    error.value = String(e)
    if (!stopped) retry = setTimeout(connect, 1500)
  }
}
export function disconnect() {
  stopped = true
  clearTimeout(retry)
  socket?.close()
}
export function clearBuffers() { buffers.clear(); revision.value++ }
export function dropBuffers(id: string) { buffers.delete(id); revision.value++ }
export const number = (value: number | null | undefined, digits = 2) => value == null || !Number.isFinite(value) ? '—' : value.toFixed(digits)
export const side = (value: string | null) => value === 'l' ? '左脚' : value === 'r' ? '右脚' : '未指定侧别'
export const places: Record<string, string> = { leftInstep: '左脚背', rightInstep: '右脚背', leftWrist: '左手腕', rightWrist: '右手腕', leftAnkle: '左踝', rightAnkle: '右踝', leftShank: '左小腿', rightShank: '右小腿', leftThigh: '左大腿', rightThigh: '右大腿', pelvis: '骨盆' }
export const where = (device: Pick<Device, 'foot' | 'placement' | 'placement_label'>) => device.placement_label || places[device.placement || ''] || side(device.foot)
