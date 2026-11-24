<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { api, status, upload, type ActionRules, type IdModel, type VisionTarget } from '../store'
import { cameraVision } from '../cameraVision'
const video = ref<HTMLVideoElement>()
const canvas = ref<HTMLCanvasElement>()
const devices = ref<MediaDeviceInfo[]>([])
const selected = ref('')
const resolution = ref('1280x720')
const fps = ref(30)
const linked = ref(true)
const active = ref(false)
const recording = ref(false)
const rotate = ref(0)
const mirror = ref(false)
const message = ref('录像按块写入本次记录目录；无需在内存累积整段视频。')
const transformed = computed(() => rotate.value !== 0 || mirror.value)
let stream: MediaStream | undefined
let recorder: MediaRecorder | undefined
let painted: MediaStream | undefined
let raf = 0
let chain: Promise<void> = Promise.resolve()
let pending = 0
let bytes = 0
let failed = false
let starting = false
let session = ''
let stopPromise: Promise<void> | undefined
let resolveStop: (() => void) | undefined
async function enumerate() {
  devices.value = (await navigator.mediaDevices.enumerateDevices()).filter(device => device.kind === 'videoinput')
}
// 预览与录像共用一套变换：MediaRecorder 录的是流，DOM 上的 transform 进不了文件，
// 旋转只能靠 canvas 重绘后再 captureStream。
function render(target: HTMLCanvasElement, limit = Infinity) {
  const v = video.value
  const w = v?.videoWidth || 0, h = v?.videoHeight || 0
  if (!v || !w || !h) return
  const swap = rotate.value % 180 !== 0
  const scale = Math.min(1, limit / Math.max(w, h))
  const cw = Math.round((swap ? h : w) * scale), ch = Math.round((swap ? w : h) * scale)
  if (target.width !== cw || target.height !== ch) { target.width = cw; target.height = ch }
  const ctx = target.getContext('2d')
  if (!ctx) return
  ctx.save()
  ctx.translate(cw/2, ch/2)
  ctx.rotate(rotate.value * Math.PI/180)
  if (mirror.value) ctx.scale(-1, 1)
  ctx.drawImage(v, -w*scale/2, -h*scale/2, w*scale, h*scale)
  ctx.restore()
}
const visionTarget = ref<VisionTarget>('workstation')
const idModel = ref<IdModel>('collection-reid')
const actionRules = ref<ActionRules>('r13')
const { enabled: visionOn, message: visionMessage, people, review, start: startVision, stop: stopVision,
  finish: finishVision } =
  cameraVision(() => video.value, target => render(target, 960))
watch(visionTarget, target => {
  if (target === 'local' && (idModel.value === 'collection-reid' || idModel.value === 'tracktrack-reid')) idModel.value = 'bytetrack'
})
const overlayStyle = ref<Record<string, string>>({})
const preview = ref<HTMLElement>()
let resize: ResizeObserver | undefined
function sizeOverlay() {
  const v = video.value, c = canvas.value
  const element = transformed.value ? c : v
  if (!element || !v?.videoWidth || !preview.value) return
  const outer = preview.value.getBoundingClientRect(), r = element.getBoundingClientRect()
  const swap = rotate.value % 180 !== 0
  const w = swap ? v.videoHeight : v.videoWidth
  const h = swap ? v.videoWidth : v.videoHeight
  const fit = Math.min(r.width / w, r.height / h)
  const width = w * fit, height = h * fit
  overlayStyle.value = { left: `${r.left-outer.left+(r.width-width)/2}px`, top: `${r.top-outer.top+(r.height-height)/2}px`,
    width: `${width}px`, height: `${height}px` }
}
function toggleVision() {
  if (!visionOn.value) { startVision(visionTarget.value, idModel.value, actionRules.value); return }
  finishVision()
}
function loop() { const c = canvas.value; if (c) render(c); raf = requestAnimationFrame(loop) }
function startLoop() {
  cancelAnimationFrame(raf); raf = 0
  if (transformed.value && active.value && canvas.value) raf = requestAnimationFrame(loop)
}
function stopLoop() { cancelAnimationFrame(raf); raf = 0 }
function releasePainted() { painted?.getTracks().forEach(track => track.stop()); painted = undefined }
// 录像源跟着变换走：不旋转时直接录摄像头流，旋转/镜像时录重绘后的 canvas
function recordSource() {
  if (!transformed.value) return stream
  startLoop()
  painted = canvas.value!.captureStream(fps.value)
  return painted
}
async function open() {
  try {
    const [width, height] = resolution.value.split('x').map(Number)
    stream = await navigator.mediaDevices.getUserMedia({ video: { deviceId: selected.value ? { exact: selected.value } : undefined, width: { ideal: width }, height: { ideal: height }, frameRate: { ideal: fps.value } }, audio: false })
    video.value!.srcObject = stream
    active.value = true
    await enumerate()
    const settings = stream.getVideoTracks()[0]!.getSettings()
    message.value = `已开启 ${settings.width}×${settings.height} · ${settings.frameRate?.toFixed(1)} fps（实际协商值）`
  } catch (error) { message.value = String(error) }
}
async function start() {
  if (!stream || recording.value || starting) return
  starting = true
  try {
    if (!status.value?.recorder.recording && !status.value?.recorder.paused) await api('record_start')
    const current = await (await fetch('/api/status')).json()
    session = current.recorder.session_id
    // 优先 mp4：离线标注包只认 mp4/mov/mkv 一类容器，webm 需要服务端重封装
    const mime = ['video/mp4;codecs=avc1', 'video/mp4', 'video/webm;codecs=vp8', 'video/webm']
      .find(type => MediaRecorder.isTypeSupported(type)) || ''
    const source = recordSource()
    if (!source) throw new Error('录像源不可用')
    recorder = new MediaRecorder(source, mime ? { mimeType: mime, videoBitsPerSecond: 4_000_000 } : undefined)
    const id = crypto.randomUUID().replaceAll('-', '')
    let index = 0
    pending = 0; bytes = 0; failed = false; chain = Promise.resolve()
    stopPromise = new Promise(resolve => { resolveStop = resolve })
    recorder.ondataavailable = event => {
      if (!event.data.size || failed) return
      pending += event.data.size
      if (pending > 48 * 1024 * 1024) {
        failed = true
        message.value = '录像写盘跟不上，已停止录像；此前已写盘部分保留，IMU继续记录。'
        if (recorder?.state === 'recording') recorder.stop()
        return
      }
      const part = index++
      chain = chain.then(async () => {
        if (failed) return
        await upload(`/api/sessions/${encodeURIComponent(session)}/video/${id}/${part}`, event.data)
        bytes += event.data.size
        message.value = `录像已写盘 ${(bytes / 1048576).toFixed(1)} MB · 待写 ${(pending / 1048576).toFixed(1)} MB`
      }).catch(error => {
        failed = true; message.value = `录像写盘失败：${error}；已写盘部分保留`
        if (recorder?.state === 'recording') recorder.stop()
      }).finally(() => { pending -= event.data.size })
    }
    recorder.onstop = async () => {
      await chain
      recording.value = false
      releasePainted()
      if (!failed) message.value = `录像已保存 ${(bytes / 1048576).toFixed(1)} MB · 随本次记录ZIP导出`
      resolveStop?.()
    }
    recorder.onerror = event => { message.value = '录像编码错误：' + String(event); failed = true }
    recorder.start(1000)
    recording.value = true
  } catch (error) { message.value = String(error) }
  finally { starting = false }
}
async function stop() {
  if (recorder?.state === 'recording') recorder.stop()
  if (stopPromise) await stopPromise
  releasePainted()
}
async function close() { stopVision(); await stop(); stream?.getTracks().forEach(track => track.stop()); stream = undefined; active.value = false; stopLoop() }
async function pip() {
  try { await video.value?.requestPictureInPicture() } catch (error) { message.value = String(error) }
}
async function shot() {
  if (!video.value || !active.value) { message.value = '先开启摄像头再拍照'; return }
  // 拍照走同一套变换，图里看到的和文件里的一致
  const canvas = document.createElement('canvas')
  if (transformed.value) {
    render(canvas)
  } else {
    canvas.width = video.value.videoWidth; canvas.height = video.value.videoHeight
    canvas.getContext('2d')!.drawImage(video.value, 0, 0)
  }
  canvas.toBlob(async blob => {
    if (!blob) return
    const anchor = document.createElement('a'); anchor.href = URL.createObjectURL(blob); anchor.download = `camera_${new Date().toISOString().replaceAll(':', '-')}.png`; anchor.click(); URL.revokeObjectURL(anchor.href)
  })
}
// 旋转/镜像只改变绘制方式，录像中途改会让码流几何跳变，所以录像时锁住
watch([rotate, mirror], () => {
  if (!recording.value) startLoop()
  if (visionOn.value) {
    stopVision()
    visionMessage.value = '画面方向已变更，可重新开启动作识别'
  }
  sizeOverlay()
}, { flush: 'post' })
watch(() => status.value?.recorder.recording, async (value, old) => {
  if (value && !old && linked.value && active.value && !recording.value) await start()
  if (!value && old) {
    if (visionOn.value && idModel.value === 'collection-reid') finishVision()
    if (recording.value) await stop()
  }
})
function unload(event: BeforeUnloadEvent) { if (recording.value) { event.preventDefault(); event.returnValue = '' } }
onMounted(() => {
  enumerate().catch(() => {})
  window.addEventListener('beforeunload', unload)
  resize = new ResizeObserver(sizeOverlay)
  if (video.value) resize.observe(video.value)
  if (canvas.value) resize.observe(canvas.value)
})
onUnmounted(() => { resize?.disconnect(); close(); window.removeEventListener('beforeunload', unload) })
defineExpose({ stop, recording })
</script>

<style scoped>
.preview-canvas { object-fit: contain; }
.vision-overlay { position: absolute; pointer-events: none; overflow: hidden; }
.vision-person { position: absolute; border: 2px solid #58edc4; color: #102032; }
.vision-person.candidate { border-color: #ffc85a; }
.vision-person.unknown { border-color: #dbe5f7; }
.vision-person span { display: inline-block; white-space: nowrap; background: #fff; border-radius: 0 0 4px 0; padding: 2px 5px; font-size: 12px; font-weight: 650; }
</style>
<template><section class="panel camera-panel"><div class="panel-head"><h2>摄像头</h2><span class="badge" :class="{ live: recording }">{{ recording ? '录像写盘中' : active ? '预览中' : '未开启' }}</span></div><div class="toolbar"><select v-model="selected" :disabled="active"><option value="">默认摄像头</option><option v-for="device in devices" :key="device.deviceId" :value="device.deviceId">{{ device.label || device.deviceId }}</option></select><select v-model="resolution" :disabled="active"><option>640x480</option><option>1280x720</option><option>1920x1080</option><option>3840x2160</option></select><select v-model="fps" :disabled="active"><option :value="15">15 fps</option><option :value="30">30 fps</option><option :value="60">60 fps</option></select><button v-if="!active" @click="open">开启摄像头</button><button v-else @click="close">关闭</button><button :disabled="!active" @click="pip">画中画</button><button :disabled="!active" @click="shot">拍照</button><select v-model="visionTarget" :disabled="visionOn" aria-label="推理位置"><option value="workstation">推理：工作站</option><option value="local">推理：本机</option></select><label title="ID 仅在本次识别会话内有效。切换前请先关闭动作识别。TrackTrack + ReID 需在工作站 GPU 上运行，本机会直接报错，不会自动切换模式。">ID 识别模型 <select v-model="idModel" :disabled="visionOn"><option value="collection-reid" :disabled="visionTarget !== 'workstation'">采集身份复核（GPU）</option><option value="bytetrack">ByteTrack（原版）</option><option value="reid">人体重识别（ReID）</option><option value="tracktrack-reid" :disabled="visionTarget !== 'workstation'">TrackTrack + ReID（GPU）</option></select></label><label title="实时分相：短时判断站立与深蹲，结合步态和连续跨阶判断走跑与楼梯。证据不足时保留未判定。">动作规则 <select v-model="actionRules" :disabled="visionOn"><option value="r13">v4.9.8-r13</option><option value="live">实时分相（试用）</option></select></label><button :disabled="!active" :aria-pressed="visionOn" @click="toggleVision">{{ visionOn ? '关闭动作识别' : '实时动作识别' }}</button></div><div ref="preview" class="camera-preview"><video ref="video" autoplay playsinline muted :class="{ veiled: transformed }" @loadedmetadata="sizeOverlay" @resize="sizeOverlay"></video><canvas v-show="transformed" ref="canvas" class="preview-canvas"></canvas><div v-if="visionOn" class="vision-overlay" :style="overlayStyle"><div v-for="person in people" :key="person.id" class="vision-person" :class="person.status" :style="{ left: `${person.box[0]*100}%`, top: `${person.box[1]*100}%`, width: `${(person.box[2]-person.box[0])*100}%`, height: `${(person.box[3]-person.box[1])*100}%` }"><span>ID {{ person.id }} · {{ person.action }}<template v-if="person.count"> · 深蹲 {{ person.count }} 次</template></span></div></div></div><div class="toolbar"><select v-model.number="rotate" :disabled="!active || recording"><option :value="0">方向：原始</option><option :value="90">方向：顺时针90°</option><option :value="180">方向：180°</option><option :value="270">方向：顺时针270°</option></select><label><input v-model="mirror" type="checkbox" :disabled="!active || recording"> 镜像</label><label><input v-model="linked" type="checkbox"> 随传感器记录启停</label><button :disabled="!active || recording" @click="start">开始录像</button><button :disabled="!recording" @click="stop">停止录像</button></div><p v-if="active && visionMessage" class="micro" role="status">{{ visionMessage }}</p><p class="muted">{{ message }}</p><p class="micro">方向与镜像对预览、拍照和录像文件同时生效（录像走 canvas 重绘，不是给画面加 CSS 旋转）；录像中锁定，避免码流分辨率中途跳变。传感器记录不依赖页面；摄像头仍由浏览器持有，刷新前请先停止录像。同步点击启停不代表帧级时间同步。</p><details v-if="review">
  <summary>身份复核 · {{ review.tracks.filter(row => row.status === 'needs_review').length }} 段待人工确认</summary>
  <p>时间以摄像头预览时间轴为准，不是 IMU 或导出录像的起点。推荐编号未计入动作。</p>
  <p v-if="review.overflow">轨迹数达到会话容量，后续轨迹未进入复核。</p>
  <ul><li v-for="row in review.tracks" :key="row.track_id">
    {{ row.start_s.toFixed(2) }}–{{ row.last_seen_s.toFixed(2) }} 秒 ·
    {{ row.person_id ? `ID ${row.person_id}${row.status === 'live_reconnected' ? '（已实时接回）' : ''}` : row.suggested_id ? `建议 ID ${row.suggested_id}（待复核）` : '身份未确定' }}
  </li></ul>
</details></section></template>
