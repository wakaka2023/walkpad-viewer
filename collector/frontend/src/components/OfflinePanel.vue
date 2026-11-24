<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { cancelOffline, getOffline, submitOfflineSession, submitOfflineVideo } from '../store'

type Session = { session_id: string; status: string; frames: number }
type Job = { id: string; state: string; video: string; bytes: number; preview?: boolean; session?: string | null; elapsed_s?: number; error?: string | null; log_tail?: string[]; result?: any }
type Segment = { track_id: number; activity: string; activity_zh: string; start_video_s: number; end_video_s: number; duration_s: number; evidence_level: string; repetition_count: number }
const props = defineProps<{ sessions: Session[]; session?: string }>()

const health = ref<Record<string, any> | null>(null)
const file = ref<File | null>(null)
const preview = ref(true)
const job = ref<Job | null>(null)
const result = ref<any>(null)
const busy = ref(false)
const message = ref('')
const session = ref('')
const videos = ref<{ name: string; bytes: number }[]>([])
const chosen = ref('')
let timer: ReturnType<typeof setInterval> | undefined

const active = computed(() => !!job.value && ['uploading','queued','running'].includes(job.value.state))
const stateText: Record<string, string> = { uploading: '上传中', queued: '排队中', running: '推理中', done: '已完成', failed: '失败', cancelled: '已取消', interrupted: '已中断' }
const segments = computed<Segment[]>(() => result.value?.segments || [])
const hasVideo = computed(() => (result.value?.products || []).includes('review.mp4'))
const reviewPath = computed(() => `/api/offline/jobs/${job.value?.id}/review?v=${Math.round(result.value?.at || 0)}`)

function size(bytes: number | undefined) {
  const value = bytes || 0
  return value > 1073741824 ? `${(value / 1073741824).toFixed(2)} GB` : `${Math.max(0.1, value / 1048576).toFixed(1)} MB`
}
function seconds(value: number | null | undefined) {
  if (value == null || !Number.isFinite(value)) return '—'
  const total = Math.max(0, value)
  return `${Math.floor(total / 60)}:${(total % 60).toFixed(1).padStart(4, '0')}`
}
async function loadHealth() {
  try { health.value = await getOffline('/api/offline/config') }
  catch (exception) { health.value = { reachable: false, error: String(exception) } }
}
function pick(event: Event) {
  const picked = (event.target as HTMLInputElement).files?.[0]
  file.value = picked || null
  if (picked) { chosen.value = ''; }
}
async function start() {
  if (!file.value || busy.value || active.value) return
  busy.value = true; message.value = ''; result.value = null; job.value = null
  try {
    job.value = await submitOfflineVideo(file.value, preview.value)
    poll()
    message.value = '已提交，离线包在结果出来前不占用工作台'
  } catch (exception) { message.value = String(exception) }
  finally { busy.value = false }
}
async function startSession() {
  if (!session.value || !chosen.value || busy.value || active.value) return
  busy.value = true; message.value = ''; result.value = null; job.value = null
  try {
    job.value = await submitOfflineSession(session.value, chosen.value, preview.value)
    poll()
  } catch (exception) { message.value = String(exception) }
  finally { busy.value = false }
}
async function poll() {
  if (!job.value) return
  clearInterval(timer)
  timer = setInterval(refresh, 2000)
  await refresh()
}
async function refresh() {
  const current = job.value
  if (!current) return
  try {
    const next = await getOffline<Job>(`/api/offline/jobs/${current.id}`)
    job.value = next
    if (!active.value) {
      clearInterval(timer)
      if (next.state === 'done') await load()
    }
  } catch (exception) {
    clearInterval(timer)
    message.value = String(exception)
  }
}
async function load() {
  if (!job.value) return
  const id = job.value.id
  try { result.value = { ...(await getOffline(`/api/offline/jobs/${id}/result`)), at: Date.now() } }
  catch (exception) { message.value = String(exception) }
}
async function cancel() {
  if (!job.value) return
  try { job.value = await cancelOffline(job.value.id) }
  catch (exception) { message.value = String(exception) }
}
async function loadVideos() {
  chosen.value = ''; videos.value = []
  if (!session.value) return
  try {
    const data = await getOffline(`/api/sessions/${encodeURIComponent(session.value)}/videos`)
    videos.value = data.videos || []
    if (videos.value.length) chosen.value = videos.value[0]!.name
  } catch (exception) { message.value = String(exception) }
}
watch(() => props.session, value => { if (value) { session.value = value; void loadVideos() } })
watch(session, loadVideos)
onMounted(() => { loadHealth(); if (props.session) { session.value = props.session; void loadVideos() } })
onUnmounted(() => clearInterval(timer))
</script>

<template>
  <section class="panel offline-panel">
    <div class="panel-head">
      <h2>离线动作标注</h2>
      <span class="badge" :class="{ live: health?.reachable }">{{ health?.reachable ? '工作站在线' : '工作站不可达' }}</span>
      <button @click="loadHealth">刷新状态</button>
    </div>
    <p class="muted">整段视频事后标注：输出带 ID/动作/深蹲阶段的标注视频、逐段动作区间与逐帧表。推理在工作站显卡上排队，关掉工作台不影响已提交的任务。</p>
    <p v-if="health && !health.reachable" class="micro">{{ health.error || '无法连接离线服务，确认工作站已开机且 18844 端口可达。' }}</p>

    <div class="toolbar">
      <label class="file-label">导入视频<input type="file" accept="video/*" @change="pick"></label>
      <span v-if="file" class="micro">{{ file.name }} · {{ size(file.size) }}</span>
      <label class="preview-toggle"><input v-model="preview" type="checkbox">生成标注视频（更慢）</label>
      <button class="primary" :disabled="!file || busy || active || health?.reachable === false" @click="start">开始标注</button>
    </div>

    <div class="toolbar">
      <label>记录<select v-model="session" :disabled="busy || active"><option value="">选择记录</option><option v-for="item in sessions" :key="item.session_id" :value="item.session_id">{{ item.session_id }} · {{ item.frames || 0 }}帧</option></select></label>
      <label>录像<select v-model="chosen" :disabled="!videos.length || busy || active"><option v-for="item in videos" :key="item.name" :value="item.name">{{ item.name }} · {{ size(item.bytes) }}</option></select></label>
      <button :disabled="!chosen || busy || active || health?.reachable === false" @click="startSession">送标注</button>
      <span v-if="session && !videos.length" class="micro">该记录没有录像文件。</span>
    </div>
    <p v-if="message" class="micro">{{ message }}</p>

    <div v-if="job" class="job-card">
      <div class="job-head">
        <strong>{{ job.video }}</strong>
        <span class="badge" :class="{ live: active }">{{ stateText[job.state] || job.state }}</span>
        <span class="micro">{{ size(job.bytes) }} · 任务 {{ job.id }}<template v-if="job.elapsed_s"> · 用时 {{ job.elapsed_s }} 秒</template><template v-if="job.session"> · 归属记录 {{ job.session }}</template></span>
        <button v-if="active" @click="cancel">取消</button>
        <a v-if="job.state === 'done'" :href="`/api/offline/jobs/${job.id}/archive`">下载全部产物 ZIP</a>
      </div>
      <p v-if="job.error" class="micro error-text">{{ job.error }}</p>
      <p v-if="active" class="micro">单卡串行排队；长视频按 1 秒分片推理，标注视频需要重编码，耗时会明显变长。</p>
      <pre v-if="job.log_tail?.length" class="log-tail">{{ job.log_tail.join('\n') }}</pre>
    </div>

    <template v-if="result">
      <div class="result-head">
        <h3>动作分段 <span class="micro">{{ segments.length }} 段 · {{ result.version }} · {{ result.frame_count }} 帧 · {{ (result.people || []).length }} 人</span></h3>
      </div>
      <video v-if="hasVideo" class="review" :src="reviewPath" controls preload="metadata"></video>
      <p v-else class="micro">本次没有生成标注视频（未勾选生成标注视频）。</p>
      <table class="segments">
        <thead><tr><th>ID</th><th>动作</th><th>起止</th><th>时长</th><th>证据</th><th>重复</th></tr></thead>
        <tbody>
          <tr v-for="(segment, index) in segments" :key="index" :class="{ inferred: segment.evidence_level !== 'direct' }">
            <td>{{ segment.track_id }}</td>
            <td>{{ segment.activity_zh || segment.activity }}</td>
            <td>{{ seconds(segment.start_video_s) }} – {{ seconds(segment.end_video_s) }}</td>
            <td>{{ seconds(segment.duration_s ?? (segment.end_video_s - segment.start_video_s)) }}</td>
            <td>{{ segment.evidence_level }}</td>
            <td>{{ segment.repetition_count || '' }}</td>
          </tr>
        </tbody>
      </table>
      <p class="micro">{{ result.action_rules }}</p>
    </template>
  </section>
</template>

<style scoped>
.offline-panel .toolbar { align-items: center; }
.offline-panel .file-label { display: inline-flex; align-items: center; gap: 6px; }
.offline-panel .file-label input { max-width: 15rem; }
.offline-panel .preview-toggle { display: inline-flex; align-items: center; gap: 6px; }
.job-card { border: 1px solid #23304a; border-radius: 8px; padding: 10px 12px; margin: 10px 0; }
.job-head { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.log-tail { max-height: 7rem; overflow: auto; margin: 6px 0 0; font-size: .74rem; line-height: 1.4; color: #93a4c3; }
.error-text { color: #ff8f8f; }
.result-head { display: flex; align-items: baseline; gap: 10px; margin: 14px 0 8px; }
.review { width: 100%; max-width: 46rem; border-radius: 8px; background: #0a1120; }
.segments { width: 100%; border-collapse: collapse; font-size: .8rem; margin-top: 10px; }
.segments th, .segments td { text-align: left; padding: 4px 8px; border-bottom: 1px solid #1c2740; }
.segments th { color: #93a4c3; font-weight: 500; }
.segments tr.inferred td { color: #a8b6cf; font-style: italic; }
</style>
