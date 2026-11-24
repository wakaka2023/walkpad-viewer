# Walkpad 数据采集工作台

电脑直连蓝牙采集鞋垫与 IMU，浏览器里实时看波形、零偏与速度，录完导出 parquet。数据只落本机 `records/`，不上传。

## 快速开始

Windows 双击 `open_workbench.cmd`；其它平台：

```bash
pip install -r requirements-workbench.txt
python launch_workbench.py          # 起后台并在浏览器打开，端口从 8875 起找空闲的
python launch_workbench.py --stop   # 关闭后台
python launch_workbench.py --legacy # 旧版维特多设备页面
```

后台是 FastAPI，只监听 `127.0.0.1`，控制接口要带启动时生成的一次性 token（页面自动带上）。浏览器全部断开后后台会自行退出，可用 `--idle-exit` 调整宽限。

## 支持的设备

| 设备 | 识别方式 | 协议 | 压力 | 零偏补偿 |
| --- | --- | --- | --- | --- |
| 自研鞋垫（D56） | 广播 0x1111，产品类型 1 | 同下，`46+2n` 字节帧，56 点压力 | 有 | 有 |
| 自研独立 IMU | 广播 0x1111，产品类型 3 | 与鞋垫同一套（去压力通道） | 无 | 有 |
| 维特 WT901 / JY901 | 名称前缀 `WT` | `0x55 0x61` 帧 | 无 | 无，用原始值 |

扫描优先解析广播里的厂商数据（公司号 `0x1111`：布局版本、产品类型、左右脚、配对号），广播未登记时才回退到名称前缀。自研两类设备共用 GATT（notify `2a37` / write `ffe1`）与六轴换算 `(raw_y, -raw_x, raw_z)`，加速度 0.488 mg/LSB、陀螺 0.07 dps/LSB。

## 零偏

按设备记账，鞋垫与独立 IMU 都走同一条链路（静止检测 → 陀螺零偏估计 → 逐帧扣除），四种操作：自动跟踪、手动锁定、静止校准、重置。维特设备不参与。速度估计在自研设备上引用这份零偏做误差状态约束。逐帧偏置与就绪状态落盘，导出时写成 `imuCalibration` 的补偿列，原始角速度不被改写。

## 采集与导出

点击「停止并保存」，待写盘完成后，可从页首直接导出本次「原始 ZIP」或「测量包（查看器）」。其他格式和过往记录在「历史记录与回放」。

- 落盘：`records/<session>/frames_*.parquet`，`schema_version=capture_frames.v3`，含原始帧十六进制、六轴原值与物理量、设备日历、`device_kind`，以及逐帧零偏四列（`imu_bias_dps`、`imu_bias_ready`、`imu_bias_revision`、`imu_bias_gate`）。
- 标准导出（`format=standard`）：列名与 app 的 `standard_measurement_frames` 同构，末尾多 5 个自定义列（`placement_code`、`gyro_temperature_celsius`、`device_date`、`device_time`、`dropped_frames_before`），在 parquet 的 kvMetadata 里登记。
- 测量包导出（`format=package`）：与 labpad 的 `lab_measurement_package.v2.6` 同构的 zip。成员为 `manifest.json`（`physicalSchemas` 写完整列定义，`sessionTimeline` 是排除暂停的有效周期表，`commonAlignedWindow` 是各来源有效帧覆盖在主机接收单调钟上的交集，`videos` 列每个录像的资产与对齐状态）、`frames/<设备目录>/frame-00001.parquet`、`timeline/footer.parquet`（sections `periods`/`segments`/`eventOrder`/`mappings`/`videos`）与 `设备目录对照.txt`；设备目录名取 `<deviceFamily>_<蓝牙名>_<MAC首2位>-<MAC末4位>`，帧表 26 列，ZSTD 压缩、行组 8192。鞋垫写 `frameType=insole` 与 56 通道压力，独立 IMU 写 `inertial` 且压力三项为 null。
- 录像对齐：工作台只记录像首末块到达的主机时刻，没有逐帧对时表，所以 `videos[].alignmentStatus=unverified`、`frameAlignmentMode=full_capture`，`periodId` 指首块到达时所在周期；画面帧不与采集帧对齐，也不计入 `commonAlignedWindow`。
- 零偏列：该帧有扣偏结果时写 `imuCalibration.status='ready'`、`gyroBiasDps`、`angularVelocityCompensatedDps`（＝原值 − 偏置，与 `angularVelocityDps` 同属 (y,−x,z) 输出坐标）与 `biasRevision`；未就绪写 `uncalibrated`、`biasRevision=0`、补偿与偏置列 null。`angularVelocityDps` 始终是原始换算值，不被覆盖。
- 与 app 的两处差异：本工作台没有手机—设备对时，`sourceZeroUs` 取该设备首帧设备日历、`offsetBasis=legacy_zero_fallback`；不保存静置原件，故 `calibrationOriginals=[]`、`initializationId` 写成 `<sourceId>-bias-<revision>`，包内 `设备目录对照.txt` 有同样说明。
- 独立 IMU 导出 `pressure_status='absent'`、压力数组留空，不写 70 个 0 冒充实测零压。
- CSV 转 parquet：`python csv2parquet.py <in.csv> [out.parquet]`。

## 摄像头实时动作识别

在采集工作台勾选「摄像头」，开启设备后选择「ID 识别模型」和「动作规则」，再点「实时动作识别」。预览层显示 ID、动作或候选及已完成的深蹲次数。

- **采集身份复核（GPU，默认）**：先让所有受试者同时入画，等每人都有 ID 后再开始动作。首位已登记人员离画后不再自动发新号。实时只有高置信、一对一的轨迹才接回原 ID；结束识别或停录后展示复入候选，未经复核不计入动作。
- **ByteTrack（原版）**：使用跟踪器轨迹 ID；离场、遮挡后可能换号。
- **人体重识别（ReID）**：同一次实时识别内，人员离场再入画时尝试恢复原 ID 和已确认的深蹲次数，不续接未完成动作。特征只留在当前会话内存，不识别人脸、不绑定设备；相似外观、遮挡或角度变化仍可能换号或错配。
- **TrackTrack + ReID（GPU）**：可选的短时跟踪 + 会话内复入模式，只在「推理：工作站」的 GPU 上提供，本机不加载。选择「推理：本机」会直接报错并提示需要工作站，不会自动降级为 ReID 或 ByteTrack；状态栏显示 `TrackTrack/ReID GPU`。身份尚未确认的轨迹只在状态栏显示「待确认身份 N」，不进入 people、不参与动作与深蹲统计。工作站按 ALPN 区分该模式，旧模式与旧工作站照常兼容。
- **v4.9.8-r13**（默认）：每人取最近 12 秒姿态窗口约每秒重算一次。直接证据不足的动作统一显示「动作候选」（例如「走路候选」）。本版新增原地转身识别：同一轨迹内肩与髋的有向投影同向变化、脚相对背景重新落位、净位移不足以解释为连续走路，抖动、只转头扭肩与相机平移分别排除；转身与走路、跑步折返、深蹲分开判定。步态中的断点与近镜头裁切只补候选：同轨同类可靠步态给出本人周期参考，断点内实测双腿运动吻合才补，短停、丢轨、异类证据与已确认深蹲不被插值覆盖。投影俯身降为审计证据，不再作为主动作种子。玻璃隔断后的人像和墙上照片的自动排除只在离线成片流程里生效，实时窗口不启用；窗口短于离线整段视频，结果不保证一致。
- **实时分相（试用）**：站立和深蹲用逐人短时姿态；看清站定、下降、蹲底并回位后才计一次，不跨断点补数。走跑和楼梯分别用步态与连续跨阶证据；楼梯宜用固定机位并拍全双脚。只在有明确非目标姿态时显示「其他」，证据不足显示「未判定」。输出用于实时预览，不是导出的正式动作标注。

离线成片流程使用 `offline/v4.9.8-r13-gait-switch-fix`：在双侧连续性判定之前，用同一 ID 自己已确认的走路与跑步片段作参考，比较骨盆与肩在相机补偿后的信号、双侧踝行程和膝屈曲差异；观测到的步周期只决定窗口长度，不用来比较哪一类更快。另有原地转身的错时判定与跨片段参考补齐，权重、动作参数与轨道 ID 不变。只有视频内同时存在该人独立支撑的走路和跑步样本、且交替迈步与共同起伏持续出现时才改写，缺视角、相机未对齐、深蹲事件、部分轨迹与走跑以外的标签一律保留原判。实时页跑的是 12 秒窗口规则，不启用该分支。

切换模型、规则、旋转或镜像需重新开启识别；新识别会话不沿用旧 ID。识别不写入录像、IMU 记录或导出文件。需要与离线成片一致的结论时，仍以离线包处理录像为准。

公司采集网内默认直连 `192.168.100.29:18843`，使用工作站「采集身份复核（GPU）」与 r13。旧客户端选 r5/r6 时工作站会实际运行并回报 r13；更早的规则不再支持。客户端不用放权重、装 CUDA、配地址或持有个人证书。连接使用 TLS，仓库内的 `backend/vision_lan_ca.crt` 仅用于验证工作站，并不是登录凭证；只传识别帧和结果，识别帧不在工作站保存，录像及 IMU 仍在本机。选择本机推理才需要本机权重，GPU 专用模式也不可用。

工作站以 `python -m backend.vision_workstation --host 192.168.100.29 --port 18843 --allow-lan 192.168.100.0/24` 启动，绑定指定地址并仅为指定采集网段提供免客户端证书接入；同网段能访问端口的设备均可使用，不得转发到公网。只允许单会话，选择一张低负载且有足够显存的 GPU；少量驻留显存不影响选择。省略 `--allow-lan` 则要求双向 TLS。客户端可用 `runtime/vision/workstation.json` 覆盖 `host`、`port`；非默认地址默认要求双向 TLS，也可显式设 `"client_auth": false` 并配置该服务的 `runtime/vision/tls/ca.crt`。恢复默认地址的双向 TLS 时，显式设 `"client_auth": true` 并放好客户端证书和密钥。

实际执行视觉推理的一端加载 `runtime/vision/yolo26s-pose.pt`（启动时校验 SHA-256）。CPU ReID 使用 Open Model Zoo 的 `person-reidentification-retail-0288`（FP32 `.xml`/`.bin`）；工作站 GPU ReID 使用 `runtime/vision/yolo26n-reid.onnx` 和 CUDA 12.8 兼容的 `onnxruntime-gpu==1.26.0`（`runtime/vision/reid-cu128-deps/`）。权重、运行依赖、私钥及本地配置不入库。

同事可以在团队协作仓开分支修改 `backend/vision_*` 或 `backend/rules_r4/`。选择工作站时执行的是**工作站部署的代码**：分支部署到工作站独立测试服务后再验证，通过后用对应 Git 版本更新共用服务。只修改自己的采集电脑不会改变共享算法；连接成功后界面会显示工作站服务版本。

## 目录

```
backend/      FastAPI 服务：蓝牙采集、协议解码、零偏、速度、导出
frontend/     Vue 工作台页面（dist/ 是预构建产物）
activity/     活动识别：ONNX 权重、预处理、对拍脚本
js/ css/      旧版维特多设备页面（wt901ble_live.html）
test/         Node 端到端脚本（部分需要 Chrome 调试端口）
tests_v2/     pytest：后端接口与速度估计
offline/      离线成片动作标注包（整包不入库，权重放 models/ 并校验 SHA-256）
```

## 测试

```bash
pytest tests_v2
node test/check_bias.cjs        # 零偏 worker
node test/check_press.js        # 导出列与协议解析
```

需要浏览器的脚本依赖 Chrome 远程调试端口，未启动浏览器时会卡住等待。

## 已知限制

- 独立 IMU 没有压力通道，不参与压力相关的步态推断；活动识别只出 IMU 侧结果。
- 手机 App 链路（`backend/phone.py`）按协议名定设备类型，尚未识别广播里的产品类型 3。
- 设备端 IMU 标定命令（0x18 标定 / 0x19 清除 / 0x0d 查状态）未接入，零偏目前是软件侧估计。
- 解包的 APK 只作协议参考，不入库。
