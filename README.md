# Walkpad 工具集

Walkpad 鞋垫与 IMU 的两套本地工具，数据全部留在本机，不上传。

| 目录 | 应用 | 用途 | 入口 |
| --- | --- | --- | --- |
| [`viewer/`](viewer/README.md) | 数据查看器 | 打开已录制的 zip / parquet / 维特 TXT，看压力与 IMU 波形、活动分段、走跑步态报告 | 双击 `viewer/index.html`，不需要安装 |
| [`collector/`](collector/README.md) | 数据采集工作台 | 电脑直连蓝牙采集鞋垫、独立 IMU、维特设备，含零偏补偿、速度估计、导出 | `python collector/launch_workbench.py` |

## 数据查看器 `viewer/`

纯前端，`file://` 直接打开，加载仓库里预构建的 `viewer/dist/app.js`。改代码才需要 Node.js 20 以上：

```bash
cd viewer
npm install
npm run build       # 产物写到 dist/app.js
npm run typecheck
npm test
```

三个页面：`index.html`（主查看器）、`axis.html`（IMU 轴向推断）、`imu.html`（维特单设备波形）。模型权重以 JSON 内联进 bundle，运行时不联网。指标口径见 [`viewer/README.md`](viewer/README.md)。

## 数据采集工作台 `collector/`

Python 后台 + Vue 前端，在浏览器里操作。Windows 双击 `collector/open_workbench.cmd`，或：

```bash
pip install -r collector/requirements-workbench.txt
python collector/launch_workbench.py          # 起服务并打开页面
python collector/launch_workbench.py --stop   # 关掉后台
```

支持三类设备：自研鞋垫（D56，带压力）、自研独立 IMU（同协议、无压力通道）、维特 WT901 系列。零偏补偿对自研两类设备都生效，维特走原始值。录到的帧按 app 同构的列名导出 parquet。设备与协议细节见 [`collector/README.md`](collector/README.md)。

## `imu-cop`

[`imu-cop/`](imu-cop/README.md) 是独立的 Python 实验模块，用双脚六轴 IMU 以 50 Hz 估计每只脚的压力中心。右脚使用承重门控遮蔽摆动帧，配套权重可从 `imu-cop` 预发布下载。

## 约定

- 采集数据、导出的 parquet、`collector/runtime/` 与 `collector/records/` 不入库；解包的 APK（`apk_labpad/`）只在本地做协议参考，也不入库。
- 两个应用的构建产物（`viewer/dist`、`collector/frontend/dist`）与模型权重（查看器的 JSON、采集器的 ONNX）都入库，克隆下来即可离线用。
