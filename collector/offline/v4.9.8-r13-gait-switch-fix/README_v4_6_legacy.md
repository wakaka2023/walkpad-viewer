# 动作边界 v4.6：视频自动分段与中文标注

本版在用户提供的 v4.5 包上新增自动活动入口。使用视频关键点、背景相机运动补偿和时间连续性自动生成候选区间；没有把三个视频的人工时间表写进识别程序。包内模型权重、原深蹲检测参数与评分参数保持原值。原双视角综合评分说明见 `README_v4_5_legacy.md` 与 `SCORING_RULES.md`。

## 使用方法

在本目录，使用安装了 `requirements.txt` 的 Python。实测环境为 Python 3.9.25、Ultralytics 8.4.211、PyTorch 2.8.0+cu126、PyAV 15.1.0；依赖新增 Pillow 11.3.0 用于中文标注。字体默认查找 Windows 微软雅黑或 Linux Noto CJK；可用 `--font` 指定支持中文的字体。不含这些字体时回退为英文标签。

```powershell
python annotate_videos.py "C:\data\video.mp4" --output results_actions --task activities --device 0
python annotate_videos.py "C:\data\videos" --recursive --output results_actions --task activities --device 0
```

CPU 使用 `--device cpu`。`--no-preview` 仅输出区间、逐帧表和诊断。对同一输出目录使用 `--resume`，会核对视频、权重、姿态缓存与相机参数后复用缓存，并重新运行分类和导出。`--action-config my_actions.json` 可覆盖 `action_config.json` 中的参数。不要把人工起止时间作为检测参数。

也可以在 PyCharm 中编辑 `run_actions_local.py` 的输入路径、输出目录与设备后运行。多人需显式指定 `--track-id`。

## 输出

每个输入视频一个以文件名命名的子目录：

| 文件 | 含义 |
|---|---|
| `review.mp4` | 自动中文标注视频；上下边栏显示类别、时间、单次动作编号与进度条，保留完整源画面和音频 |
| `review.html` | 本地播放器与可点击的区间表 |
| `activities.csv` / `activities.json` | 连续区间、活动名称、开始/结束、可见性和判断依据 |
| `action_events.csv` | 支持完整下降-回升的单次屈伸/俯身候选起止点与峰值 |
| `boundary_events.csv` | 区间转换时间、转换前后类别与复核上下文 |
| `frame_labels.csv` | 每个源帧一个标签，含区间首尾标记与单次动作编号 |
| `rejected_candidates.csv` | 未通过膝屈伸周期判断的候选及拒绝原因 |
| `keypoints.npz` / `camera_motion.npz` / `action_features.npz` | 可审查的姿态、背景运动与数值特征缓存 |
| `metadata.json` / `preview_audit.json` | 输入哈希、统计、逐帧 PTS 与音频保留验证 |

时间为相对第一个源帧的原始 PTS 秒数，区间 `[开始,结束)`；源帧号从 1 起、首尾均包含。输出不读写传感器 CSV。

## 当前可识别范围与限制

提供站立、走动、连续迈步、较快连续迈步、局部身体持续运动、手部/姿态调整、下肢屈伸、分腿/宽站距屈伸、俯身候选及组内回位标签；证据不足保留未知。走/跑、设备操作意图及精确的分腿/宽蹲类别不能由当前二维骨架可靠确认。当前活动入口支持单摄像机画面；双画面仍用下方原有深蹲入口。

这些是**自动候选**，所有 `reviewed=0`、`needs_review=1`。证据等级不是准确率或概率。复核上下文前后 0.5 秒不是已测误差上限。屈伸边界是可观测运动阈值交叉，不保证涵盖动作前最早的准备调整。详见 `ACTION_RULES.md`。

三个开发视频全部真实运行：原始帧数 1616、2093、2580，共 6289 帧；每帧恰有一个连续标签；导出检查原始逐帧 PTS 和音频。当前未知时间占比：视频1 16.5%、视频2 9.7%、视频3 4.4%。第三个视频输出 6 次分腿/宽站距屈伸候选，第二个视频输出 4 次俯身候选；这不是专家确认的动作总数，仍存在漏检与类别不确定片段。第一个视频原版误检的站立深蹲已被排除。

共 192 项回归/规则测试通过，其中本版新增 12 项，检查相机平移、长时间关键点缺失、运动足部拒绝、真实髋下降/恢复、源时间轴覆盖及缓存身份核对。三个视频参与开发，**未测独立动作准确率或边界误差**；不能把测试通过当成识别准确率。当前导出在源画面上下增加边栏，因此输出尺寸为 720×1532，原画面 720×1280 完整保留。

## 原有深蹲与双视角评分

```powershell
python annotate_videos.py "C:\data\P48SQ2.mp4" --output results_squat --task squats --layout front-side --device 0
```

`--task` 默认仍为 `squats`。原 v4.5 的深蹲质量评分与双画面融合保持原入口；新增活动入口不把走动或俯身套用深蹲好中差评分。历史验证见带 `legacy` 的文件与原有验证目录。
