# v4.7 自动动作识别与边界

输入一个视频，自动输出活动类别、连续活动段、单次动作起止与峰值、逐帧阶段。不要求输入动作次数或人工起止时间。模型权重、原深蹲参数和双视角评分保持原版；本次改动为活动推断、时间连续性、俯身恢复边界与显示。

## 运行

在本目录安装 `requirements.txt` 后运行：

```powershell
python recognize_actions.py "C:\data\video.mp4" --output results_actions
python recognize_actions.py "C:\data\videos" --recursive --output results_actions
```

默认自动选择 CUDA 或 CPU；`--device cpu` 可强制 CPU。已有目录使用 `--resume`；`--no-preview` 只输出表格。中文字体默认查找微软雅黑/Noto CJK，可用 `--font` 指定。PyCharm 可编辑 `run_actions_local.py` 的输入路径后运行。多人画面需显式指定 `--track-id`。原入口 `annotate_videos.py --task squats` 保留；活动入口也可用 `annotate_videos.py --task activities`。

## 类别和输出层次

自动区分站立/等待、走路/移动、连续迈步、准备/手部调整、下蹲/起立、宽站距下肢屈伸、俯身/起身；下肢出画时可保留“持续运动”，不足以判断时保留未知。准备/手部调整不等于已确认按按钮；连续迈步不等于已确认跑步，当前走/跑仍不能可靠区分。

| 输出 | 内容 |
|---|---|
| `review.mp4` | 主标题显示完整活动段，次行显示组内第几次、下降/回升或组间回位；不因每次回位反复更改主标题 |
| `activities.csv/json` | 活动段、起止、组内次数和事件列表 |
| `action_events.csv` | 每次下肢屈伸/俯身的开始、峰值、结束与判断依据 |
| `frame_labels.csv` | 每个源帧的主动作、单次动作编号、`phase`、原始类别 `raw_activity`、观测状态 |
| `boundary_events.csv` | 主活动转换时间；不把每个重复动作当作一次主类别转换 |
| `raw_activities.csv` / `raw_frame_labels.csv` | 稳定主活动之前的规则标签，便于审查合并过程 |
| `rejected_candidates.csv` | 膝屈伸/俯身候选被拒绝的原因 |
| `review.html` | 本地视频与可点击的活动区间 |
| `metadata.json` / `preview_audit.json` | 输入哈希、推断比例、原始时间轴和音频保留检查 |

`phase` 的 down/up 为投影运动下降/回升，between_reps 为组内回位。短暂缺失只在限定时间和相邻活动支持下推断，`context_inferred=1`，不伪装成直接观测。`observation_status` 区分 observed、partial、context_inferred、unobserved；长时间缺失仍保留未知。CSV 的 `raw_activity` 可追溯原始规则判断。

时间为相对首个源帧的原始 PTS 秒数；区间 `[开始,结束)`，源帧号从 1 开始且首尾包含。最后一帧时长用相邻 PTS 中位数估计。边界前后 0.5 秒只是播放复核上下文，不是精度保证。

## 本次实际检查

| 视频 | 主活动段数 v4.6 → v4.7 | 单次动作候选 | 上下文推断时间比例 |
|---|---:|---|---:|
| 视频1 | 21 → 8 | {} | 9.7% |
| 视频2 | 24 → 12 | {'bending': 5} | 1.0% |
| 视频3 | 35 → 15 | {'split_squat': 6} | 2.1% |

第二个视频约 62.8 秒的俯身，通过清楚的躯干周期、局部稳定下肢观测和前后两次完整俯身支持自动恢复，候选从 4 次变为 5 次。最后一次俯身恢复基准不再追随起身后靠近镜头产生的尺度增大，结束点约 65.7 秒。第三个视频 6 次下肢屈伸仍单独保留起止，约 47.1–77.1 秒的迈步在主活动层保持一整段。活动段数减少包含显示层次调整，不能当作准确率提升的量化证据。

205 项回归与规则检查通过；三个视频 6289 个源帧全部覆盖，逐帧相对 PTS 与音频包检查通过。交付时用当前检测代码重放缓存，核对导出主标签、单次事件、阶段和推断状态一致。输入视频和旧版交付未覆盖。视频上下增加标注区，原始画面完整保留，输出 720×1532。

## 限制

这是自动姿态与时间上下文规则系统，没有新增经过标注训练的通用动作语义模型，也不声称任意动作都能识别。三个视频参与了开发与复查；未进行独立专家逐帧标注的准确率和边界误差评估。所有结果仍带待复核标记。场景变化、严重遮挡、镜头视差和二维姿态误差可能影响判断。规则见 `ACTION_RULES.md`，全部阈值见 `action_config.json`。历史结果见带 legacy 的说明与验证文件。
