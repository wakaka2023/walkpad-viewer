# v4.9.8-r13-gait-switch-fix

This release extends r13-turn-fix. See [walk/run switch rules](RULES_GAIT_SWITCH_FIX.md).

# v4.9.8-r13-turn-fix

完整包，保留上版镜头边缘补偿；新增同一轨迹内肩、髋、脚错时转身检测。详见 [RULES_TURN_FIX.md](RULES_TURN_FIX.md)。

# v4.9.8-r13-camera-fix

此为 r13 完整包，仅并入轨迹边缘镜头补偿修复。见 [RULES_CAMERA_FIX.md](RULES_CAMERA_FIX.md)，其余规则见 [RULES_R13.md](RULES_R13.md)。

# v4.9.8-r13

See [RULES_R13.md](RULES_R13.md) for the active changes.

# v4.9.8-r12

See [RULES_R12.md](RULES_R12.md) for active changes. Older notes are kept as reference.

# v4.9.8-r11

See [RULES_R11.md](RULES_R11.md) for active changes. Earlier sections are retained for reference.

# v4.9.8-r10

See [RULES_R10.md](RULES_R10.md) for the active rules and regression results. Older sections remain for reference.

# v4.9.8-r9

See [GAIT_R9.md](GAIT_R9.md) for the latest gait transition rules. The sections below are inherited.

# v4.9.8-r8

See [GAIT_R8.md](GAIT_R8.md) for the active gait rules and validation. The notes below document the inherited pipeline.

# v4.9.8-r7 多人动作标注（基于 r6）

r7 在身份串联和编号前新增墙面照片、海报等静态平面人像的复核。只有在检测轨迹局部稳定、四边形成纸面边界且与周围墙面配准后图案没有独立运动时才排除。不是把站着不动的人删掉，不需要为本视频手填区域。见 `FLAT_DEPICTION_R7.md` 和 `VALIDATION_R7.md`。

r6 增加同人短慢跑补判：用当前分析片段里已支持的走路与跑步建立个人参考，比较腿部展开、髋肩同步起伏和屈肘。只补原本走跑未定的画面，输出「跑步候选」；不把未知一律填成跑步。不使用视频名、ID 或某个时间点强制判别。详见 `PERSONAL_SHORT_RUN_R6.md`。

每个视频独立识别和编号。得到有效轨迹后按首次出现顺序分配 **ID 1、2、3…**；同帧首次出现按画面从左到右。短断轨或离画后在衣着证据足够时沿用原 ID，原始追踪编号不影响成片编号。

r5 在身份重连和编号前核验人体姿态。只有重复出现可信的人体结构，候选轨迹才有资格获得显示 ID。动作仍可能是“其他”，真人不因静止或动作未分类而被过滤。保留腿脚近景入口，普通误检的过滤不依赖特定视频或区域。

r4 支持顶部自带固定字幕条的腿脚近景自动进入步态分支，不使用字幕文字。保留 r3 的动作衔接策略。r3 仅补同轨迹、两侧有相符证据的短时“其他”；成片和逐帧表均区分推断。见 `ACTION_RULES.md` 末节。

## 快速开始

在解压后的版本目录运行（已有依赖时跳过安装）：

```powershell
python -m pip install -r requirements.txt
python recognize_actions.py --multi-person "video.mp4" --output "out" --device auto
```

也可传多个视频或一个目录。`--rotate 90` 只用于源视频方向元数据不正确的情况；`--no-preview` 只输出表格；在同一输出目录使用 `--resume` 可复用签名匹配的中间结果。

支持站立、走路、跑步、深蹲、上楼梯和下楼梯；证据不足显示候选或其他。同装、遮挡、近景脚踝不可见、动作转换和长时间离画都可能无法确认。详细参数与规则在 `RECOGNITION_R2.md` 和 `ACTION_RULES.md`，r6 验证见 `VALIDATION_R6.md`，r5 人体过滤验证见 `VALIDATION_R5.md`，历史记录见 `VALIDATION_R4.md`、`VALIDATION_R2.md`。

**玻璃区域不是全局默认参数。** 换视频正常先用上面的通用命令。仅固定机位镜像已知时设 `--ignore-static-roi x1,y1,x2,y2`；本次飞书视频的区域和复现命令见 `RECOGNITION_R2.md`。

| 输出 | 用途 |
|---|---|
| `review.mp4` | 带显示 ID、动作及深蹲阶段的视频。 |
| `activities.csv` / `boundary_events.csv` | 每个显示 ID 的动作区间、边界。区间为 `[开始, 结束)`，以原视频首帧为零点。 |
| `frame_labels.csv` | 逐帧标注，`annotation` 与画面标签一致。 |
| `action_events.csv` | 深蹲单次起点、最低点和结束位置。 |
| `track_id_map.csv` | 原始 ID、内部 ID、显示 ID 及过滤状态。没有显示 ID 的轨迹不上画面。 |
| `flat_depiction.csv` | 平面人像候选的配准、边界、纹理和删除记录。 |
| `person_evidence.csv` | 身份确认前的姿态证据审计：各类有效帧数、拒绝原因和删去的观测数。 |
| `reentry_links.csv` / `track_links.csv` / `ignored_tracks.csv` | 重入、短断轨、已知区域过滤的审计。 |

主要参数：`action_config.json`；YOLO 权重：`models/yolo26s-pose.pt`；ByteTrack 配置：`bytetrack_actions.yaml`。
