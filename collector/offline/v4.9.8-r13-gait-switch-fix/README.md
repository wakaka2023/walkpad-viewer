# r13 离线录像步态标注

对已录制的单目视频进行多人动作标注。程序按人物轨迹输出站立、走路、跑步、深蹲和上下楼的区间，并为深蹲记录起点、最低点和结束点。不需要鞋垫或 IMU 数据，不接入采集工作台的实时识别。

r13 主要处理三类容易误判的情况：原地转身、连续步态中的短缺口，以及近镜头裁切后的走跑切换。缺少足够证据时保留候选或「其他」，不会单凭固定步频划分走路和跑步。细节见 [r13 判别规则](RULES_R13.md) 和 [动作参数](action_config.json)。

## 运行

先按[权重说明](models/README.md)准备并校验 `models/yolo26s-pose.pt`，再在本目录运行。输出目录建议放在仓库外：

```powershell
python -m pip install -r requirements.txt
python recognize_actions.py --multi-person "D:/videos/example.mp4" --output "D:/video-labels/r13" --device auto
```

不需要标注成片时，可加 `--no-preview` 只生成表格。

## 输出

| 文件 | 用途 |
| --- | --- |
| `review.mp4` | 查看人物 ID、动作标签和深蹲阶段。 |
| `activities.csv`、`boundary_events.csv` | 人物动作区间和边界。时间以源视频首帧为零，区间左闭右开。 |
| `frame_labels.csv` | 逐帧审查动作和候选状态。 |
| `action_events.csv` | 查看深蹲单次的起点、最低点和结束点。 |

## 使用限制

遮挡、追踪中断或脚部出画时，部分动作只能标作候选；转身是根据二维视频推断，不是三维角度测量。随包的回归结果不代表陌生机位的独立准确率。

原交付说明见 [README.original.md](README.original.md)，文件校验见 [MANIFEST.json](MANIFEST.json)。
