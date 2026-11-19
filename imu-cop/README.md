# imu-cop

从左右脚背的六轴 IMU 估计各脚的压力中心（CoP）。原通用坐标模型用最近 150 帧；实验分支用最近 31 帧估计右脚载荷，并与原模型载荷分数融合。右脚显示门控更新；左脚门控和两脚原始坐标仍来自通用模型。每 20 ms 输出一次，推理不使用未来帧。这是独立的 Python 推理模块，Walkpad 查看器的网页不会自动调用它。

从 [imu-cop 预发布](https://github.com/wakaka2023/walkpad-viewer/releases/tag/imu-cop)下载两份配套权重，放入本目录。也可以下载该预发布的完整源码快照来运行示例。

| GitHub Release 文件 | 用途 |
|---|---|
| `imu-cop-universal-50hz-seed42.pt` | 双脚原坐标与载荷模型 |
| `imu-cop-right-gate-50hz-seed42.pt` | 右脚短窗口权重与校准参数 |
| `SHA256SUMS.txt` | 权重文件校验值（加载时也校验） |

文件为 PyTorch 研究权重。训练标签来源为 EMG、Grouvel、ImbalanSense、UnderPressure 和 WHIP 的公开研究数据。此目录与 Release 不包含原始数据、个人 IMU 记录或设备标识。使用或再分发应遵守相应来源的授权。

## 最小推理示例

在仓库根目录运行：

```bash
cd imu-cop
python -m pip install -r requirements.txt
```

以下示例接收**已经经过因果低通和 50 Hz 对齐**的窗口：

```python
import torch
from imu_cop import RightFootCoP

model = RightFootCoP.load(
    'imu-cop-universal-50hz-seed42.pt',
    'imu-cop-right-gate-50hz-seed42.pt',
)
imu = torch.zeros(1, 2, 6, 150, dtype=torch.float32)  # 示例形状；换成实际传感器窗口
ready = torch.ones(1, 2, 150, dtype=torch.bool)
with torch.inference_mode():
    cop, load_score, show = model(imu, ready)
```

轴顺序为 `[ax, ay, az, gx, gy, gz]`，加速度单位 `g`，角速度 `deg/s`；两脚按左、右排序，最新帧在最后。`cop` 是按脚长归一化的 `[内外侧, 前后]` 原始坐标，乘 100 转为脚长百分比。`load_score` 是压力载荷代理；`show` 为假的帧应在画面置空，避免把摆动阶段的诊断坐标连成轨迹。每脚至少需要 50 个可用历史帧且当前可用。

对原生传感器事件，用 `RightFootCoP.native_stream(coordinate_path, right_gate_path, rates={'left': 51.2, 'right': 51.2})` 创建处理器；`rates` 应使用各设备实际采样率。传入 `stream.push(t, {'left': [left_sample], 'right': [right_sample]})`。`t` 是严格递增的公共事件时间（秒）；同一时刻的样本合为一次调用，不同时刻可只传一只脚。输入需先完成设备间对时。处理器使用 8 Hz 因果低通和 50 Hz 的过去样本对齐；输出每脚的 `cop` 为坐标或 `None`。

## 验证与限制

使用固定受试者划分，阈值只由验证集选择。下表是公开压力测试集的**右脚**分来源等权结果，跑步覆盖 8 人、24 个连续片段：

| 版本 | 跑步低载仍出点（越低越好） | 跑步高载保留 | 行走高载保留 |
|---|---:|---:|---:|
| 原通用模型 | 43.35% | 98.40% | 98.18% |
| 上一版双脚融合 | 24.84% | 94.53% | 95.52% |
| 本实验 | 29.19% | 95.36% | 95.98% |

本实验比上一版多保留部分承重帧，但低载误显示也有所回升，且未达到相对原通用模型的预设高载保留标准。换用实验门控会改变断点；缺少同步压力的脚背 IMU 记录不能验证断点是否与真实离地吻合，也不能验证 CoP 坐标精度。正式使用前应以目标设备同步压力作为对照。
