# imu-cop

从左右脚背的六轴 IMU 估计各脚的压力中心（CoP）。原通用坐标模型用最近 150 帧；实验分支用最近 31 帧估计右脚载荷，并与原模型载荷分数融合。右脚显示门控更新；左脚门控和两脚原始坐标仍来自通用模型。每 20 ms 输出一次，推理不使用未来帧。可用 ONNX Runtime 或 PyTorch 独立运行，Walkpad 查看器的网页不会自动调用它。

从 [imu-cop 预发布](https://github.com/wakaka2023/walkpad-viewer/releases/tag/imu-cop)下载 **`imu-cop.onnx`** 即可使用 ONNX 推理。只有运行原 PyTorch 实现或重新导出时才需要两份 `.pt` 文件。

| GitHub Release 文件 | 用途 |
|---|---|
| `imu-cop.onnx` | 单文件模型，包含双脚坐标网络和右脚承重门控（ONNX opset 17） |
| `imu-cop-universal-50hz-seed42.pt` | 双脚原坐标与载荷模型 |
| `imu-cop-right-gate-50hz-seed42.pt` | 右脚短窗口权重与校准参数 |
| `SHA256SUMS.txt` | 两份 PyTorch 权重的校验值（加载时也校验） |

训练标签来源为 EMG、Grouvel、ImbalanSense、UnderPressure 和 WHIP 的公开研究数据。此目录与 Release 不包含原始数据、个人 IMU 记录或设备标识。使用或再分发应遵守相应来源的授权。

## ONNX 推理

安装 `python -m pip install onnxruntime numpy`，然后在 `imu-cop.onnx` 所在目录运行：

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession('imu-cop.onnx', providers=['CPUExecutionProvider'])
imu = np.zeros((1, 2, 6, 150), dtype=np.float32)  # 示例形状，替换成实际传感器窗口
available = np.ones((1, 2, 150), dtype=bool)
cop, load_score, show = session.run(None, {'imu': imu, 'available': available})
visible_cop = np.where(show[..., None], cop, np.nan)
```

`imu` 为 `float32[B,2,6,150]`，`available` 为 `bool[B,2,150]`，批量大小 `B` 可变。输入须先做 8 Hz 因果低通并用过去样本对齐至 50 Hz；可用帧的数值必须有限。输出依次是 `cop` (`float32[B,2,2]`)、`load_score` (`float32[B,2]`) 和 `show` (`bool[B,2]`)。用 `show` 隐藏摆动、缺帧和未热身的点。实时使用时，每 20 ms 更新一次窗口；模型接收窗口，不负责读取原始文件或接收 BLE 数据。

如需从两份 PyTorch 权重复现导出，在本目录安装 `python -m pip install -r requirements.txt onnx` 后运行：

```bash
python export_onnx.py imu-cop-universal-50hz-seed42.pt imu-cop-right-gate-50hz-seed42.pt
```

## PyTorch 推理

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
