"""Export the two matching checkpoints as a single self-contained ONNX model."""
import argparse
from pathlib import Path

import onnx
import torch
from torch import nn

from imu_cop import CONTEXT, RATE, RightFootCoP
from imu_cop.predict import COORDINATE_SHA256, RIGHT_GATE_SHA256
from imu_cop.settings import WARMUP

INPUTS = ('imu', 'available')
OUTPUTS = ('cop', 'load_score', 'show')


class WindowGraph(nn.Module):
    """Tensor-only inference. The caller supplies finite values at available frames."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, imu, available):
        imu = imu.masked_fill(~available.unsqueeze(2), 0.)
        return self.model._predict_masked(imu, available)


def export_model(coordinate_path, gate_path, output):
    model = RightFootCoP.load(coordinate_path, gate_path)
    inputs = (torch.zeros(2, 2, 6, CONTEXT), torch.ones(2, 2, CONTEXT, dtype=torch.bool))
    with torch.no_grad():
        # Fold the learned rotation matrix, which does not depend on the IMU input.
        frozen = torch.jit.freeze(torch.jit.trace(WindowGraph(model).eval(), inputs))
        torch.onnx.export(
            frozen, inputs, str(output), dynamo=False, opset_version=17, external_data=False,
            input_names=list(INPUTS), output_names=list(OUTPUTS),
            dynamic_axes={name: {0: 'batch'} for name in (*INPUTS, *OUTPUTS)},
        )
    graph = onnx.load(output, load_external_data=False)
    graph.doc_string = '50 Hz foot IMU to per-foot CoP with the experimental right-foot load gate.'
    onnx.helper.set_model_props(graph, {
        'sample_rate_hz': str(RATE),
        'window_frames': str(CONTEXT),
        'warmup_frames': str(WARMUP),
        'foot_order': 'left,right',
        'axis_order': 'ax,ay,az,gx,gy,gz',
        'imu_units': 'g,deg/s',
        'preprocessing': 'causal 8 Hz low-pass and past-only alignment to 50 Hz; performed by caller',
        'cop_order': 'mediolateral,anteroposterior; relative to foot length',
        'show': 'false for swing or unavailable frames; suppress visualization on those frames',
        'coordinate_weights_sha256': COORDINATE_SHA256,
        'right_gate_weights_sha256': RIGHT_GATE_SHA256,
    })
    onnx.checker.check_model(graph, full_check=True)
    onnx.save(graph, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('coordinate_path', type=Path)
    parser.add_argument('gate_path', type=Path)
    parser.add_argument('-o', '--output', type=Path, default=Path('imu-cop.onnx'))
    args = parser.parse_args()
    export_model(args.coordinate_path, args.gate_path, args.output)


if __name__ == '__main__':
    main()
