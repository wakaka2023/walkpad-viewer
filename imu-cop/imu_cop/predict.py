"""Load the matched coordinate and contact weights; predict on filtered 50 Hz windows."""
import hashlib
from io import BytesIO
from pathlib import Path

import torch
from torch import nn

from .action_model import from_checkpoint
from .contact_gate import RIGHT_POLICY, fuse_right, gate_from_checkpoint
from .settings import CONTEXT, WARMUP

COORDINATE_SHA256 = '3b32483388104eb1beba00b00501ce996c644e2f3f159dc914d579641a77d87e'
RIGHT_GATE_SHA256 = '03f606c9d6f522929bb2d7715d44af334b3c092d3fecaf506e2ac981d648dba8'


def _load_verified(path, sha256):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError(f'Unexpected checksum for {Path(path).name}')
    return torch.load(BytesIO(raw), map_location='cpu', weights_only=True)


def _load_pair(coordinate_path, right_gate_path):
    base = _load_verified(coordinate_path, COORDINATE_SHA256)
    spec = _load_verified(right_gate_path, RIGHT_GATE_SHA256)
    if spec['base_checkpoint_sha256'] != COORDINATE_SHA256 or spec.get('policy') != RIGHT_POLICY:
        raise ValueError('The right-foot gate does not match the coordinate model')
    return base, spec


class RightFootCoP(nn.Module):
    """CoP coordinates and pressure-load proxy at the last sample of each window."""

    def __init__(self, base, gate, calibration):
        super().__init__()
        self.base = base
        self.gate = gate
        self.calibration = dict(calibration)
        self.register_buffer('thresholds', torch.tensor([.5, calibration['threshold']]))

    @classmethod
    def load(cls, coordinate_path, right_gate_path, *, device='cpu'):
        """Load both verified assets from the imu-cop release."""
        base, spec = _load_pair(coordinate_path, right_gate_path)
        gate, _ = gate_from_checkpoint(spec, device)
        config = {k: spec[k] for k in ('threshold', 'new_logit_weight', 'old_logit_cap')}
        return cls(from_checkpoint(base, device), gate, config).to(device).eval()

    def forward(self, imu, mask):
        """Return raw CoP [B,2,2], load proxy [B,2], and display mask [B,2].

        imu is a prefiltered float32 [B,2,6,150] tensor in g and deg/s.
        mask is boolean [B,2,150], True for available samples. Left precedes right.
        Raw coordinates and load proxy are diagnostics wherever display is false.
        """
        if imu.ndim != 4 or imu.shape[1:] != (2, 6, CONTEXT):
            raise ValueError('IMU windows must have shape [B, 2, 6, 150]')
        if mask.shape != (imu.shape[0], 2, CONTEXT) or mask.dtype != torch.bool:
            raise ValueError('Availability must be a boolean [B, 2, 150] tensor')
        if imu.dtype != torch.float32:
            raise ValueError('IMU windows must be float32')
        # Do not let padded/missing samples influence the IMU features.
        imu = imu.masked_fill(~mask.unsqueeze(2), 0.)
        if not torch.isfinite(imu).all():
            raise ValueError('Available IMU samples must be finite')
        return self._predict_masked(imu, mask)

    def _predict_masked(self, imu, mask):
        available = mask.to(dtype=imu.dtype)
        cop, logit, _ = self.base(imu, available)
        probability = fuse_right(logit, self.gate(imu, available), self.calibration).sigmoid()
        ready = (mask.sum(dim=-1) >= WARMUP) & mask[..., -1]
        return cop, probability, ready & (probability >= self.thresholds)

    @classmethod
    def native_stream(cls, coordinate_path, right_gate_path, rates, *, device='cpu'):
        """Process native g/deg/s samples with the deployed causal filter and 50 Hz clock.

        Install the dependencies in ``requirements.txt`` before calling this.
        """
        from .native_runtime import PairedStream

        base, spec = _load_pair(coordinate_path, right_gate_path)
        return PairedStream(base, rates, device=device, contact_gate=gate_from_checkpoint(spec, device))
