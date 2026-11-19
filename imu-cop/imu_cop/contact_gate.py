"""Independent, rotation-invariant causal foot load head for the 50 Hz stream."""
import math

import torch
from torch import nn

VERSION = 'foot-load-causal-50hz-v1'
RELATIVE_VERSION = 'foot-load-relative-50hz-v2'
RIGHT_POLICY = 'right_bounded_v1'
RATE = 50
FRAMES = 25
LAGS = (1, 3, 6)
INPUT_FRAMES = FRAMES + max(LAGS)
CHANNELS = 2 + 2*len(LAGS) + 1


def features(imu, mask):
    """[..., 6, T] in g and deg/s. A fixed rotation of either IMU cancels out."""
    a, g = imu[..., :3, :], imu[..., 3:, :]/100
    am, gm = torch.linalg.vector_norm(a, dim=-2), torch.linalg.vector_norm(g, dim=-2)
    out = [am.log1p(), gm.log1p()]
    for lag in LAGS:
        indices = (torch.arange(imu.shape[-1], device=imu.device)-lag).clamp_min(0)
        seen = mask*mask.index_select(-1, indices)
        for v in (a, g):
            diff = torch.linalg.vector_norm(v-v.index_select(-1, indices), dim=-2)
            out.append(diff.log1p()*seen)
    # Bounded relative direction between acceleration and angular velocity.
    out.append((a*g).sum(-2).abs()/(am*gm).clamp_min(.05))
    return torch.stack(out, -2)*mask.unsqueeze(-2)


class ContactGate(nn.Module):
    def __init__(self, relative=False):
        super().__init__()
        self.relative = relative
        self.in_channels = CHANNELS+1+(CHANNELS-1 if relative else 0)
        self.local = nn.Sequential(nn.Conv1d(self.in_channels, 24, 5), nn.GELU(),
                                   nn.Conv1d(24, 24, 5, dilation=2), nn.GELU(),
                                   nn.Conv1d(24, 24, 3, dilation=3), nn.GELU())
        self.encoder = nn.Sequential(nn.Linear(48,32), nn.GELU())
        self.classifier = nn.Sequential(nn.Linear(65,24), nn.GELU(), nn.Linear(24,1))

    def from_features(self, feat, mask):
        b = feat.shape[0]
        m = mask.unsqueeze(-2)
        vals = [feat*m]
        if self.relative:
            dynamic = feat[..., :-1, :]*m
            scale = dynamic.sum(-1,keepdim=True)/m.sum(-1,keepdim=True).clamp_min(1)
            vals.append((dynamic/(scale+.05)).clamp_max(6))
        seq = torch.cat([*vals, m], -2)
        hidden = self.local(seq.reshape(b*2,self.in_channels,FRAMES))
        enc = self.encoder(torch.cat([hidden[..., -1],hidden.mean(-1)], -1)).reshape(b,2,32)
        other = mask[..., -1:].flip(1)
        return self.classifier(torch.cat([enc,enc.flip(1)*other,other],-1)).squeeze(-1)

    def forward(self, imu, mask):
        imu, mask = imu[..., -INPUT_FRAMES:], mask[..., -INPUT_FRAMES:]
        return self.from_features(features(imu, mask)[..., -FRAMES:], mask[..., -FRAMES:])


def gate_from_checkpoint(checkpoint, device='cpu'):
    architecture=checkpoint.get('architecture')
    if architecture not in (VERSION,RELATIVE_VERSION) or (checkpoint.get('rate'),checkpoint.get('input_frames'))!=(RATE,INPUT_FRAMES):
        raise ValueError('Expected an independent 50 Hz causal foot load checkpoint')
    if not 0 < float(checkpoint['threshold']) < 1:
        raise ValueError('Invalid foot load threshold')
    if not 0 < float(checkpoint.get('new_logit_weight', 1.)) <= 1:
        raise ValueError('Invalid short load head weight')
    if checkpoint.get('policy') not in (None,RIGHT_POLICY):
        raise ValueError('Unknown foot load policy')
    if checkpoint.get('policy')==RIGHT_POLICY:
        cap=checkpoint.get('old_logit_cap')
        cut=float(checkpoint['right_logit_cut'])
        if (cap is not None and (not math.isfinite(cap) or cap<=0)) or not math.isfinite(cut):
            raise ValueError('Invalid right-foot calibration values')
        if abs(float(torch.tensor(cut).sigmoid())-float(checkpoint['threshold']))>1e-7:
            raise ValueError('Right-foot threshold does not match its calibrated logit')
    model = ContactGate(relative=architecture==RELATIVE_VERSION)
    model.load_state_dict(checkpoint['state'])
    return model.to(device).eval(), checkpoint


def fuse_right(base_logit, short_logit, checkpoint):
    """Use the short load head only on the right channel; leave the left exact."""
    right = base_logit[..., 1]
    cap = checkpoint.get('old_logit_cap')
    if cap is not None:
        right = right.clamp_max(cap)
    w = checkpoint['new_logit_weight']
    return torch.stack((base_logit[..., 0], (1-w)*right+w*short_logit[..., 1]), dim=-1)
