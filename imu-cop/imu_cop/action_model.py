"""A pretrained causal bilateral encoder, generic decoder, five residual experts.

Auto routing consumes IMU-derived features only. File labels can only be supplied
to the explicitly requested oracle mode, which is an offline diagnostic.
"""
import torch
import torch.nn.functional as F
from torch import nn

from .actions import ACTION_NAMES, LABEL_VERSION
from .base_model import PairedCoP, rotation
from .settings import RATE, validate_checkpoint


class ActionCoP(PairedCoP):
    def __init__(self, mean, std, positions, specialists=True):
        super().__init__(mean, std, positions, bilateral=True)
        self.specialists = specialists
        if specialists:
            self.experts = nn.ModuleList([
                nn.Sequential(nn.Linear(128, 48), nn.GELU(), nn.Linear(48, 17))
                for _ in ACTION_NAMES[1:]])
            for expert in self.experts:
                nn.init.zeros_(expert[-1].weight)
                nn.init.zeros_(expert[-1].bias)
            self.router = nn.Sequential(nn.Linear(256, 64), nn.GELU(), nn.Linear(64, len(ACTION_NAMES)))
        self.threshold = .5

    def components(self, x, mask=None):
        b, _, _, t = x.shape
        if mask is None:
            mask = torch.ones(b, 2, t, device=x.device, dtype=x.dtype)
        r = rotation(self.sensor_rotation)
        acc = torch.einsum('sij,bsjt->bsit', r, x[:, :, :3])
        gyro = torch.einsum('sij,bsjt->bsit', r, x[:, :, 3:])
        gain = self.sensor_log_gain.exp().repeat_interleave(3, dim=1)[None, :, :, None]
        x = (torch.cat([acc, gyro], 2)*gain+self.sensor_bias[None, :, :, None]-self.mean)/self.std.clamp_min(1e-5)
        side = torch.tensor([-1., 1.], device=x.device, dtype=x.dtype)[None, :, None, None].expand(b, 2, 1, t)
        inp = torch.cat([x*mask[:, :, None], mask[:, :, None], side*mask[:, :, None]], 2).reshape(b*2, 8, t)
        h = self.blocks(F.gelu(self.stem(inp))).reshape(b, 2, self.width, t)
        def average(n):
            m = mask[:, :, -n:]
            return (h[:, :, :, -n:]*m[:, :, None]).sum(-1)/m.sum(-1).clamp_min(1)[:, :, None]
        own = self.encode(torch.cat([h[:, :, :, -1], average(RATE), average(t)], -1))
        other = own.flip(1)
        available = mask[:, :, -1:].flip(1)
        if self.training:
            available = available*(torch.rand(b, 2, 1, device=x.device) > .15)
        gate = torch.sigmoid(self.gate(torch.cat([own, other], -1)))*available
        h = self.head(torch.cat([own, other*gate, own*other*gate, available], -1))
        h = h+self.personal(h)
        base_l = self.pressure(h)
        base_c = self.contact(h).squeeze(-1)
        if not self.specialists:
            p = torch.zeros(b, 6, device=x.device, dtype=x.dtype)
            p[:, 0] = 1
            return dict(logits=base_l[:, None], contact=base_c[:, None], probabilities=p)
        delta = torch.stack([e(h) for e in self.experts], 1)
        logits = torch.cat([base_l[:, None], base_l[:, None]+delta[..., :16]], 1)
        contact = torch.cat([base_c[:, None], base_c[:, None]+delta[..., 16]], 1)
        probabilities = self.router(h.reshape(b, 256)).softmax(-1)
        return dict(logits=logits, contact=contact, probabilities=probabilities)

    def weights(self, components, mode='auto', labels=None, threshold=None, detach_router=False):
        p = components['probabilities']
        if detach_router:
            p = p.detach()
        if mode == 'base' or not self.specialists:
            w = torch.zeros_like(p)
            w[:, 0] = 1
            return w
        if mode == 'oracle':
            if labels is None:
                raise ValueError('Oracle mode requires explicit file-level action labels')
            labels = torch.as_tensor(labels, device=p.device, dtype=torch.long)
            if labels.shape != (len(p),) or torch.any((labels < -1) | (labels >= 6)):
                raise ValueError('Action IDs must be -1 or 0..5, one per window')
            return F.one_hot(labels.clamp_min(0), 6).to(p.dtype)
        if mode != 'auto':
            raise ValueError(mode)
        threshold = self.threshold if threshold is None else float(threshold)
        if not 0 <= threshold <= 1:
            raise ValueError('threshold must lie in [0, 1]')
        confidence = p.max(-1).values
        strength = ((confidence-threshold)/max(1-threshold, 1e-6)).clamp(0, 1)
        expert_weights = p[:, 1:]*strength[:, None]
        return torch.cat([1-expert_weights.sum(-1, keepdim=True), expert_weights], -1)

    def decode(self, components, mode='auto', labels=None, threshold=None, detach_router=False):
        weights = self.weights(components, mode, labels, threshold, detach_router)
        if not self.specialists:
            logits = components['logits'][:, 0]
            contact = components['contact'][:, 0]
            distribution = logits.softmax(-1)
        elif mode == 'base':
            logits = components['logits'][:, 0]
            contact = components['contact'][:, 0]
            distribution = logits.softmax(-1)
        else:
            distribution = (components['logits'].softmax(-1)*weights[:, :, None, None]).sum(1)
            logits = distribution.clamp_min(1e-8).log()
            probability = (components['contact'].sigmoid()*weights[:, :, None]).sum(1)
            contact = torch.logit(probability.clamp(1e-6, 1-1e-6))
        cop = torch.einsum('bsp,spc->bsc', distribution, self.positions)+.05*torch.tanh(self.output_bias)[None]
        return cop, contact, logits

    def forward(self, x, mask=None, *, mode='auto', labels=None, threshold=None):
        # No domain, subject, pressure, CoP, or file name is accepted for auto mode.
        if mode != 'oracle' and labels is not None:
            raise ValueError('Labels are only accepted in oracle mode')
        return self.decode(self.components(x, mask), mode, labels, threshold)


def from_checkpoint(checkpoint, device='cpu'):
    validate_checkpoint(checkpoint)
    if checkpoint.get('label_version') != LABEL_VERSION:
        raise ValueError('Incompatible action label version')
    m = ActionCoP(checkpoint['stats']['mean'], checkpoint['stats']['std'],
                  checkpoint['positions'], checkpoint['specialists']).to(device)
    m.load_state_dict(checkpoint['state'])
    m.threshold = checkpoint['threshold']
    return m
