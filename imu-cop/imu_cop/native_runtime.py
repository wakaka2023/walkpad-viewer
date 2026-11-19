"""Asynchronous paired native IMU -> shared causal window; labels never accepted."""
import math
from collections import deque

import numpy as np
import torch
from scipy.signal import butter, sosfilt, sosfilt_zi

from .action_model import from_checkpoint
from .actions import ACTION_NAMES
from .contact_gate import RIGHT_POLICY, fuse_right
from .settings import CONTEXT, RATE, WARMUP, validate_checkpoint


def make_model(checkpoint, device):
    validate_checkpoint(checkpoint)
    return from_checkpoint(checkpoint, device)


class PairedStream:
    def __init__(self,checkpoint,rates,device='cpu',*,contact_gate=None):
        self.checkpoint=checkpoint;self.model=make_model(checkpoint,device).eval();self.device=device
        self.contact_gate=contact_gate
        self.contact_threshold=.5 if contact_gate is None else float(contact_gate[1]['threshold'])
        self.right_only=contact_gate is not None and contact_gate[1].get('policy')==RIGHT_POLICY
        self.rates=rates;self.feet={};self.rows=deque(maxlen=CONTEXT);self.masks=deque(maxlen=CONTEXT);self.origin=None;self.tick=0;self.last_event=None
        self.resets={'left':0,'right':0}
        for s,rate in rates.items():
            if s not in ('left','right') or not math.isfinite(rate) or not 16<rate<=1000:raise ValueError('Invalid foot/rate')

    def clear(self,side):
        for x,m in zip(self.rows,self.masks):x[side]=0;m[side]=0

    def _frame(self,t):
        x=np.zeros((2,6));mask=np.zeros(2)
        for side,name in enumerate(('left','right')):
            foot=self.feet.get(name)
            # push() treats timestamps within 10 ns of the grid as the same event.
            if foot is not None and -1e-8<=t-foot['t']<=.2+1e-8:
                x[side]=foot['filtered'];mask[side]=1
            elif foot is not None:self.clear(side)
        self.rows.append(x);self.masks.append(mask)
        xx=np.zeros((CONTEXT,2,6),dtype=np.float32);mm=np.zeros((CONTEXT,2),dtype=np.float32)
        xx[-len(self.rows):]=np.stack(self.rows);mm[-len(self.rows):]=np.stack(self.masks)
        available=mm.sum(0);out={}
        if available.max()<WARMUP:
            return {name:dict(t=t,status='warming',model_cop=None,cop=None,contact_probability=None,warmup_frames=int(available[s])) for s,name in enumerate(('left','right'))}
        route = {}
        with torch.inference_mode():
            inputs = torch.as_tensor(xx.transpose(1,2,0)[None],device=self.device)
            masks = torch.as_tensor(mm.T[None],device=self.device)
            if self.checkpoint.get('architecture') == 'action-causal-tcn-50hz-v1':
                components = self.model.components(inputs, masks)
                y,c,_ = self.model.decode(components, mode='auto')
                if self.model.specialists:
                    p = components['probabilities'][0].cpu().numpy()
                    route = dict(router_top_class=ACTION_NAMES[int(p.argmax())],
                                 router_confidence=float(p.max()),
                                 generic_branch_weight=float(self.model.weights(components)[0,0]))
            else:
                y,c,_=self.model(inputs,masks)
            if self.contact_gate is not None:
                short_model,short_checkpoint=self.contact_gate
                w=short_checkpoint.get('new_logit_weight',1.)
                short_logit=short_model(inputs,masks)
                if self.right_only:
                    c=fuse_right(c,short_logit,short_checkpoint)
                else:
                    c=(1-w)*c+w*short_logit
            y=y[0].cpu().numpy();c=c[0].sigmoid().cpu().numpy()
        for s,name in enumerate(('left','right')):
            ready=mask[s] and available[s]>=WARMUP
            threshold=.5 if self.right_only and name=='left' else self.contact_threshold
            on_load=c[s]>=threshold
            out[name]=dict(t=t,status=('contact' if on_load else 'swing') if ready else ('warming' if mask[s] else 'stale'),
                           model_cop=y[s].tolist() if ready else None,cop=y[s].tolist() if ready and on_load else None,
                           contact_probability=float(c[s]) if ready else None,warmup_frames=int(available[s]),
                           full_context=bool(available[s]>=CONTEXT),**(route if ready else {}))
        return out

    def push(self,t,batches,boundaries=None):
        t=float(t);boundaries=boundaries or {}
        if not math.isfinite(t) or (self.last_event is not None and t<=self.last_event):raise ValueError('Joint receive event times must strictly increase')
        validated={}
        for name,values in batches.items():
            if name not in self.rates:raise ValueError('Unconfigured foot')
            values=np.asarray(values,dtype=float)
            if values.ndim!=2 or values.shape[1]!=6 or not len(values) or not np.isfinite(values).all():raise ValueError('Ordered finite Nx6 native samples required')
            validated[name]=values
        if not validated:raise ValueError('Empty joint event')
        if self.origin is None:self.origin=t
        output=[]
        while self.origin+self.tick/RATE<t-1e-8:
            output.append(self._frame(self.origin+self.tick/RATE));self.tick+=1
        for name,values in validated.items():
            previous=self.feet.get(name);boundary=boundaries.get(name)
            reset=previous is None or t-previous['t']>max(.2,3/self.rates[name]) or previous.get('boundary')!=boundary
            if reset:
                if previous is not None:self.resets[name]+=1
                self.clear(0 if name=='left' else 1);sos=butter(4,8,fs=self.rates[name],output='sos')
                previous=dict(sos=sos,zi=sosfilt_zi(sos)[:,:,None]*values[0][None,None,:])
            filtered,zi=sosfilt(previous['sos'],values,axis=0,zi=previous['zi'])
            self.feet[name]=dict(sos=previous['sos'],zi=zi,filtered=filtered[-1],t=t,boundary=boundary)
        if abs(self.origin+self.tick/RATE-t)<=1e-8:
            output.append(self._frame(self.origin+self.tick/RATE));self.tick+=1
        self.last_event=t
        return output
