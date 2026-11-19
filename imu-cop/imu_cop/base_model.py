"""Shared causal foot encoder, optional bilateral fusion, physical sensor adapter."""
import torch
import torch.nn.functional as F
from torch import nn

from .settings import RATE


class Block(nn.Module):
    def __init__(self,width,dilation):
        super().__init__();self.pad=4*dilation
        self.a=nn.Conv1d(width,width,5,dilation=dilation);self.b=nn.Conv1d(width,width,5,dilation=dilation)
        self.drop=nn.Dropout(.1)
    def forward(self,x):
        y=self.drop(F.gelu(self.a(F.pad(x,(self.pad,0)))))
        return x+self.drop(F.gelu(self.b(F.pad(y,(self.pad,0)))))


def rotation(vector):
    x,y,z=vector.unbind(-1);zero=torch.zeros_like(x)
    skew=torch.stack([zero,-z,y,z,zero,-x,-y,x,zero],-1).reshape(*vector.shape[:-1],3,3)
    return torch.matrix_exp(skew)


class PairedCoP(nn.Module):
    def __init__(self,mean,std,positions,bilateral=True,width=48):
        super().__init__();self.bilateral=bilateral;self.width=width
        self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32).reshape(1,1,6,1))
        self.register_buffer('std',torch.as_tensor(std,dtype=torch.float32).reshape(1,1,6,1))
        self.register_buffer('positions',torch.as_tensor(positions,dtype=torch.float32))
        self.sensor_rotation=nn.Parameter(torch.zeros(2,3));self.sensor_bias=nn.Parameter(torch.zeros(2,6))
        self.sensor_log_gain=nn.Parameter(torch.zeros(2,2))
        self.stem=nn.Conv1d(8,width,1);self.blocks=nn.Sequential(*(Block(width,d) for d in (1,2,4,8)))
        self.encode=nn.Sequential(nn.Linear(width*3,128),nn.GELU(),nn.Dropout(.1))
        self.gate=nn.Linear(256,1);nn.init.constant_(self.gate.bias,-1.)
        self.head=nn.Sequential(nn.Linear(385,128),nn.GELU(),nn.Dropout(.1))
        self.personal=nn.Sequential(nn.Linear(128,16),nn.GELU(),nn.Linear(16,128))
        nn.init.zeros_(self.personal[-1].weight);nn.init.zeros_(self.personal[-1].bias)
        self.pressure=nn.Linear(128,16);self.contact=nn.Linear(128,1)
        self.output_bias=nn.Parameter(torch.zeros(2,2))

    def freeze_for_public(self):
        for name,p in self.named_parameters():
            p.requires_grad=not name.startswith(('sensor_','personal.','output_bias'))

    def personalize(self,mode='adapter'):
        for name,p in self.named_parameters():
            p.requires_grad=mode=='full' or name.startswith(('sensor_','personal.','output_bias','pressure.','contact.'))

    def forward(self,x,mask=None):
        # x: B,2,6,T in g/deg/s. No labels or subject/domain IDs accepted.
        b,_,_,t=x.shape
        if mask is None:mask=torch.ones(b,2,t,device=x.device,dtype=x.dtype)
        r=rotation(self.sensor_rotation)
        acc=torch.einsum('sij,bsjt->bsit',r,x[:,:,:3]);gyro=torch.einsum('sij,bsjt->bsit',r,x[:,:,3:])
        gain=self.sensor_log_gain.exp().repeat_interleave(3,dim=1)[None,:,:,None]
        x=(torch.cat([acc,gyro],2)*gain+self.sensor_bias[None,:,:,None]-self.mean)/self.std.clamp_min(1e-5)
        side=torch.tensor([-1.,1.],device=x.device,dtype=x.dtype)[None,:,None,None].expand(b,2,1,t)
        inp=torch.cat([x*mask[:,:,None],mask[:,:,None],side*mask[:,:,None]],2).reshape(b*2,8,t)
        h=self.blocks(F.gelu(self.stem(inp))).reshape(b,2,self.width,t)
        def average(n):
            m=mask[:,:,-n:];return (h[:,:,:,-n:]*m[:,:,None]).sum(-1)/m.sum(-1).clamp_min(1)[:,:,None]
        own=self.encode(torch.cat([h[:,:,:,-1],average(RATE),average(t)],-1))
        other=own.flip(1);available=mask[:,:,-1:].flip(1)
        if not self.bilateral:available=available*0
        if self.training:
            available=available*(torch.rand(b,2,1,device=x.device)>.15)
        gate=torch.sigmoid(self.gate(torch.cat([own,other],-1)))*available
        h=self.head(torch.cat([own,other*gate,own*other*gate,available],-1))
        h=h+self.personal(h)
        logits=self.pressure(h);distribution=logits.softmax(-1)
        cop=torch.einsum('bsp,spc->bsc',distribution,self.positions)+.05*torch.tanh(self.output_bias)[None]
        return cop,self.contact(h).squeeze(-1),logits


def augment(x):
    """Virtual mounting rotations, gain/bias/noise; one transform per window/foot."""
    b=x.shape[0];r=rotation(torch.randn(b,2,3,device=x.device)*(.20))
    a=torch.einsum('bsij,bsjt->bsit',r,x[:,:,:3]);g=torch.einsum('bsij,bsjt->bsit',r,x[:,:,3:])
    y=torch.cat([a,g],2)*(1+.04*torch.randn(b,2,1,1,device=x.device))
    scale=torch.tensor([.02]*3+[.6]*3,device=x.device)[None,None,:,None]
    return y+torch.randn(b,2,6,1,device=x.device)*scale+torch.randn_like(y)*scale*.2
