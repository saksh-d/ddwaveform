"""Fresh DD/CFO FP32 real pairs: default MPS or explicitly selected CPU.

Pairs (real, imaginary) implement complex algebra; saved complex arrays are I/O.
This avoids requiring native complex kernels on Apple's MPS backend.
"""
import math
import os
import torch
import numpy as np
from .core import seed

DEVICE = torch.device('mps')
DTYPE = torch.float32


def select_backend(backend):
    global DEVICE
    if backend not in ('mps','cpu'):
        raise ValueError('BACKEND must be mps or cpu; no automatic fallback or CUDA path')
    DEVICE = torch.device(backend)


def synchronize():
    if DEVICE.type == 'mps':
        torch.mps.synchronize()


def require_backend(environment):
    if torch.device(environment['device']).type != DEVICE.type:
        raise RuntimeError('Backend changed after workflow creation; restart and Run All')


def environment():
    if DEVICE.type == 'mps' and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
        raise RuntimeError('Disable PYTORCH_ENABLE_MPS_FALLBACK; fresh runs require MPS without CPU fallback')
    if DEVICE.type == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable: no CPU fallback. Select BACKEND=cpu explicitly or use an MPS-capable environment.')
    if DEVICE.type == 'cpu':
        torch.set_num_threads(1)
    probe = torch.ones(1, device=DEVICE, dtype=DTYPE)
    result = dict(device=str(probe.device), real_dtype=str(probe.dtype),
                  complex_representation='two real FP32 tensors', torch=torch.__version__, fallback=False)
    if DEVICE.type == 'cpu':
        result.update(threads=torch.get_num_threads(), random_engine='CPU torch.Generator; namespace seeds; distinct from MPS streams')
    return result


def tensor(x):
    return torch.as_tensor(x, device=DEVICE, dtype=DTYPE)


def pair(x):
    a = np.asarray(x)
    return tensor(a.real.copy()), tensor(a.imag.copy())


def array(a):
    return a[0].detach().cpu().numpy() + 1j*a[1].detach().cpu().numpy()


def draw(shape, namespace, normal=False, bits=False):
    if DEVICE.type == 'cpu':
        generator = torch.Generator(device='cpu').manual_seed(seed(*namespace) % (2**32))
        if bits:
            return torch.randint(0, 2, shape, device=DEVICE, dtype=torch.int32, generator=generator)
        return (torch.randn if normal else torch.rand)(shape, device=DEVICE, dtype=DTYPE, generator=generator)
    # Preserve the original MPS random stream exactly; CPU has a distinct generator.
    torch.mps.manual_seed(seed(*namespace) % (2**32))
    if bits:
        return torch.randint(0, 2, shape, device=DEVICE, dtype=torch.int32)
    return (torch.randn if normal else torch.rand)(shape, device=DEVICE, dtype=DTYPE)


def mm(a, b):
    return a[0]@b[0]-a[1]@b[1], a[0]@b[1]+a[1]@b[0]


def adj(a):
    return a[0].transpose(-1,-2), -a[1].transpose(-1,-2)


def transpose(a):
    return a[0].transpose(-1,-2), a[1].transpose(-1,-2)


def power(a):
    return a[0].square()+a[1].square()


def eye(n):
    return torch.eye(n, device=DEVICE, dtype=DTYPE), torch.zeros(n,n,device=DEVICE,dtype=DTYPE)


def dft(n):
    k = torch.arange(n, device=DEVICE, dtype=torch.int32)
    # Reduce integer roots before FP32 trig; unreduced O(N) angles lose phase.
    angle = -2*math.pi*((k[:,None]*k[None,:]) % n).to(DTYPE)/n
    return angle.cos()/math.sqrt(n), angle.sin()/math.sqrt(n)


def phase(angle):
    return angle.cos(), angle.sin()


def multiply(a,b):
    return a[0]*b[0]-a[1]*b[1], a[0]*b[1]+a[1]*b[0]


def atoms(tau, nu, n):
    k = torch.arange(n,device=DEVICE,dtype=DTYPE)
    f = dft(n)
    delay = phase(-2*math.pi*tau[...,None]*k/n)
    pi = mm(adj(f), multiply((delay[0][..., :,None],delay[1][..., :,None]), f))
    doppler = phase(-2*math.pi*nu[...,None]*k/n)
    return multiply((doppler[0][..., :,None],doppler[1][..., :,None]), pi)


def sample(count, model, namespace, atomic=False):
    n,p = model['n'],1 if atomic else model['paths']
    lo,hi = model['delay_range']; tau=lo+(hi-lo)*draw((count,p),(*namespace,'delay'))
    lo,hi = model['doppler_range']; nu=lo+(hi-lo)*draw((count,p),(*namespace,'doppler'))
    op=atoms(tau,nu,n)
    if atomic:
        return (op[0][:,0],op[1][:,0]), dict(tau=tau,nu=nu)
    gain=(draw((count,p),(*namespace,'gain-real'),normal=True)/math.sqrt(2*p),
          draw((count,p),(*namespace,'gain-imag'),normal=True)/math.sqrt(2*p))
    raw=multiply((gain[0][...,None,None],gain[1][...,None,None]),op)
    raw=raw[0].sum(1),raw[1].sum(1)
    scale=(power(raw).sum((-2,-1))/n).sqrt()
    if not torch.isfinite(scale).all() or (scale<1e-7).any():
        raise RuntimeError('Degenerate channel draw; not silently resampled')
    return (raw[0]/scale[:,None,None],raw[1]/scale[:,None,None]), dict(tau=tau,nu=nu,gains=gain,scale=scale,raw=raw)


def classical(n, grid):
    # Positive synthesis DFT, with OTFS stream order matching the archived model.
    f=adj(dft(n)); small=adj(dft(grid[1])); ident=eye(grid[0])[0]
    otfs=torch.kron(small[0].contiguous(),ident),torch.kron(small[1].contiguous(),ident)
    k=torch.arange(n,device=DEVICE,dtype=DTYPE)
    l1=phase(2*math.pi/(2*n)*k.square())
    l2=phase(2*math.pi*math.sqrt(2)/(100*n)*k.square())
    afdm=multiply(multiply((l1[0][:,None],l1[1][:,None]),f),(l2[0][None,:],l2[1][None,:]))
    return dict(OTFS=otfs,AFDM=afdm,OFDM=f)


def effective(w,h,r=None):
    return mm(adj(w) if r is None else r,mm(h,w))


def mask(q,k):
    # Deterministic stable native-index tie order; discovery only, not realization top-K.
    order=torch.argsort(q,dim=-1,descending=True,stable=True)
    return torch.zeros_like(q,dtype=torch.bool).scatter_(-1,order[...,:k],True)


def solve(a,b):
    block=torch.cat((torch.cat((a[0],-a[1]),-1),torch.cat((a[1],a[0]),-1)),-2)
    rhs=torch.cat(b,-2)
    out=torch.linalg.solve(block,rhs)
    n=a[0].shape[-1]
    result=out[...,:n,:],out[...,n:,:]
    residual=mm(a,result)
    relative=(power((residual[0]-b[0],residual[1]-b[1])).sum((-2,-1))/power(b).sum((-2,-1)).clamp_min(1e-20)).sqrt()
    if not torch.isfinite(out).all() or relative.max()>3e-3:
        raise RuntimeError(f'{DEVICE.type.upper()} FP32 solve failed residual gate: {float(relative.max()):.3g}; no precision/device fallback')
    return result


def receiver(a,variance,support=None):
    model=a if support is None else (a[0]*support,a[1]*support)
    gram=mm(adj(model),model)
    n=gram[0].shape[-1]
    return solve((gram[0]+variance*eye(n)[0],gram[1]),adj(model))


def constellation(mod):
    b=2 if mod=='QPSK' else 4
    if mod not in ('QPSK','16QAM'): raise ValueError('Qualified modulations: QPSK/16QAM')
    ids=torch.arange(2**b,device=DEVICE,dtype=torch.int32)
    shifts=torch.arange(b-1,-1,-1,device=DEVICE,dtype=torch.int32)
    labels=(ids[:,None]>>shifts)&1
    if b==2:
        points=((1-2*labels[:,0]).to(DTYPE)/math.sqrt(2),(1-2*labels[:,1]).to(DTYPE)/math.sqrt(2))
    else:
        levels=tensor([-3,-1,3,1])
        points=levels[labels[:,0]*2+labels[:,1]]/math.sqrt(10),levels[labels[:,2]*2+labels[:,3]]/math.sqrt(10)
    return points,labels


def symbols(bits,mod):
    points,_=constellation(mod)
    shifts=torch.arange(bits.shape[-1]-1,-1,-1,device=DEVICE,dtype=torch.int32)
    ids=(bits*(2**shifts)).sum(-1)
    return points[0][ids],points[1][ids]


def decode(x,mod):
    points,labels=constellation(mod)
    distances=(x[0][...,None]-points[0]).square()+(x[1][...,None]-points[1]).square()
    return labels[distances.argmin(-1)]


def unitarity(w):
    gram=mm(adj(w),w); n=gram[0].shape[-1]
    return float(torch.max(torch.sqrt(power((gram[0]-eye(n)[0],gram[1])))))
