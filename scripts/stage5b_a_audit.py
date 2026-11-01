"""Bounded gradient and local geometry checks; no fitting."""
import copy
import json
from pathlib import Path
import numpy as np
import torch
from torch.func import functional_call, jvp
from src.stage5a import MultiscaleUnitary, matrix_loss
from src.stage5b_a import CartesianUnitary
from scripts.stage5b_a import OUT, dump

class MatrixView(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model=model
    def forward(self):
        return torch.view_as_real(self.model.matrix()).flatten()

def geometry(model, dense=False):
    wrapper=MatrixView(model)
    params=dict(wrapper.named_parameters())
    def function(p):
        return functional_call(wrapper,p,())
    if dense:
        jac=torch.func.jacfwd(function)(params)
        array=torch.cat([a.reshape(2*model.n**2,-1) for a in jac.values()],dim=1)
        values=torch.linalg.svdvals(array)
        return dict(n=model.n,rank_threshold=1e-7,rank=int((values>1e-7).sum()),
            singular_values=values.tolist(),parameters=array.shape[1])
    rows=[]
    for name,p in params.items():
        g=torch.Generator().manual_seed(981)
        tangent={k:torch.zeros_like(v) for k,v in params.items()}
        tangent[name]=torch.randn(p.shape,dtype=p.dtype,generator=g)
        tangent[name]/=tangent[name].norm()
        _,derivative=jvp(function,(params,),(tangent,))
        rows.append(dict(parameter=name,directional_jacobian_norm=float(derivative.detach().norm())))
    return rows

def fd(model,target,label):
    model.zero_grad()
    matrix_loss(model.matrix(),target).backward()
    rows=[]
    for name,p in model.named_parameters():
        for index in [0,p.numel()//2,p.numel()-1]:
            flat=p.data.view(-1)
            original=flat[index].item()
            flat[index]=original+1e-6
            plus=float(matrix_loss(model.matrix(),target).detach())
            flat[index]=original-1e-6
            minus=float(matrix_loss(model.matrix(),target).detach())
            flat[index]=original
            automatic=p.grad.flatten()[index].item()
            numerical=(plus-minus)/2e-6
            rows.append(dict(label=label,parameter=name,index=index,autograd=automatic,
                central=numerical,absolute_error=abs(automatic-numerical),
                relative_error_nonzero=abs(automatic-numerical)/abs(automatic) if abs(automatic)>1e-7 else None,
                passed=abs(automatic-numerical)<2e-8+2e-4*abs(automatic)))
    return rows

def main():
    torch.set_num_threads(1)
    rows=[]; geometries=[]
    for n,passes,mode in [(8,1,'random'),(8,1,'identity'),(8,4,'random'),(8,4,'near'),(64,4,'random'),(64,4,'identity')]:
        polar=MultiscaleUnitary(n,passes,seed=97)
        polar.phase.data.copy_(torch.linspace(-2,2,n))
        if mode in ['identity','near']:
            polar.theta.data.mul_(0 if mode=='identity' else 1e-9)
        target=MultiscaleUnitary(n,passes,seed=98).matrix().detach()
        for arm,model in [('A',polar),('B',CartesianUnitary.from_polar(polar))]:
            label=f'N{n}_p{passes}_{mode}_{arm}'
            rows+=fd(model,target,label)
            geometries.append(dict(label=label,geometry=geometry(model,n==8)))
    # Frozen converged/failed representatives, selected before reading outcomes.
    for arm in ['A','B']:
        for category in ['0.001','cold']:
            model=MultiscaleUnitary(64,4,seed=1)
            if arm=='B': model=CartesianUnitary.from_polar(model)
            key=f'ascending_p4_s0_{category}'
            model.load_state_dict(torch.load(OUT/'AB'/f'{key}_{arm}_best.pt',weights_only=True))
            target=torch.load(OUT/'AB'/f'{key}_input.pt',weights_only=True)['target']
            rows+=fd(model,target,f'checkpoint_{key}_{arm}')
    dump(OUT/'gradient_audit.json',dict(step=1e-6,absolute_tolerance=2e-8,
        relative_tolerance=2e-4,relative_reporting_floor=1e-7,checks=rows))
    dump(OUT/'geometry.json',geometries)
    print('FD',len(rows),'passed',sum(r['passed'] for r in rows),'max_abs',max(r['absolute_error'] for r in rows))

if __name__=='__main__': main()
