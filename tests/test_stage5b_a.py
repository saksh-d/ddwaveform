import math
import numpy as np
import pytest
import torch
from src.stage5a import MultiscaleUnitary, matrix_loss
from src.stage5b_a import CartesianUnitary, coefficients

@pytest.mark.parametrize('radius',[0.,1e-12,1e-5,.4,math.pi,7.])
def test_gate_against_independent_exponential(radius):
    a=torch.tensor(radius*.6,dtype=torch.float64,requires_grad=True)
    b=torch.tensor(radius*.8,dtype=torch.float64,requires_grad=True)
    c,s=coefficients(a,b)
    gate=torch.stack([torch.stack([c+0j,s]),torch.stack([-s.conj(),c+0j])])
    z=-torch.complex(a,-b)
    zero=z*0
    k=torch.stack([torch.stack([zero,-z.conj()]),torch.stack([z,zero])])
    assert torch.allclose(gate,torch.matrix_exp(k),atol=3e-15,rtol=1e-14)
    assert torch.allclose(gate.mH@gate,torch.eye(2,dtype=torch.complex128),atol=3e-15)
    (gate.real.sum()+gate.imag.sum()).backward()
    assert torch.isfinite(a.grad) and torch.isfinite(b.grad)

@pytest.mark.parametrize('topology',['ascending','descending'])
def test_dense_independent_order_and_native_adjoint(topology):
    p=MultiscaleUnitary(8,4,topology,81)
    p.phase.data.copy_(torch.linspace(-2,2,8))
    m=CartesianUnitary.from_polar(p)
    dense=np.eye(8,dtype=complex)
    for order,theta,phi in zip(p.order,p.theta.detach(),p.phi.detach()):
        layer=np.eye(8,dtype=complex)
        for (j,k),t,f in zip(order.reshape(-1,2).tolist(),theta.numpy(),phi.numpy()):
            s=np.sin(t)*np.exp(1j*f)
            layer[np.ix_([j,k],[j,k])]=[[np.cos(t),s],[-s.conjugate(),np.cos(t)]]
        dense=layer@dense
    f=np.exp(-2j*np.pi*np.outer(np.arange(8),np.arange(8))/8)/np.sqrt(8)
    dense=f.conj().T@np.diag(np.exp(1j*p.phase.detach().numpy()))@dense
    assert np.max(abs(m.matrix().detach().numpy()-dense))<4e-15
    assert torch.allclose(m.matrix(),p.matrix(),atol=1e-15)
    x=torch.eye(8,dtype=torch.complex128)[:3]
    assert torch.allclose(m.apply_W_hermitian(m.apply_W(x)),x,atol=2e-15)
    assert sum(t.numel() for t in m.parameters())==sum(t.numel() for t in p.parameters())

@pytest.mark.parametrize('passes,scale',[(1,0.),(1,1e-9),(4,.2)])
@pytest.mark.parametrize('cartesian',[False,True])
def test_central_finite_differences(passes,scale,cartesian):
    p=MultiscaleUnitary(8,passes,seed=82)
    p.theta.data.mul_(scale)
    m=CartesianUnitary.from_polar(p) if cartesian else p
    target=MultiscaleUnitary(8,passes,seed=83).matrix().detach()
    matrix_loss(m.matrix(),target).backward()
    for parameter in m.parameters():
        for index in [0,parameter.numel()//2,parameter.numel()-1]:
            flat=parameter.data.view(-1)
            original=flat[index].item()
            flat[index]=original+1e-6
            plus=float(matrix_loss(m.matrix(),target).detach())
            flat[index]=original-1e-6
            minus=float(matrix_loss(m.matrix(),target).detach())
            flat[index]=original
            actual=parameter.grad.flatten()[index].item()
            assert abs((plus-minus)/2e-6-actual)<2e-8+2e-4*abs(actual)
