"""Stage 5B-A: identical mixer family in regular local coordinates."""
import copy
import math
import time

import torch
from .stage5a import MultiscaleUnitary, pair_action, matrix_loss


def coefficients(a, b):
    """Analytic in q=a²+b² near zero; avoid sqrt's undefined derivative at zero."""
    q = a.square() + b.square()
    r = q.clamp_min(1e-8).sqrt()
    c = torch.where(q < 1e-8, 1-q/2+q.square()/24-q.pow(3)/720, r.cos())
    sinc = torch.where(q < 1e-8, 1-q/6+q.square()/120-q.pow(3)/5040, r.sin()/r)
    # Native gate is [[c,s],[-s*,c]]; candidate z=-conj(a+ib).
    return c, torch.complex(a, b)*sinc


class CartesianUnitary(MultiscaleUnitary):
    """theta/phi fields store Cartesian a/b, not polar angles; no extra parameters."""
    @classmethod
    def from_polar(cls, polar):
        model = cls(polar.n, polar.passes, polar.topology)
        with torch.no_grad():
            model.theta.copy_(polar.theta*polar.phi.cos())
            model.phi.copy_(polar.theta*polar.phi.sin())
            model.phase.copy_(polar.phase)
        return model

    def initialize(self, *args, **kwargs):
        raise RuntimeError('Convert an initialized polar model explicitly with from_polar')

    def compile(self):
        raise NotImplementedError('Stage5B-A fits only; no deployment cache')

    def apply_W(self, x):
        c, s = coefficients(self.theta, self.phi)
        for i in range(len(self.order)):
            x = pair_action(x, self.order[i], self.inverse[i], c[i], s[i])
        return torch.fft.ifft(x*torch.exp(1j*self.phase), dim=-1, norm='ortho')

    def apply_W_hermitian(self, x):
        x = torch.fft.fft(x, dim=-1, norm='ortho')*torch.exp(-1j*self.phase)
        c, s = coefficients(self.theta, self.phi)
        for i in reversed(range(len(self.order))):
            x = pair_action(x, self.order[i], self.inverse[i], c[i], s[i], True)
        return x


def local_lbfgs(model, target, budget=500):
    """One solver intervention: omit Adam displacement; strict tolerances, hard eval cap."""
    start = time.perf_counter()
    best, best_loss, trace = copy.deepcopy(model.state_dict()), float('inf'), []
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1., max_iter=budget,
        max_eval=budget, tolerance_grad=1e-13, tolerance_change=1e-19,
        line_search_fn='strong_wolfe', history_size=30)
    class BudgetReached(Exception):
        pass
    def closure():
        nonlocal best, best_loss
        if len(trace) >= budget:
            raise BudgetReached()
        optimizer.zero_grad()
        loss = matrix_loss(model.matrix(), target)
        loss.backward()
        value = float(loss.detach())
        grad = float(sum(p.grad.square().sum() for p in model.parameters()).sqrt())
        if not math.isfinite(value+grad):
            raise RuntimeError('nonfinite fitting state')
        if value < best_loss:
            best_loss, best = value, copy.deepcopy(model.state_dict())
        trace.append(dict(evaluation=len(trace), phase='lbfgs_only', loss=value,
                          gradient_norm=grad, best_loss=best_loss))
        return loss
    try:
        optimizer.step(closure)
        if len(trace) < budget:
            closure()
    except BudgetReached:
        pass
    model.load_state_dict(best)
    return model, trace, dict(initial_loss=trace[0]['loss'], final_loss=trace[-1]['loss'],
        best_loss=best_loss, evaluations=len(trace), seconds=time.perf_counter()-start)
