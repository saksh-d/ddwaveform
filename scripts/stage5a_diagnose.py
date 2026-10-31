"""Bounded optimizer-basin and parameter-count diagnostics; no final-bank access."""
from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from scripts.stage5a import OUT, ROOT, sha, write_csv, write_json
from src.stage5a import MultiscaleUnitary, fit_matrix, role_seed
from src.waveforms import random_unitary


def main():
    torch.set_num_threads(1)
    if (OUT / "optimizer_diagnosis.csv").exists():
        raise RuntimeError("preserve completed diagnostic")
    protocol=ROOT / "reports/stage5a_diagnosis_protocol.json"
    write_json(OUT / "diagnosis_protocol.json",json.loads(protocol.read_text()))
    rows=[]
    for topology,passes in itertools.product(("ascending","descending"),(1,2,4)):
        payload=torch.load(OUT / "checkpoints" / f"SYNTHETIC_{topology}_p{passes}_r0.pt",weights_only=True)
        truth=MultiscaleUnitary(64,passes,topology)
        truth.load_state_dict(payload["target"])
        target=truth.matrix().detach()
        exact_loss=float((truth.matrix().detach()-target).abs().square().sum()/64)
        local=copy.deepcopy(truth)
        g=torch.Generator().manual_seed(role_seed("planted-neighborhood",topology,passes))
        with torch.no_grad():
            for p in local.parameters():
                p.add_(1e-3*torch.randn(p.shape,generator=g,dtype=p.dtype))
        local,trace,stats=fit_matrix(local,target,adam_steps=250,lbfgs_steps=200)
        rows.append({"topology":topology,"passes":passes,"exact_planted_loss":exact_loss,
                     **stats,"local_qualified":stats["best_loss"]<1e-8})
        write_csv(OUT / "traces" / f"LOCAL_{topology}_p{passes}.csv",trace)
        torch.save(local.state_dict(),OUT / "checkpoints" / f"LOCAL_{topology}_p{passes}.pt")
        print(rows[-1],flush=True)
    write_csv(OUT / "optimizer_diagnosis.csv",rows)
    scaling=[]
    for n in (128,256):
        model=MultiscaleUnitary(n,1,seed=role_seed("action-scaling",n))
        w=model.matrix().detach()
        action=model.compile()
        x=random_unitary(n,role_seed("action-scaling-probe",n))[:3]
        scaling.append({"n":n,"forward_max_error":float((action.apply_W(x)-x@w.T).abs().max()),
                        "adjoint_max_error":float((action.apply_W_hermitian(x)-x@w.conj()).abs().max()),
                        "roundtrip_max_error":float((action.apply_W_hermitian(action.apply_W(x))-x).abs().max()),
                        "unitarity_frobenius":float(torch.linalg.matrix_norm(w.mH@w-torch.eye(n))),
                        "learned_target_fitted":False})
    write_csv(OUT / "action_scaling_checks.csv",scaling)
    ranks=[]
    for passes in (1,2,4):
        model=MultiscaleUnitary(8,passes,seed=role_seed("jacobian",passes))
        parameters=list(model.parameters())
        vector=torch.cat([p.detach().flatten() for p in parameters])
        def matrix_at(v):
            position=0
            with torch.no_grad():
                for p in parameters:
                    count=p.numel()
                    p.copy_(v[position:position+count].reshape(p.shape))
                    position+=count
            return torch.view_as_real(model.matrix().detach()).flatten().numpy()
        step=1e-6
        jac=[]
        for j in range(len(vector)):
            delta=torch.zeros_like(vector)
            delta[j]=step
            jac.append((matrix_at(vector+delta)-matrix_at(vector-delta))/(2*step))
        singular=np.linalg.svd(np.stack(jac,axis=-1),compute_uv=False)
        ranks.append({"n":8,"passes":passes,"nominal_parameters":len(vector),
                      "local_rank_threshold_1e-7":int((singular>1e-7).sum()),
                      "singular_values":singular.tolist()})
    write_json(OUT / "local_parameter_ranks.json",ranks)
    write_json(OUT / "diagnosis_complete.json",{"protocol_sha256":sha(protocol),
                                               "source_sha256":sha(__file__),"cases":len(rows)})


if __name__=="__main__":
    main()
