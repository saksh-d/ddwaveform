"""Frozen Stage5B-A recovery suite; append-only run files, no historical writes."""
import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import subprocess

import torch
from src.stage5a import MultiscaleUnitary, fit_matrix, matrix_loss, role_seed
from src.stage5b_a import CartesianUnitary, local_lbfgs

OUT = Path('outputs/stage5b_a')

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

def seed(*parts):
    return role_seed('stage5b-a', *parts)

def initial(topology, passes, index, category):
    truth = MultiscaleUnitary(64, passes, topology, seed('generation', topology, passes, index))
    target = truth.matrix().detach()
    fit_seed = seed('fit', topology, passes, index, category)
    if category == 'cold':
        model = MultiscaleUnitary(64, passes, topology, fit_seed)
        model.initialize(target)
    else:
        model = copy.deepcopy(truth)
        generator = torch.Generator().manual_seed(fit_seed)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(float(category)*torch.randn(p.shape, dtype=p.dtype, generator=generator))
    return truth, target, model, fit_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['AB','C'], default='AB')
    args = parser.parse_args()
    torch.set_num_threads(1)
    run = OUT / args.arm
    run.mkdir(exist_ok=False)
    for path in ['src/stage5a.py','src/stage5b_a.py','scripts/stage5b_a.py','reports/stage5b_a_protocol.json']:
        shutil.copy(path, run / Path(path).name)
    dump(run/'environment.json', dict(git=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        torch=torch.__version__, threads=torch.get_num_threads(), precision='float64/complex128'))
    rows=[]
    for topology, passes, index, category in itertools.product(['ascending','descending'],[1,2,4],range(3),['0.001','0.03','cold']):
        truth, target, polar, fit_seed = initial(topology, passes, index, category)
        key=f'{topology}_p{passes}_s{index}_{category}'
        torch.save(dict(target=target, truth=truth.state_dict(), polar_initial=polar.state_dict()),run/f'{key}_input.pt')
        for arm in (['A','B'] if args.arm=='AB' else ['C']):
            model=copy.deepcopy(polar) if arm=='A' else CartesianUnitary.from_polar(polar)
            conversion_error=float((model.matrix()-polar.matrix()).abs().max())
            try:
                if arm=='C':
                    model,trace,stats=local_lbfgs(model,target,budget=500)
                else:
                    model,trace,stats=fit_matrix(model,target,adam_steps=250,lbfgs_steps=200)
                w=model.matrix().detach()
                row=dict(key=key,arm=arm,topology=topology,passes=passes,index=index,category=category,
                    generation_seed=seed('generation',topology,passes,index),fit_seed=fit_seed,
                    initial_error=stats['initial_loss']**.5,error=stats['best_loss']**.5,
                    success=stats['best_loss']<1e-8,conversion_error=conversion_error,
                    parameters=sum(p.numel() for p in model.parameters()),
                    unitarity=float(torch.linalg.matrix_norm(w.mH@w-torch.eye(64))),
                    numerical_failure=None,**stats)
                dump(run/f'{key}_{arm}_trace.json',trace)
                torch.save(model.state_dict(),run/f'{key}_{arm}_best.pt')
            except RuntimeError as exc:
                row=dict(key=key,arm=arm,category=category,numerical_failure=str(exc),success=False)
            rows.append(row)
            dump(run/'results.json',rows)
            print(key,arm,row.get('error'),row.get('evaluations'),flush=True)
    dump(run/'complete.json',dict(fits=len(rows), expected=108 if args.arm=='AB' else 54,
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in run.iterdir() if p.is_file()}))

if __name__=='__main__':
    main()
