"""Independent artifact accounting: reload every selected matrix, no re-fitting."""
import json
from pathlib import Path
import hashlib
import torch
from src.stage5a import MultiscaleUnitary
from src.stage5b_a import CartesianUnitary
from scripts.stage5b_a import OUT, dump


def main():
    torch.set_num_threads(1)
    checks=[]; planted=[]; prefixes=[]
    for armset in ['AB','C']:
        run=OUT/armset
        if not (run/'complete.json').exists(): continue
        rows=json.loads((run/'results.json').read_text())
        expected=108 if armset=='AB' else 54
        assert len(rows)==expected and len({(r['key'],r['arm']) for r in rows})==expected
        for row in rows:
            assert row['numerical_failure'] is None
            key,arm=row['key'],row['arm']
            payload=torch.load(run/f'{key}_input.pt',weights_only=True)
            target=payload['target']
            polar=MultiscaleUnitary(64,row['passes'],row['topology'])
            polar.load_state_dict(payload['truth'])
            if arm=='A' or armset=='C':
                alt=CartesianUnitary.from_polar(polar).matrix().detach()
                planted.append(dict(key=key,armset=armset,
                    polar_error=float((polar.matrix().detach()-target).norm()/8),
                    converted_error=float((alt-target).norm()/8)))
            model=polar if arm=='A' else CartesianUnitary.from_polar(polar)
            model.load_state_dict(torch.load(run/f'{key}_{arm}_best.pt',weights_only=True))
            error=float((model.matrix().detach()-target).norm()/8)
            trace=json.loads((run/f'{key}_{arm}_trace.json').read_text())
            assert abs(error-row['error'])<1e-13
            assert row['success']==(error<1e-4)
            assert len(trace)==row['evaluations']
            assert abs(min(t['loss'] for t in trace)-error**2)<1e-14
            assert all(t['best_loss']==min(x['loss'] for x in trace[:i+1]) for i,t in enumerate(trace))
            assert row['parameters']==64+384*row['passes']
            assert row['generation_seed']!=row['fit_seed']
            checks.append(dict(key=key,arm=arm,error=error,success=error<1e-4))
        if armset=='AB':
            for key in sorted({r['key'] for r in rows}):
                traces={a:json.loads((run/f'{key}_{a}_trace.json').read_text()) for a in ['A','B']}
                n=min(map(len,traces.values()))
                prefixes.append(dict(key=key,evaluations=n,**{a:min(t['loss'] for t in v[:n])**.5 for a,v in traces.items()}))
    dump(OUT/'verification.json',dict(checks=checks,planted=planted,common_prefix=prefixes,
        note='Independent artifact accounting; independent SciPy gate/dense implementation supplied by reviewer.'))
    print('verified',len(checks),'fits; planted max',max(r['converted_error'] for r in planted))

if __name__=='__main__': main()
