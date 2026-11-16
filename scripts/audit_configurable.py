"""Audit local executed notebooks, archives, source hashes and checkpoint replay."""
import json
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from waveforms.artifacts import sha,verify


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--qualification',default='local/qualification-release')
    args=parser.parse_args();qualification=(ROOT/args.qualification).resolve()
    if not qualification.is_relative_to(ROOT/'local'): raise ValueError('Audit inputs must stay in worktree local/')
    ledger=json.loads((qualification/'execution.json').read_text()); results=[]; runs={}
    for item in ledger:
        nb=json.loads(Path(item['executed_notebook']).read_text())
        if item['case']=='cfo_train_rejected':
            assert item['error'] and item['inline_pngs']==0
            results.append(dict(case=item['case'],expected_rejection=True)); continue
        assert item['error'] is None
        final=next(o['data'][key] for o in nb['cells'][-1]['outputs'] for key in ('application/vnd.waveforms.run+json','application/json') if key in o.get('data',{}))
        run=Path(final.get('optional_local_archive',final.get('archive'))); runs[item['case']]=run
        assert run.resolve().is_relative_to(ROOT/'local')
        manifest=json.loads((run/'run.json').read_text())
        for name,digest in manifest['source_sha256'].items(): assert sha(ROOT/'waveforms'/name)==digest,(item['case'],name,'source drift')
        for name,digest in manifest['files_sha256'].items(): assert sha(run/name)==digest,(item['case'],name,'archive drift')
        pngs=list(run.glob('*.png'))
        assert len(pngs)==item['inline_pngs']
        assert {sha(p) for p in pngs}=={sha(qualification/p) for p in item['images']}
        for p in pngs:
            with Image.open(p) as im: assert abs(im.info['dpi'][0]-300)<.1
        detail=dict(case=item['case'],seconds=item['seconds'],inline_pngs=len(pngs),run=str(run),
                    environment=manifest['environment'])
        if (run/'spectral_coupling.npz').exists():
            with np.load(run/'spectral_coupling.npz') as a:
                detail.update(unitarity_max=float(a['unitarity_max']),bridge_max=float(a['bridge_max']),group_sizes=np.bincount(a['labels']).tolist())
        if manifest.get('selected_learned',{}).get('steps_executed') is not None:
            selected=manifest['selected_learned']; detail.update(steps=selected['steps_executed'],selected_step=selected['selected_step'],training_seconds=selected['seconds'],validation_leakage=selected['validation_leakage'])
        results.append(detail)
    checkpoint_exact = None
    if {'train64','reload64'} <= runs.keys():
        with np.load(runs['train64']/'matrices.npz') as a, np.load(runs['reload64']/'matrices.npz') as b:
            for key in a.files: np.testing.assert_array_equal(a[key],b[key])
        a=pd.read_csv(runs['train64']/'ber.csv'); b=pd.read_csv(runs['reload64']/'ber.csv')
        pd.testing.assert_frame_equal(a.drop(columns='seconds'),b.drop(columns='seconds'))
        for p in runs['train64'].glob('*dB.npz'):
            with np.load(p) as a, np.load(runs['reload64']/p.name) as b:
                for key in a.files: np.testing.assert_array_equal(a[key],b[key])
        checkpoint_exact = True
    verified={}
    for name,config in [('artifacts','artifacts.json'),('scaling-artifacts','scaling_artifacts.json')]:
        verified[name]=verify(ROOT/'local'/name,json.loads((ROOT/'configs'/config).read_text()))
    notebook=json.loads((ROOT/'Waveform_Exploration.ipynb').read_text())
    assert all(c.get('outputs',[])==[] and c.get('execution_count') is None for c in notebook['cells'] if c['cell_type']=='code')
    summary=dict(results=results,total_notebook_seconds=sum(x['seconds'] for x in ledger),inline_pngs=sum(x['inline_pngs'] for x in ledger),
                 checkpoint_reload_matrices_counts_BER_exact=checkpoint_exact,verified_inputs=verified,
                 source_notebook_output_free=True,sources={str(p.relative_to(ROOT)):sha(p) for p in [*sorted((ROOT/'waveforms').glob('*.py')),ROOT/'Waveform_Exploration.ipynb']})
    (qualification/'audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
