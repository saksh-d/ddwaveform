"""Execute local notebook review matrix; never write outputs to tracked notebook.

Run from project root with conda pytorch. Cases use only this checkout's local/.
"""
import argparse
import base64
import json
from pathlib import Path
import time
import nbformat
from nbclient import NotebookClient


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cases',nargs='+',default=['replay64','fresh64','train64','fresh64_changed','train64_changed','replay128','train128','replay256','train256','fresh256_changed','recalibrate64','reload64','cfo','cfo_changed','cfo_train_rejected'])
    parser.add_argument('--out',default='local/qualification')
    args=parser.parse_args(); root=Path.cwd().resolve(); out=(root/args.out).resolve()
    if not out.is_relative_to(root/'local'): raise ValueError('Qualification output must stay in this worktree local/')
    out.mkdir(parents=True,exist_ok=True)
    template=nbformat.read(root/'notebooks/results.ipynb',as_version=4); results=[]
    for case in args.cases:
        nb=nbformat.from_dict(json.loads(json.dumps(template)))
        cell=next(c for c in nb.cells if 'configuration' in c.metadata.get('tags',[]))
        source=cell.source
        n=128 if '128' in case else 256 if '256' in case else 64
        source=source.replace('N = 64 ',f'N = {n} ')
        is_train=case.startswith('train'); is_cfo=case.startswith('cfo'); fresh=not case.startswith('replay')
        if is_train or case=='cfo_train_rejected': source=source.replace('MODE = "load_existing"','MODE = "train_new"')
        if fresh: source=source.replace('EVIDENCE_MODE = "saved_replay"','EVIDENCE_MODE = "fresh"')
        if case=='recalibrate64': source=source.replace('SUPPORT_POLICY = "frozen"','SUPPORT_POLICY = "recalibrate"').replace('K_FRACTION = 3 / 8','K_FRACTION = 1 / 4')
        if is_cfo: source=source.replace('SCENARIO = "dd_v1"','SCENARIO = "residual_cfo"')
        # Execute default train64 and CFO budgets unchanged; scaling/stress checks bounded.
        if n>64:
            source+='\nTRAINING.update(iterations=3,validate_every=1,batch_channels=3,validation_channels=3)\n' if is_train else ''
            if fresh: source+='\nEVALUATION.update(max_channels=2,min_channels=2,blocks_per_channel=1,snrs_db=(0.0,12.0),diagnostic_atoms=3,diagnostic_channels=3)\n'
        if case.endswith('_changed'):
            if is_cfo:
                source+='\nMODEL.update(cfo_range=(0.2,0.3))\nEVALUATION.update(seed=909,snrs_db=(1.0,11.0),max_channels=4,min_channels=2)\n'
            else:
                source+='\nMODEL.update(paths=2,delay_range=(0.1,1.2),doppler_range=(-0.2,0.3))\nEVALUATION.update(seed=909,snrs_db=(1.0,11.0),max_channels=4,min_channels=2)\n'
                if is_train: source+='\nTRAINING.update(model=dict(MODEL),validation_model=dict(MODEL),seed=707,validation_seed=808,iterations=5,validate_every=1)\n'
        if case=='train64': source+=f'\nCHECKPOINT = Path({str(out / "default_train64.npz")!r})\n'
        if case=='reload64':
            source+=f'\nCHECKPOINT = Path({str(out / "default_train64.npz")!r})\nTRAINING = None\nDISPLAY = dict(basis_columns=(0,1),example_columns=(0,),snr_max_db=25.)\n'
        cell.source=source
        started=time.perf_counter(); error=None
        client=NotebookClient(nb,timeout=300,kernel_name='python3',resources={'metadata':{'path':str(root)}})
        try: client.execute()
        except Exception as exc: error=str(exc)
        destination=out/(case+'.ipynb'); nbformat.write(nb,destination)
        images=[]
        for ci,c in enumerate(nb.cells):
            for oi,item in enumerate(c.get('outputs',[])):
                png=item.get('data',{}).get('image/png')
                if png:
                    p=out/f'{case}_{ci}_{oi}.png';p.write_bytes(base64.b64decode(png));images.append(p.name)
        item=dict(case=case,seconds=time.perf_counter()-started,error=error,inline_pngs=len(images),images=images,executed_notebook=str(destination))
        results.append(item); (out/'execution.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps({k:v for k,v in item.items() if k not in ('images','error')}|{'error':error[-1000:] if error else None}),flush=True)
        if error and case!='cfo_train_rejected': raise RuntimeError(f'{case} failed: {error}')
        if case=='cfo_train_rejected' and (not error or 'Residual CFO train_new unsupported' not in error or images): raise RuntimeError('Expected fail-before-training not observed')


if __name__=='__main__': main()
